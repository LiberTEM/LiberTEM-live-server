import os
os.environ['OMP_NUM_THREADS'] = '1'  # must come before numba (because of OMP stuff?)

import time
import json
import asyncio
import typing
from typing import Coroutine, Dict, Any, Set, Tuple, Callable, AsyncGenerator
from collections import OrderedDict
from contextlib import asynccontextmanager
import math
import logging
import enum
import copy

import numba
import numpy as np
import uuid
import websockets
import click
import tomllib

from websockets.legacy.server import WebSocketServerProtocol
from websockets.legacy.client import WebSocketClientProtocol

from libertem import masks
from libertem.udf import UDF
# from libertem.udf.sum import SumUDF
# from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.executor.pipelined import PipelinedExecutor
from libertem_live.api import LiveContext
from libertem.api import Context
from libertem_live.udf.monitor import (
    PartitionMonitorUDF
)
from libertem.common.tracing import maybe_setup_tracing

from libertem.udf.base import UDFResults
from libertem.common.async_utils import sync_to_async
# from libertem_icom.udf.icom import ICoMUDF
from libertem.udf.com import CoMUDF
from libertem.udf.raw import PickCorrectedUDF

from result_codecs import BsLz4, LossyU16


log = logging.getLogger(__name__)


@enum.unique
class Encoding(enum.StrEnum):
    DeltaBsLZ4 = "bslz4"
    LossyU16 = "lossy-u16-bslz4"


T = typing.TypeVar('T')


class EncodedResult:
    def __init__(
        self,
        compressed_data: memoryview,
        bbox: typing.Tuple[int, int, int, int],
        full_shape: typing.Tuple[int, int],
        delta_shape: typing.Tuple[int, int],
        dtype: str,
        encoding: str,
        encoding_meta: Dict[str, Any],
        channel_name: str,
        udf_name: str,
    ):
        self.compressed_data = compressed_data
        self.bbox = bbox
        self.full_shape = full_shape
        self.delta_shape = delta_shape
        self.dtype = dtype
        self.channel_name = channel_name
        self.udf_name = udf_name
        self.encoding = encoding
        self.encoding_meta = encoding_meta

    def is_empty(self):
        return len(self.compressed_data) == 0


class Closed(Exception):
    pass


class LatestContainer(typing.Generic[T]):
    """
    Producer/consumer container that has at most one item inside.

    Used here to make sure always the latest partial results are sent over.
    """

    def __init__(self) -> None:
        self._item: typing.Optional[T] = None
        self._cond = asyncio.Condition()
        self._closed = False

    async def take(self) -> T:
        """
        Wait until an item is available, and then return it, emptying the container.

        If the container is closed, this can be called successfully one last
        time, to retrieve the last item.

        Raises `Closed` if the container is closed and empty.
        """
        if self._closed and self._item is None:
            raise Closed()
        async with self._cond:
            await self._cond.wait_for(lambda: self._item is not None or self._closed)
            if self._closed and self._item is None:
                raise Closed()
            item = self._item
            self._item = None

        # this is enforced by the `wait_for` above, so this assert should never fail:
        assert item is not None
        return item

    async def put(self, item: T):
        """
        Put an item into the container, or replaces the one that is currently in
        there. Wakes up anyone waiting to `take`.

        Raises `Closed` if the container is closed.
        """
        async with self._cond:
            if self._closed:
                raise Closed()
            self._item = item
            self._cond.notify_all()

    async def close(self):
        """
        Mark this container as closed.
        """
        async with self._cond:
            self._closed = True
            self._cond.notify_all()


class ResultSampler:
    """
    This class holds references to all current websocket clients,
    and starts per-client loops that send out results as fast as
    the client can handle.
    """
    def __init__(self, parameters: "ParameterContainer", udfs: "UDFContainer"):
        self.clients: Set[WebSocketClientProtocol] = set()
        self.client_queues: Dict[
            WebSocketClientProtocol,
            asyncio.Queue[Tuple[str, LatestContainer[UDFResults]]]
        ] = {}
        self._sampler_tasks: Dict[WebSocketClientProtocol, asyncio.Task] = {}
        self._min_delta = 1/60.0  # should this be a parameter?
        self._parameters = parameters
        self._udfs = udfs

    @asynccontextmanager
    async def handle_acquisition(self, acq_id: str) -> AsyncGenerator[
        Callable[[UDFResults], Coroutine[None, None, None]], None
    ]:
        # inform all our per-client sampler loops that a new acquisition was started,
        # and give them access to a new `LatestContainer`
        to_clients: Dict[WebSocketClientProtocol, LatestContainer] = {
            client: LatestContainer()
            for client in self.clients
        }
        for client, lc in to_clients.items():
            log.info("sending some LCs to sampler loops...")
            await self.client_queues[client].put((acq_id, lc))

        # an inner helper that is given to the caller to push results to
        # clients:
        async def _result_sink(partial_results: UDFResults):
            # result_copy = copy.deepcopy(partial_results)
            result_copy = partial_results

            if False:
                # XXX
                import cloudpickle
                a = cloudpickle.dumps(result_copy)
                print(len(a))
            for client in to_clients:
                await to_clients[client].put(result_copy)
                await self._check_task_status()

        yield _result_sink
        for lc in to_clients.values():
            await lc.close()

    async def _check_task_status(self):
        for client, task in self._sampler_tasks.items():
            if task.done():
                exc = task.exception()
                if exc is not None:
                    raise exc

    async def add_client(self, client: WebSocketClientProtocol):
        self.clients.add(client)
        self.client_queues[client] = asyncio.Queue()
        await self._spawn_task(client)

    async def remove_client(self, client: WebSocketClientProtocol):
        self.clients.remove(client)
        del self.client_queues[client]
        await self._cancel_task(client)

    async def _spawn_task(self, client: WebSocketClientProtocol):
        self._sampler_tasks[client] = asyncio.ensure_future(self.sampler_loop(client))

    async def _cancel_task(self, client: WebSocketClientProtocol):
        task = self._sampler_tasks[client]
        del self._sampler_tasks[client]
        task.cancel()

    async def send_initial(self, websocket: WebSocketClientProtocol):
        await websocket.send(json.dumps({
            'event': 'UPDATE_PARAMS',
            'parameters': self._parameters.get_parameters(),
        }))
        # for acq_id in self.results.keys():
        #     result = self.results.get_result(acq_id)
        #     await self.handle_partial_result(
        #         result=result,
        #         acq_id=acq_id,
        #         previous_results=None,
        #         client=websocket,
        #     )

    async def make_deltas(
        self, partial_results: UDFResults, previous_results: typing.Optional[UDFResults]
    ) -> np.ndarray:
        deltas = []
        udf_names = list(self._udfs.get_udfs().keys())
        for idx in range(len(partial_results.buffers)):
            udf_name = udf_names[idx]
            for channel_name in partial_results.buffers[idx].keys():
                data = partial_results.buffers[idx][channel_name].data
                # filter out non-2d result channels:
                if len(data.shape) != 2:
                    continue
                if previous_results is None:
                    data_previous = np.zeros_like(data)
                else:
                    data_previous = previous_results.buffers[idx][channel_name].data

                delta = data - data_previous
                deltas.append({
                    'delta': delta,
                    'udf_name': udf_name,
                    'channel_name': channel_name,
                })
        return deltas

    async def encode_result(
        self, delta: np.ndarray, udf_name: str, channel_name: str
    ) -> EncodedResult:
        """
        Slice `delta` to its non-zero region and compress that. Returns the information
        needed to reconstruct the the full result.
        """
        loop = asyncio.get_running_loop()
        nonzero_mask = await loop.run_in_executor(None, lambda: ~np.isclose(0, delta))

        if np.count_nonzero(nonzero_mask) == 0:
            log.debug("zero-delta update, skipping")
            # skip this update if it is all-zero
            return EncodedResult(
                compressed_data=memoryview(b""),
                bbox=(0, 0, 0, 0),
                full_shape=delta.shape,
                delta_shape=(0, 0),
                dtype=delta.dtype,
                channel_name=channel_name,
                udf_name=udf_name,
                encoding=Encoding.DeltaBsLZ4,
                encoding_meta={
                    "shape": (0, 0),
                    "dtype": "uint8",
                },
            )

        bbox = get_bbox(delta)
        ymin, ymax, xmin, xmax = bbox
        delta_for_blit = delta[ymin:ymax + 1, xmin:xmax + 1]

        # FIXME: lossy/lossless selection in UDF somehow? be smart about it? for large bboxes?
        # or for large updates byte-wise?
        if delta.dtype.kind == 'f' and delta_for_blit.nbytes > 1024*64:
            encoding = Encoding.LossyU16
            codec = LossyU16()
            log.debug(f"encoding channel {channel_name} of udf {udf_name} as lossy")
        else:
            encoding = Encoding.DeltaBsLZ4
            codec = BsLz4()
        compressed, encoding_meta = await sync_to_async(lambda: codec.encode(delta_for_blit))

        log.debug("encode_result: bbox=%r", bbox)

        return EncodedResult(
            compressed_data=memoryview(compressed),
            bbox=bbox,
            full_shape=delta.shape,
            delta_shape=delta_for_blit.shape,
            dtype=delta.dtype,
            channel_name=channel_name,
            udf_name=udf_name,
            encoding=encoding,
            encoding_meta=encoding_meta,
        )

    async def handle_partial_result(
        self,
        client: WebSocketClientProtocol,
        previous_results: typing.Optional[UDFResults],
        partial_results: UDFResults,
        acq_id: str,
    ):
        deltas = await self.make_deltas(partial_results, previous_results)

        delta_results: typing.List[EncodedResult] = []
        for delta in deltas:
            delta_results.append(
                await self.encode_result(
                    delta['delta'],
                    delta['udf_name'],
                    delta['channel_name']
                )
            )
        header_msg = json.dumps({
            "event": "RESULT",
            "id": acq_id,
            "timestamp": time.time(),
            "channels": [
                {
                    "bbox": result.bbox,
                    "full_shape": result.full_shape,
                    "delta_shape": result.delta_shape,
                    "dtype": str(result.dtype),
                    "encoding": result.encoding,
                    "encoding_meta": result.encoding_meta,
                    "channel_name": result.channel_name,
                    "udf_name": result.udf_name,
                }
                for result in delta_results
            ],
        }, indent=4)

        # log.info("header_msg: %s", header_msg)

        await client.send(header_msg)
        for result in delta_results:
            await client.send(result.compressed_data)

    async def sampler_loop(self, client: WebSocketClientProtocol):
        """
        This loop is spawned as a task per-client, and takes care of
        synchronizing the result state.

        It also means that we have to take care of sampling the current state at
        the beginning of the connection, so we have an idea of what the client
        knows about.
        """
        log.info(f"sampler_loop started for {client.remote_address}")
        await self.send_initial(client)
        last_update = time.monotonic()
        while True:
            log.info(
                f"sampler_loop waiting for next acquisition for client {client.remote_address}"
            )
            # first, let's get the `LatestContainer` for the current acquisition:
            acq_id, lc = await self.client_queues[client].get()

            log.info(f"got our lc for {acq_id} / {client.remote_address}")

            previous_results = None
            while True:
                # if we are sending faster than the results become available,
                # we will wait here for the next result, otherwise we'll simply take
                # the latest available (the other end might have called `put` multiple times
                # in the mean time):
                since_last_update = time.monotonic() - last_update
                # make sure we respect the maximum update rate:
                if since_last_update < self._min_delta:
                    await asyncio.sleep(self._min_delta - since_last_update)
                try:
                    latest = await lc.take()
                    log.debug(
                        f"got a new result in acquisition {acq_id} for {client.remote_address}"
                    )
                except Closed:
                    break  # acquisition is done, wait for the next one
                await self.handle_partial_result(
                    client=client,
                    previous_results=previous_results,
                    partial_results=latest,
                    acq_id=acq_id,
                )
                previous_results = latest


@numba.njit(cache=True)
def get_bbox(arr) -> typing.Tuple[int, ...]:
    xmin = arr.shape[1]
    ymin = arr.shape[0]
    xmax = 0
    ymax = 0

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if abs(value) < 1e-8:
                continue
            # got a non-zero value, update indices
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
    return int(ymin), int(ymax), int(xmin), int(xmax)


class SingleMaskUDF(ApplyMasksUDF):
    def get_result_buffers(self):
        dtype = np.result_type(self.meta.input_dtype, self.get_mask_dtype())
        return {
            'intensity': self.buffer(
                kind='nav', extra_shape=(1,), dtype=dtype, where='device', use='internal',
            ),
            'intensity_nav': self.buffer(
                kind='nav', extra_shape=(), dtype=dtype, where='device', use='result',
            ),
        }

    def get_results(self):
        # bummer: we can't reshape the data, as the extra_shape from the buffer
        # will override our desired shape. so we have to use two result buffers
        # instead:
        return {
            'intensity_nav': self.results.intensity.reshape(self.meta.dataset_shape.nav),
        }


class ParameterContainer:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def set_parameters(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters


class UDFContainer:
    def __init__(self, udfs: OrderedDict[str, UDF]):
        self._udfs = udfs

    def get_udfs(self) -> OrderedDict[str, UDF]:
        return self._udfs

    def set_udfs(self, udfs: OrderedDict[str, UDF]):
        self._udfs = udfs


class WSServer:
    def __init__(self, detector_settings_file):
        self.ws_connected = set()
        self.parameters = ParameterContainer({
            'cx': 516/2.0,
            'cy': 512/2.0,
            'ri': 200.0,
            'ro': 530.0,
        })
        self.detector_settings_file = detector_settings_file
        self.udfs = UDFContainer(self.get_udfs())
        self.sampler = ResultSampler(parameters=self.parameters, udfs=self.udfs)
        
        self.connect()

    def get_udfs(self) -> OrderedDict[str, "UDF"]:
        parameters = self.parameters.get_parameters()
        cx = parameters['cx']
        cy = parameters['cy']
        ri = parameters['ri']
        ro = parameters['ro']

        def _ring():
            return masks.ring(
                centerX=cx,
                centerY=cy,
                imageSizeX=516,
                imageSizeY=516,
                radius=ro,
                radius_inner=ri)

        mask_udf = SingleMaskUDF(mask_factories=[_ring])
        return OrderedDict({
            # "brightfield": SumSigUDF(),
            "annular": mask_udf,
            # "annular1": mask_udf,
            # "annular2": mask_udf,
            # "annular3": mask_udf,
            # "annular4": mask_udf,
            # "sum": SumUDF(),
            # "monitor": SignalMonitorUDF(),
            "monitor_partition": PartitionMonitorUDF(),
            # "icom": ICoMUDF.with_params(cx=cx, cy=cy, r=ro, flip_y=True),
            "com": CoMUDF.with_params(cx=cx, cy=cy, r=ro, flip_y=True, regression=1),
        })

    async def __call__(self, websocket: WebSocketServerProtocol):
        await self.client_loop(websocket)

    async def register_client(self, websocket: WebSocketClientProtocol):
        self.ws_connected.add(websocket)
        await self.sampler.add_client(websocket)

    async def unregister_client(self, websocket: WebSocketClientProtocol):
        self.ws_connected.remove(websocket)
        await self.sampler.remove_client(websocket)

    async def client_loop(self, websocket: WebSocketClientProtocol):
        try:
            try:
                await self.register_client(websocket)
                async for msg in websocket:
                    await self.handle_message(msg, websocket)
            except websockets.exceptions.ConnectionClosedError:
                await websocket.close()
        finally:
            await self.unregister_client(websocket)

    async def handle_message(self, msg, websocket):
        try:
            msg = json.loads(msg)
            # FIXME: hack to not require the 'event' "tag":
            if 'event' not in msg or msg['event'] == 'UPDATE_PARAMS':
                print(f"parameter update: {msg}")
                self.parameters.set_parameters(msg['parameters'])
                self.udfs.set_udfs(self.get_udfs())
                # broadcast to all clients:
                msg['event'] = 'UPDATE_PARAMS'
                await self.broadcast(json.dumps(msg))

            elif msg["event"] == "PREPARE_CORRECTED_PICK":
                asyncio.ensure_future(self.prepare_descan_corrected(msg["params"]))

            elif msg["event"] == "OFFLINE_PROCESSING":
                if msg["udf"] == "CORRECTED_PICK":
                    params = msg.get("params", {})
                    asyncio.ensure_future(self.pick_descan_corrected(params))

        except Exception as e:
            print(e)
            raise

    
    async def offline_processing(self, udf, **kwargs):
        return await self.offlinectx.run_udf(udf=udf, sync=False, **kwargs)
    
    
    async def prepare_descan_corrected(self, params):

        
        print("CPICK TO PREPARED")

        self.corpick_dataset = self.offlinectx.load(filetype = "mib", path=params["dataset"])

        comudf = CoMUDF.with_params(cy=128, cx=128, r=1000, regression=1)
        comresult = await self.offline_processing(comudf, dataset=self.corpick_dataset)
        self.regression_coefficients = comresult["regression"].data
        self.picknavimg = comresult["field_x"].data
        

        codec = BsLz4()
        compressed, encoding_meta = await sync_to_async(lambda: codec.encode(self.picknavimg))

        await self.broadcast(
            json.dumps({
                "event" : "CORRECTED_PICK_PREPARED",
                "encoding_meta" : encoding_meta
            })
        )

        await self.broadcast(
            memoryview(compressed)
        )
        
        print("CPICK PREPARED")
    
    async def pick_descan_corrected(self, params):

        corrpickudf = PickCorrectedUDF(regression_coefficients=self.regression_coefficients)
        result = await self.offline_processing(corrpickudf, dataset=self.corpick_dataset, **params)

        resultimg = result["intensity"].data

        codec = BsLz4()
        compressed, encoding_meta = await sync_to_async(lambda: codec.encode(resultimg))

        await self.broadcast(
            json.dumps({
                "event": "CORRECTED_PICK",
                "encoding_meta": encoding_meta
            })
        )

        await self.broadcast(
            memoryview(compressed)
        )


    async def broadcast(self, msg):
        websockets.broadcast(self.ws_connected, msg)

    async def handle_pending_acquisition(self, pending) -> str:
        acq_id = str(uuid.uuid4())
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_STARTED",
            "id": acq_id,
        }))
        return acq_id

    async def handle_acquisition_end(self, pending, acq_id: str):
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_ENDED",
            "id": acq_id,
        }))

    async def acquisition_loop(self):
        while True:
            pending_aq = await sync_to_async(self.conn.wait_for_acquisition, timeout=10)
            if pending_aq is None:
                continue
            acq_id = await self.handle_pending_acquisition(pending_aq)
            try:
                print(f"acquisition starting with id={acq_id}")
                t0 = time.perf_counter()
                num_updates = 0
                partial_results = None

                side = int(math.sqrt(pending_aq.nimages))

                aq = self.ctx.make_acquisition(
                    conn=self.conn,
                    pending_aq=pending_aq,
                    # frames_per_partition=10 * int(2/3 * side),
                    frames_per_partition=1 * side,
                    nav_shape=(side, side),
                )
                try:
                    udfs_only = list(self.udfs.get_udfs().values())
                    params = [udf._kwargs for udf in udfs_only]
                    part_res_iter = self.ctx.run_udf_iter(dataset=aq, udf=udfs_only, sync=False)
                    async with self.sampler.handle_acquisition(acq_id=acq_id) as result_sink:
                        async for partial_results in part_res_iter:
                            # parameter update:
                            udfs_only = list(self.udfs.get_udfs().values())
                            new_params = [udf._kwargs for udf in udfs_only]
                            if new_params != params:  # change detection
                                await part_res_iter.update_parameters_experimental(new_params)
                                params = new_params
                            await result_sink(partial_results)
                            num_updates += 1
                        await result_sink(partial_results)
                        num_updates += 1
                except Exception:
                    import traceback
                    traceback.print_exc()
                    self.ctx.close()
                    self.conn.close()
                    self.connect()
            finally:
                await self.handle_acquisition_end(pending_aq, acq_id)
            t1 = time.perf_counter()
            print(f"acquisition done with id={acq_id}; "
                  f"took {t1-t0:.3f}s; num_updates={num_updates}")
            num_updates = 0

    async def serve(self):
        # disable compression since we do our own lz4 for larger messages:
        async with websockets.serve(self, "localhost", 8444, compression=None):
            while True:
                try:
                    try:
                        await self.acquisition_loop()
                    except Exception:
                        log.exception("Exception in acquisition loop")
                        self.ctx.close()
                        self.conn.close()
                        self.connect()
                        continue
                finally:
                    self.conn.close()
                    self.ctx.close()

    def connect(self):
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(
                cpus=range(3), cudas=[]
            ),
            pin_workers=False,
            startup_timeout=120
        )
        ctx = LiveContext(executor=executor)

        with open(self.detector_settings_file, "rb") as f:
            detector_settings = tomllib.load(f)

        conn = ctx.make_connection(detector_settings["detector_type"]).open(
            **detector_settings["connection_arguments"]
        )

        offlinectx = Context()
        
        self.conn = conn
        self.ctx = ctx
        self.offlinectx = offlinectx
        log.info("live server connected and ready")


async def main(detector_settings_file):
    log.info("live server starting up...")
    server = WSServer(detector_settings_file=detector_settings_file)
    await server.serve()


@click.command()
@click.argument("detector_settings_file", type=str)
def cli(detector_settings_file):
    maybe_setup_tracing(service_name="libertem-live-server")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(detector_settings_file))


if __name__ == "__main__":
    cli()
