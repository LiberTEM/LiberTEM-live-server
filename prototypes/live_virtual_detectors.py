"""
Live plots with widgets.

Must configure `live_server.py` to use only `monitor_partition` and `annular`
UDFs (this is a limitation that can be lifted in production, this is only to
demonstrate that widget interactivity is feasible).
"""


import sys
import logging
import time
import json
import asyncio
import threading
import queue
from typing import List, Dict, Any
from typing_extensions import TypedDict

import click
import numpy as np
import numba
import bitshuffle
import websockets
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.scripting.scripting_interface import ScriptDataModel
from result_codecs import LossyU16, BsLz4


log = logging.getLogger(__name__)


class ResultItem(TypedDict):
    """
    Example instance of this class:
    {
        'bbox': [0, 515, 0, 515],
        'full_shape': [516, 516],
        'delta_shape': [516, 516],
        'dtype': 'float32',
        'encoding': 'bslz4',
        'encoding_meta': {},
        'channel_name': 'intensity',
        'udf_name': 'monitor_partition'
    }
    """
    # List[int] because JSON doesn't tuple
    bbox: List[int]   # ymin, ymax, xmin, xmax: indices
    full_shape: List[int]
    delta_shape: List[int]
    dtype: str
    encoding: str
    encoding_meta: Dict[str, Any]
    channel_name: str
    udf_name: str


class Plotter:
    """
    Plotting functionality that interacts with Panta Rhei. This has to be run
    on the main Python threads.

    Pull data from the given `State` object and update the repo data in case of
    changes (here: as fast as possible)

    Poll the GUI for parameter updates and push them to the `params_queue`,
    which are then sent back to the server in the `update_params_task`.
    """

    def __init__(
        self,
        state: "State",
        todo_event: threading.Event,
        params_queue: queue.Queue,
    ):
        self.state = state
        self.todo_event = todo_event
        self._widget = None
        self._vd_params = None
        self.data_models: Dict[str, ScriptDataModel] = {}
        self.si = PRScriptingInterface()
        self._params_queue = params_queue

    def loop(self):
        # update as fast as possible, always using the most up-to-date state:
        while True:
            # self.todo_event.wait()
            self.update_params()

            keys = self.state.keys()

            if not self.todo_event.is_set():
                time.sleep(0.01)
                continue
            else:
                # XXX what if the other thread called `set` again just before this?
                # we might skip an update if we are unlucky?
                self.todo_event.clear()

            t0 = time.time()
            for key in keys:
                with self.state.data_lock:
                    arr = self.state.composed_data[key]
                    # arr = self.state.data[key]
                    mask = self.state.valid_masks[key]
                    self.si.data_to_repo(key, arr)

                if key not in self.data_models:
                    model = self.data_models[key] = self.si.display_image(key)
                    # XXX: hacks - always put the virtual detector widget into
                    # the monitor partition UDF result display:
                    if key == "monitor_partition-intensity":
                        virtual_detector = model.insert(
                            PRScriptingTypes.VirtualDetector
                        )
                        self._widget = virtual_detector
                    print(f"data model: {self.data_models[key]}")

                self.update_display_control(key, mask)
            t1 = time.time()
            if len(keys) > 0:
                # print(f"plot updates: {t1-t0:.2f}s")
                pass

    def update_display_control(self, key: str, mask: np.ndarray):
        if key not in self.data_models:
            return
        if True:
            with self.state.data_lock:
                mask &= self.state.data[key] != 0
                # XXX hacks...
                # very simple "outlier removal" specifically for getting rid of
                # the INT_MAX stuff that dectris does for bad pixels:
                try:
                    mask = mask & (self.state.data[key] != np.max(self.state.data[key][mask]))
                except ValueError:
                    pass  # meh...

                valid_data = self.state.data[key][mask].copy()
            if len(valid_data) > 0:
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
                model = self.data_models[key]
                dc = model.get_display_control()
                params = dc.get_parameters()
                params['levels'] = [vmin, vmax]
                params['auto_contrast'] = False
                # params['color_map'] = 'temperature'
                dc.set_parameters(params)

    def update_params(self):
        if self._widget is None:
            return None
        params = self._widget.get_parameters(scale_mode='pixel')
        new_params = {
            'ri': params['inner'],
            'ro': params['outer'],
            'cx': params['center'][0],
            'cy': params['center'][1],
        }
        if self._vd_params != new_params:
            self._vd_params = new_params
            self._params_queue.put(new_params)

    def get_vd_params(self):
        return self._vd_params


class State:
    def __init__(self, todo_event: asyncio.Event):
        self.data: Dict[str, np.ndarray] = {}
        self.composed_data: Dict[str, np.ndarray] = {}
        self.valid_masks: Dict[str, np.ndarray] = {}
        self.data_lock = threading.Lock()
        self._gen_counter = 0
        self._todo_event = todo_event

    def get_or_create(
        self, key: str, shape: List[int], dtype: str
    ) -> np.ndarray:
        if key in self.data:
            return self.data[key]
        else:
            new_arr = np.zeros(tuple(shape), dtype=dtype)
            self.data[key] = new_arr
            return new_arr

    def get_or_create_valid_mask(
        self, key: str, shape: List[int]
    ) -> np.ndarray:
        if key in self.valid_masks:
            return self.valid_masks[key]
        else:
            new_arr = np.zeros(tuple(shape), dtype=bool)
            self.valid_masks[key] = new_arr
            return new_arr

    @property
    def counter(self):
        return self._gen_counter

    def keys(self):
        return list(self.valid_masks.keys())

    def apply_result_item(
        self,
        acq_id: str,
        item: ResultItem,
        compressed_data: bytes,
    ):
        if item['encoding'] == "lossy-u16-bslz4":
            codec = LossyU16()
        elif item['encoding'] == "bslz4":
            codec = BsLz4()
        delta_data = codec.decode(compressed_data, item["encoding_meta"])
        # print(f"decompressed into {decomp.nbytes} bytes")

        if delta_data.nbytes == 0:
            return

        with self.data_lock:
            self._gen_counter += 1
            key = f"{item['udf_name']}-{item['channel_name']}"
            arr = self.get_or_create(key, item['full_shape'], item['dtype'])
            delta_arr = delta_data.reshape(item['delta_shape'])

            bb = item['bbox']
            arr[
                bb[0]:bb[1] + 1,
                bb[2]:bb[3] + 1,
            ] += delta_arr

            mask = self.get_or_create_valid_mask(key, item['full_shape'])
            mask[
                bb[0]:bb[1] + 1,
                bb[2]:bb[3] + 1,
            ] = True

            if key not in self.composed_data:
                composed = np.zeros_like(arr)
                self.composed_data[key] = composed
            self.composed_data[key][mask] = arr[mask]
        self._todo_event.set()

    def acquisition_started(self, acq_id: str):
        # just clear everything for now:
        for item in self.data.values():
            item[:] = 0
        for item in self.valid_masks.values():
            item[:] = False

    def acquisition_ended(self, acq_id: str):
        pass


async def update_params_task(params_queue: queue.Queue, websocket):
    loop = asyncio.get_running_loop()
    while True:
        params = await loop.run_in_executor(None, lambda: params_queue.get())
        print("updated parameters", params)
        await websocket.send(json.dumps({
            "event": "UPDATE_PARAMS",
            "parameters": params,
        }))


class RecvThread(threading.Thread):
    def __init__(
        self, state: State, todo_event: threading.Event, plotter: Plotter, url: str,
        params_queue: queue.Queue,
    ):
        self.state = state
        self.todo = todo_event
        self.plotter = plotter
        self.url = url
        self.params_queue = params_queue
        super().__init__()

    async def main(self):
        async with websockets.connect(
            self.url, max_size=16*1024*1024,
        ) as websocket:
            last_msg = None

            update_task = asyncio.ensure_future(update_params_task(self.params_queue, websocket))

            try:
                while True:
                    msg = await websocket.recv()
                    decoded_msg = json.loads(msg)
                    last_msg = decoded_msg

                    # print(decoded_msg)

                    event = decoded_msg['event']
                    if event == "ACQUISITION_STARTED":
                        print(f"acquisition started: {decoded_msg['id']}")
                        self.state.acquisition_started(
                            acq_id=decoded_msg['id']
                        )
                    elif event == "ACQUISITION_ENDED":
                        self.state.acquisition_ended(
                            acq_id=decoded_msg['id']
                        )
                    elif event == "RESULT":
                        delta = 0.0
                        delta_apply = 0.0
                        for chan in decoded_msg['channels']:
                            msg = await websocket.recv()
                            # print(f"binary message of length {len(msg)}")
                            # print(chan)
                            t0 = time.time()
                            self.state.apply_result_item(
                                acq_id=decoded_msg['id'],
                                item=chan,
                                compressed_data=msg,
                            )
                            t1 = time.time()
                            delta_apply += t1 - t0
                        # print(f"decompression took {delta:.3f}s")
                        # print(f"apply took {delta_apply:.3f}s")
                    else:
                        print(f"last msg: {last_msg}")
            finally:
                update_task.cancel()

    async def restart_loop(self):
        while True:
            try:
                await self.main()
            except Exception as e:
                log.exception("got an exception in the main loop, reconnecting")
                continue

    def run(self):
        asyncio.run(self.restart_loop())


@click.command()
@click.option('--url', type=str, default='ws://localhost:8444')
def main(url):
    todo = threading.Event()
    state = State(todo_event=todo)

    params_queue = queue.Queue(maxsize=0)
    plotter = Plotter(state=state, todo_event=todo, params_queue=params_queue)
    recv = RecvThread(
        state=state, todo_event=todo, plotter=plotter, url=url, params_queue=params_queue,
    )
    recv.daemon = True
    recv.start()

    plotter.loop()


if __name__ == "__main__":
    main()
