"""
Live plots with widgets.

Must configure `live_server.py` to use only `monitor_partition` and `annular`
UDFs (this is a limitation that can be lifted in production, this is only to
demonstrate that widget interactivity is feasible).
"""

import asyncio
import json
import logging
import threading
import time
import warnings

import click
import numpy as np
import rerun as rr
import websockets
from libertem.viz.base import visualize_simple
from typing_extensions import TypedDict

log = logging.getLogger(__name__)


class ResultItem(TypedDict):
    """
    Example instance of this class:
    {
        'shape': [516, 516],
        'damage_shape': [516, 516],
        'dtype': 'float32',
        'channel_name': 'intensity',
        'udf_name': 'monitor_partition'
    }
    """

    # list[int] because JSON doesn't tuple
    shape: list[int]
    damage_shape: list[int]
    dtype: str
    channel_name: str
    udf_name: str


class Plotter:
    """
    Plotting functionality that interacts with Panta Rhei. This has to be run
    on the main Python threads.

    Pull data from the given `State` object and update the repo data in case of
    changes (here: as fast as possible)
    """

    def __init__(
        self,
        state: "State",
        todo_event: threading.Event,
    ):
        self.state = state
        self.todo_event = todo_event

    def loop(self):
        # Make sure we update immediately with the first result
        t0 = -np.inf
        # update as fast as possible, always using the most up-to-date state:
        while True:
            keys = self.state.keys()

            if not self.todo_event.is_set():
                time.sleep(0.01)
                continue
            else:
                # XXX what if the other thread called `set` again just before this?
                # we might skip an update if we are unlucky?
                self.todo_event.clear()
            now = time.time()
            if now - t0 > 0.1:
                t0 = now
                for key in keys:
                    with self.state.data_lock:
                        arr = self.state.data[key]
                        damage = self.state.valid_masks[key]
                        # FIXME no idea why the damage shape is wrong some times
                        if damage is not None and damage.shape != arr.shape:
                            damage = None
                        if len(arr.shape) == 2:
                            viz = visualize_simple(arr, damage=damage)

                            rr.log(key, rr.Image(viz, color_model="RGBA"))

                # t1 = time.time()
                if len(keys) > 0:
                    # print(f"plot updates: {t1-t0:.2f}s")
                    pass


class State:
    def __init__(self, todo_event: threading.Event):
        self.data: dict[str, np.ndarray] = {}
        self.valid_masks: dict[str, np.ndarray] = {}
        self.data_lock = threading.Lock()
        self._gen_counter = 0
        self._todo_event = todo_event

    @property
    def counter(self):
        return self._gen_counter

    def keys(self):
        return list(self.data.keys())

    def apply_result_item(
        self,
        acq_id: str,
        item: ResultItem,
        compressed_data: bytes,
        damage: bytes,
    ):
        # For some reason one may get shape mismatch or invalid types?
        # Not observed with simplified transport protocol
        try:
            new_arr = np.frombuffer(compressed_data, dtype=item["dtype"])
            new_arr = new_arr.reshape(item["shape"])
            damage_arr = np.frombuffer(damage, dtype=bool).reshape(item["damage_shape"])
        except Exception as e:
            warnings.warn(str(e))
            raise
            # damage_arr = None
            # new_arr = None
        if new_arr is not None:
            with self.data_lock:
                self._gen_counter += 1
                key = f"{item['udf_name']}-{item['channel_name']}"
                self.data[key] = new_arr
                self.valid_masks[key] = damage_arr
            self._todo_event.set()

    def acquisition_started(self, acq_id: str):
        pass

    def acquisition_ended(self, acq_id: str):
        pass


class RecvThread(threading.Thread):
    def __init__(
        self,
        state: State,
        todo_event: threading.Event,
        plotter: Plotter,
        url: str,
    ):
        self.state = state
        self.todo = todo_event
        self.plotter = plotter
        self.url = url
        super().__init__()

    async def main(self):
        while True:
            try:
                async with websockets.connect(
                    self.url,
                    max_size=16 * 1024 * 1024,
                ) as websocket:
                    last_msg = None

                    try:
                        while True:
                            msg = await websocket.recv()
                            try:
                                decoded_msg = json.loads(msg)
                                last_msg = decoded_msg

                                # print(decoded_msg)

                                event = decoded_msg["event"]
                                if event == "ACQUISITION_STARTED":
                                    print(f"acquisition started: {decoded_msg['id']}")
                                    self.state.acquisition_started(
                                        acq_id=decoded_msg["id"]
                                    )
                                elif event == "ACQUISITION_ENDED":
                                    self.state.acquisition_ended(
                                        acq_id=decoded_msg["id"]
                                    )
                                elif event == "RESULT":
                                    delta_apply = 0.0
                                    for chan in decoded_msg["channels"]:
                                        msg = await websocket.recv()
                                        msg_damage = await websocket.recv()
                                        # print(f"binary message of length {len(msg)}")
                                        # print(chan)
                                        t0 = time.time()
                                        self.state.apply_result_item(
                                            acq_id=decoded_msg["id"],
                                            item=chan,
                                            compressed_data=msg,
                                            damage=msg_damage,
                                        )
                                        t1 = time.time()
                                        delta_apply += t1 - t0
                                    # print(f"decompression took {delta:.3f}s")
                                    # print(f"apply took {delta_apply:.3f}s")
                                else:
                                    print(f"last msg: {last_msg}")
                            except json.JSONDecodeError as e:
                                warnings.warn(msg.decode("utf8") + str(e))
                    finally:
                        pass
            except OSError as e:
                # For some reason e.errno is not set, message says sth about multiple exceptions (?)
                # 111 is connection refused
                if "111" in str(e):
                    # reconnect
                    print("Trying to reconnect...")
                    await asyncio.sleep(1)
                    break
                else:
                    raise

    async def restart_loop(self):
        while True:
            try:
                await self.main()
            except Exception:
                log.exception("got an exception in the main loop, reconnecting")
                raise
                continue

    def run(self):
        asyncio.run(self.restart_loop())


@click.command()
@click.option("--url", type=str, default="ws://localhost:8444")
@click.argument("name", default="live server demo")
def main(url, name):
    rr.init(name)
    rr.spawn()

    todo = threading.Event()
    state = State(todo_event=todo)

    plotter = Plotter(state=state, todo_event=todo)
    recv = RecvThread(state=state, todo_event=todo, plotter=plotter, url=url)
    recv.daemon = True
    recv.start()

    plotter.loop()


if __name__ == "__main__":
    main()
