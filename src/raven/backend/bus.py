"""
Helpers for publishing to RAVEN's Detection Event Bus (a bubus EventBus) from threads other than its event loop.
"""

import asyncio
import threading

from bubus import EventBus


class ThreadSafeBus:
    """
    Wraps the event bus so code running in worker threads can publish to it.
    bubus only accepts dispatch() from the thread running its event loop, so dispatch() here hands each
    event to that thread instead of calling the bus directly. It returns right away and never blocks the worker.
    """

    def __init__(self, event_bus, loop):
        # the real bus, and the event loop (bus thread) it runs on
        self.event_bus = event_bus
        self.loop = loop

    def dispatch(self, event):
        # schedule the event dispatch to happen in the event loop thread
        self.loop.call_soon_threadsafe(self.event_bus.dispatch, event)


class BackgroundBus(ThreadSafeBus):
    """
    An event bus running on its own event loop thread, for programs whose main thread is busy elsewhere
    (the operator console's main thread runs the Qt window). dispatch() works from any thread, and
    on() / submit() / call() hand their work to the bus thread.
    """

    def __init__(self, name="raven_event_bus"):
        loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=loop.run_forever, name=name, daemon=True)
        self._thread.start()
        super().__init__(None, loop)

        # the bus is created on its own thread, the same way raven.pipeline creates it inside its event loop
        async def create():
            return EventBus(name=name, parallel_handlers=True)

        self.event_bus = self.submit(create()).result()

    def dispatch(self, event, timeout=30.0):
        """
        Publish an event from any thread without ever dropping it. bubus refuses new events while 100 are pending,
        which feeds read faster than real time can reach, so a worker waits here until the bus has room
        (slowing that feed down) instead of losing the event. Raises if the bus stays full for timeout seconds.
        """
        if threading.current_thread() is self._thread:
            return self.event_bus.dispatch(event)

        async def publish():
            while True:
                try:
                    self.event_bus.dispatch(event)
                    return
                except RuntimeError as error:
                    # bubus raises a plain RuntimeError when full; anything else is a real error
                    if "at capacity" not in str(error):
                        raise
                    await asyncio.sleep(0.05)

        self.submit(publish()).result(timeout)

    def submit(self, coroutine):
        """
        Run a coroutine on the bus thread. Returns a concurrent.futures.Future for its result.
        """
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def call(self, fn, *args):
        """
        Run a plain function on the bus thread and wait for its return value.
        """
        async def run():
            return fn(*args)

        return self.submit(run()).result()

    def on(self, event_class, handler):
        """
        Register a handler for an event class. Handlers run on the bus thread.
        """
        self.call(self.event_bus.on, event_class, handler)

    def close(self, timeout=5.0):
        """
        Hand any remaining events to the subscribers (waiting at most timeout seconds), then stop the bus and its thread.
        """
        async def shut_down():
            try:
                await asyncio.wait_for(self.event_bus.wait_until_idle(), timeout)
            except (TimeoutError, asyncio.TimeoutError):
                pass
            # clear frees the bus's name, so a later bus in the same process can reuse it
            await self.event_bus.stop(clear=True)

        try:
            self.submit(shut_down()).result(timeout + 2)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(2)
