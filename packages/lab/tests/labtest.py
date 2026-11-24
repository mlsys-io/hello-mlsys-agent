"""Shared helpers for tests that need a task runtime without a model."""
import asyncio

from mlab.web.agent import TaskRuntime
from mlab.web.bus import Bus
from mlab.web.fake_client import FakeAgentClient


def make_runtime(root, factory=FakeAgentClient):
    return TaskRuntime("9.2-1", root, Bus(), factory, opening=False)


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(.01)
