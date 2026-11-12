"""pilot.timeouts.bounded: wait_for as Python 3.12 has it, on every supported version, and no lost cancellation."""

from __future__ import annotations

import asyncio

import pytest

from pilot.timeouts import bounded


def run(coro):
    return asyncio.run(coro)


def test_a_result_in_time_is_returned():
    async def answer():
        await asyncio.sleep(0)
        return 42

    assert run(bounded(answer(), 1.0)) == 42


def test_running_out_raises_timeout_error_and_cancels_the_coroutine_where_it_stands():
    cleaned: list[str] = []

    async def slow():
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append("finally ran")

    async def main():
        with pytest.raises(TimeoutError):
            await bounded(slow(), 0.05)
        assert cleaned == ["finally ran"]  # before the caller heard of the timeout

    run(main())


def test_running_out_cancels_a_task_and_leaves_a_shielded_one_running():
    async def main():
        task = asyncio.create_task(asyncio.sleep(10))
        with pytest.raises(TimeoutError):
            await bounded(task, 0.05)
        await asyncio.sleep(0)
        assert task.cancelled()
        inner = asyncio.create_task(asyncio.sleep(0.2, result="done"))
        with pytest.raises(TimeoutError):
            await bounded(asyncio.shield(inner), 0.05)
        assert await inner == "done"

    run(main())


def test_no_bound_with_none():
    async def main():
        return await bounded(asyncio.sleep(0.01, result="ok"), None)

    assert run(main()) == "ok"


def test_zero_or_less_gives_a_finished_result_and_times_out_otherwise():
    async def main():
        done = asyncio.get_running_loop().create_future()
        done.set_result("ready")
        assert await bounded(done, 0) == "ready"
        pending = asyncio.get_running_loop().create_future()
        with pytest.raises(TimeoutError):
            await bounded(pending, -1)
        assert pending.cancelled()
        with pytest.raises(TimeoutError):
            await bounded(asyncio.sleep(1), 0)

    run(main())


def test_a_cancellation_that_arrives_as_the_awaited_thing_finishes_is_not_lost():
    """The race wait_for loses before Python 3.12 (54ab435): the result and the stop arrive together."""

    async def main():
        loop = asyncio.get_running_loop()
        for _ in range(50):
            future = loop.create_future()
            waiter = asyncio.create_task(bounded(future, 10))
            await asyncio.sleep(0)  # the waiter is now waiting on the future
            future.set_result("approved")
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

    run(main())


def test_the_timeout_error_is_asyncios_too():
    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await bounded(asyncio.sleep(10), 0.01)

    run(main())
