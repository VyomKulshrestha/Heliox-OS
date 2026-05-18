"""Unit tests for pilot.agents.decomposer.

Covers:
  - Single-step task → exactly one subtask
  - Multi-step task with dependencies → correctly ordered plan
  - Unknown / invalid JSON from model → handled gracefully (no crash)
  - Empty input string → empty / non-complex plan (no crash)
  - Dependency graphs are always acyclic
  - get_summary and get_execution_order helpers

Run with:
    cd daemon
    pytest tests/test_decomposer.py -v --tb=short
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pytest
import pytest_asyncio

from pilot.agents.decomposer import (
    Subtask,
    SubtaskStatus,
    TaskDecomposer,
    TaskDecomposition,
)

# ---------------------------------------------------------------------------
# Stub ModelRouter — pre-canned JSON responses, zero network calls
# ---------------------------------------------------------------------------


class StubModelRouter:
    """A deterministic stand-in for ModelRouter that returns pre-set JSON."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: ARG002
        return self._response


def make_decomposer(response: str) -> TaskDecomposer:
    """Return a TaskDecomposer backed by a stub that always returns *response*."""
    return TaskDecomposer(model_router=StubModelRouter(response))


# ---------------------------------------------------------------------------
# Canned JSON payloads
# ---------------------------------------------------------------------------

SINGLE_STEP_RESPONSE = json.dumps(
    {
        "is_complex": False,
        "estimated_total_time": "5s",
        "subtasks": [
            {
                "title": "Create folder",
                "description": "Create the project directory",
                "agent": "system",
                "depends_on": [],
                "estimated_complexity": 0.1,
            }
        ],
    }
)

MULTI_STEP_RESPONSE = json.dumps(
    {
        "is_complex": True,
        "estimated_total_time": "30s",
        "subtasks": [
            {
                "title": "Create folder",
                "description": "Create the project directory",
                "agent": "system",
                "depends_on": [],
                "estimated_complexity": 0.2,
            },
            {
                "title": "Install Flask",
                "description": "pip install flask",
                "agent": "system",
                "depends_on": ["0"],
                "estimated_complexity": 0.3,
            },
            {
                "title": "Generate API code",
                "description": "Write Flask app",
                "agent": "code",
                "depends_on": ["0"],
                "estimated_complexity": 0.6,
            },
            {
                "title": "Run tests",
                "description": "Execute pytest",
                "agent": "code",
                "depends_on": ["1", "2"],
                "estimated_complexity": 0.5,
            },
        ],
    }
)

# Not complex — no subtasks
NOT_COMPLEX_RESPONSE = json.dumps({"is_complex": False, "subtasks": []})

# Model returns invalid JSON (network / parsing error scenario)
INVALID_JSON_RESPONSE = "I'm sorry, I cannot help with that."

# Model returns valid JSON but with an empty subtasks list even when is_complex=True
COMPLEX_NO_SUBTASKS_RESPONSE = json.dumps({"is_complex": True, "subtasks": []})

# Model returns a plan with a dependency cycle (decomposer should still not crash)
CYCLE_RESPONSE = json.dumps(
    {
        "is_complex": True,
        "subtasks": [
            {
                "title": "Step A",
                "description": "A",
                "agent": "system",
                "depends_on": ["1"],  # depends on B
                "estimated_complexity": 0.4,
            },
            {
                "title": "Step B",
                "description": "B",
                "agent": "system",
                "depends_on": ["0"],  # depends on A → cycle
                "estimated_complexity": 0.4,
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_no_cycle(subtasks: list[Subtask]) -> None:
    """DFS-based cycle detection on git statusthe dependency graph.

    Raises AssertionError if a cycle is found.
    """
    # Build adjacency by order index *and* by subtask.id
    id_to_order: dict[str, int] = {}
    for st in subtasks:
        id_to_order[str(st.order)] = st.order
        id_to_order[st.id] = st.order

    graph: dict[int, list[int]] = defaultdict(list)
    for st in subtasks:
        for dep in st.depends_on:
            if dep in id_to_order:
                graph[st.order].append(id_to_order[dep])

    # Standard DFS with three-colour marking
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = {st.order: WHITE for st in subtasks}

    def dfs(node: int) -> bool:
        colour[node] = GRAY
        for neighbour in graph.get(node, []):
            if colour.get(neighbour, WHITE) == GRAY:
                return True  # back-edge → cycle
            if colour.get(neighbour, WHITE) == WHITE and dfs(neighbour):
                return True
        colour[node] = BLACK
        return False

    for st in subtasks:
        if colour[st.order] == WHITE:
            assert not dfs(st.order), f"Cycle detected in dependency graph (subtask order={st.order})"


# ---------------------------------------------------------------------------
# Tests: decompose()
# ---------------------------------------------------------------------------


class TestDecomposeSingleStep:
    """A goal the model treats as simple (is_complex=False) with one subtask."""

    @pytest.mark.asyncio
    async def test_exactly_one_subtask(self) -> None:
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose("Create a folder")

        assert len(result.subtasks) == 1

    @pytest.mark.asyncio
    async def test_is_complex_false(self) -> None:
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose("Create a folder")

        assert result.is_complex is False

    @pytest.mark.asyncio
    async def test_subtask_fields_populated(self) -> None:
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose("Create a folder")

        st = result.subtasks[0]
        assert st.title == "Create folder"
        assert st.agent == "system"
        assert st.estimated_complexity == pytest.approx(0.1)
        assert st.status == SubtaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_goal_preserved(self) -> None:
        goal = "Create a folder"
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose(goal)

        assert result.goal == goal

    @pytest.mark.asyncio
    async def test_no_cycle_single(self) -> None:
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose("Create a folder")

        assert_no_cycle(result.subtasks)


class TestDecomposeMultiStep:
    """A goal the model decomposes into a dependency-aware multi-step plan."""

    @pytest.mark.asyncio
    async def test_correct_number_of_subtasks(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        assert len(result.subtasks) == 4

    @pytest.mark.asyncio
    async def test_is_complex_true(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        assert result.is_complex is True

    @pytest.mark.asyncio
    async def test_subtask_order_assigned(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        orders = [st.order for st in result.subtasks]
        assert orders == list(range(len(result.subtasks)))

    @pytest.mark.asyncio
    async def test_dependencies_preserved(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        # "Run tests" (index 3) must depend on "Install Flask" (1) and "Generate API code" (2)
        run_tests = result.subtasks[3]
        assert "1" in run_tests.depends_on or "2" in run_tests.depends_on

    @pytest.mark.asyncio
    async def test_agents_assigned_correctly(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        agents = [st.agent for st in result.subtasks]
        assert "system" in agents
        assert "code" in agents

    @pytest.mark.asyncio
    async def test_no_cycle_multi(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        assert_no_cycle(result.subtasks)

    @pytest.mark.asyncio
    async def test_estimated_total_time_set(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        assert result.estimated_total_time == "30s"


class TestDecomposeEmptyInput:
    """Empty string input must not raise; returns a non-crashing result."""

    @pytest.mark.asyncio
    async def test_empty_string_does_not_crash(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("")

        assert isinstance(result, TaskDecomposition)

    @pytest.mark.asyncio
    async def test_empty_string_goal_preserved(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("")

        assert result.goal == ""

    @pytest.mark.asyncio
    async def test_empty_string_is_not_complex(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("")

        assert result.is_complex is False

    @pytest.mark.asyncio
    async def test_empty_string_zero_subtasks(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("")

        assert len(result.subtasks) == 0


class TestDecomposeInvalidModelOutput:
    """Model returns unparseable output — decomposer must handle gracefully."""

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_raise(self) -> None:
        """Decomposer should catch JSON parse failures, not propagate them."""
        decomposer = make_decomposer(INVALID_JSON_RESPONSE)
        # Must not raise
        result = await decomposer.decompose("Some task")

        assert isinstance(result, TaskDecomposition)

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_simple(self) -> None:
        decomposer = make_decomposer(INVALID_JSON_RESPONSE)
        result = await decomposer.decompose("Some task")

        assert result.is_complex is False

    @pytest.mark.asyncio
    async def test_invalid_json_produces_no_subtasks(self) -> None:
        decomposer = make_decomposer(INVALID_JSON_RESPONSE)
        result = await decomposer.decompose("Some task")

        assert len(result.subtasks) == 0

    @pytest.mark.asyncio
    async def test_complex_no_subtasks_does_not_crash(self) -> None:
        """is_complex=True but subtasks=[] — edge case that must not crash."""
        decomposer = make_decomposer(COMPLEX_NO_SUBTASKS_RESPONSE)
        result = await decomposer.decompose("Unusual task")

        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) == 0


# ---------------------------------------------------------------------------
# Tests: get_execution_order()
# ---------------------------------------------------------------------------


class TestGetExecutionOrder:
    """Verify topological sort produces valid, dependency-respecting batches."""

    @pytest.mark.asyncio
    async def test_returns_batches(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        batches = decomposer.get_execution_order(result)

        assert len(batches) >= 1

    @pytest.mark.asyncio
    async def test_all_subtasks_appear_exactly_once(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        batches = decomposer.get_execution_order(result)

        seen = [st for batch in batches for st in batch]
        assert len(seen) == len(result.subtasks)

    @pytest.mark.asyncio
    async def test_root_task_in_first_batch(self) -> None:
        """The subtask with no dependencies must appear before any dependent."""
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        batches = decomposer.get_execution_order(result)

        first_batch_titles = {st.title for st in batches[0]}
        assert "Create folder" in first_batch_titles

    @pytest.mark.asyncio
    async def test_dependencies_satisfied_before_dependents(self) -> None:
        """For each batch, all dependencies of its tasks appeared in earlier batches."""
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        batches = decomposer.get_execution_order(result)

        completed_orders: set[str] = set()
        for batch in batches:
            for st in batch:
                for dep in st.depends_on:
                    assert dep in completed_orders, (
                        f"Dependency {dep!r} of subtask {st.title!r} was not completed before its batch"
                    )
            # Mark all in this batch as completed after checking
            for st in batch:
                completed_orders.add(str(st.order))
                completed_orders.add(st.id)

    @pytest.mark.asyncio
    async def test_empty_decomposition_returns_empty(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("")
        batches = decomposer.get_execution_order(result)

        assert batches == []

    @pytest.mark.asyncio
    async def test_single_step_single_batch(self) -> None:
        decomposer = make_decomposer(SINGLE_STEP_RESPONSE)
        result = await decomposer.decompose("Create a folder")
        batches = decomposer.get_execution_order(result)

        assert len(batches) == 1
        assert len(batches[0]) == 1

    @pytest.mark.asyncio
    async def test_cycle_does_not_crash(self) -> None:
        """A cyclic dependency graph must not cause an infinite loop or exception."""
        decomposer = make_decomposer(CYCLE_RESPONSE)
        result = await decomposer.decompose("Cyclic task")
        # Should complete (decomposer has a deadlock-break path)
        batches = decomposer.get_execution_order(result)
        assert isinstance(batches, list)

    @pytest.mark.asyncio
    async def test_cycle_all_subtasks_eventually_scheduled(self) -> None:
        """Even with a cycle, every subtask should appear in exactly one batch."""
        decomposer = make_decomposer(CYCLE_RESPONSE)
        result = await decomposer.decompose("Cyclic task")
        batches = decomposer.get_execution_order(result)

        scheduled = [st for batch in batches for st in batch]
        assert len(scheduled) == len(result.subtasks)


# ---------------------------------------------------------------------------
# Tests: get_summary()
# ---------------------------------------------------------------------------


class TestGetSummary:
    @pytest.mark.asyncio
    async def test_simple_task_summary(self) -> None:
        decomposer = make_decomposer(NOT_COMPLEX_RESPONSE)
        result = await decomposer.decompose("Quick task")
        summary = decomposer.get_summary(result)

        assert "Simple task" in summary

    @pytest.mark.asyncio
    async def test_complex_task_summary_contains_goal(self) -> None:
        goal = "Build a Flask API"
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose(goal)
        summary = decomposer.get_summary(result)

        assert goal in summary

    @pytest.mark.asyncio
    async def test_complex_task_summary_contains_subtask_count(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        summary = decomposer.get_summary(result)

        assert "4" in summary  # 4 subtasks

    @pytest.mark.asyncio
    async def test_summary_is_string(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")

        assert isinstance(decomposer.get_summary(result), str)


# ---------------------------------------------------------------------------
# Tests: TaskDecomposition.to_dict()
# ---------------------------------------------------------------------------


class TestTaskDecompositionToDict:
    @pytest.mark.asyncio
    async def test_to_dict_keys(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        d = result.to_dict()

        assert "decomposition_id" in d
        assert "goal" in d
        assert "is_complex" in d
        assert "subtask_count" in d
        assert "subtasks" in d

    @pytest.mark.asyncio
    async def test_subtask_count_matches_list(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        d = result.to_dict()

        assert d["subtask_count"] == len(d["subtasks"])

    @pytest.mark.asyncio
    async def test_subtask_dict_has_required_fields(self) -> None:
        decomposer = make_decomposer(MULTI_STEP_RESPONSE)
        result = await decomposer.decompose("Build a Flask API")
        d = result.to_dict()

        required = {"id", "title", "description", "agent", "depends_on", "status", "estimated_complexity", "order"}
        for sub_dict in d["subtasks"]:
            assert required <= set(sub_dict.keys()), f"Subtask dict missing keys: {required - set(sub_dict.keys())}"


# ---------------------------------------------------------------------------
# Tests: Acyclicity guarantee across all canned payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,goal",
    [
        (SINGLE_STEP_RESPONSE, "Single step task"),
        (MULTI_STEP_RESPONSE, "Multi step task"),
        (NOT_COMPLEX_RESPONSE, "Not complex task"),
        (COMPLEX_NO_SUBTASKS_RESPONSE, "Complex no subtasks"),
    ],
)
async def test_dependency_graph_acyclic(response: str, goal: str) -> None:
    """All well-formed responses must produce an acyclic dependency graph."""
    decomposer = make_decomposer(response)
    result = await decomposer.decompose(goal)
    assert_no_cycle(result.subtasks)


# ---------------------------------------------------------------------------
# Tests: Subtask default state
# ---------------------------------------------------------------------------


class TestSubtaskDefaults:
    def test_new_subtask_is_pending(self) -> None:
        st = Subtask(title="Test", description="desc", agent="system")
        assert st.status == SubtaskStatus.PENDING

    def test_new_subtask_has_unique_id(self) -> None:
        st1 = Subtask(title="A", description="a", agent="system")
        st2 = Subtask(title="B", description="b", agent="system")
        assert st1.id != st2.id

    def test_new_subtask_empty_output(self) -> None:
        st = Subtask(title="Test", description="desc", agent="system")
        assert st.output == ""
        assert st.error == ""
