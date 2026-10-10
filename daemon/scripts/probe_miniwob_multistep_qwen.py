"""Research-only multi-step GUI rollout on pinned public MiniWoB tasks.

Both policies start from independently verified resets. Only terminal task
reward is scored; this probe never controls the user's desktop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from scripts.collect_miniwob_branches import SOURCE_COMMIT, SOURCE_REPO, PinnedAssets, _reset, visual_feature
from scripts.collect_miniwob_reward_actions import candidate_actions, goal_description
from scripts.probe_miniwob_reward_qwen import MODEL_REVISIONS, parse_index


def parse_action_index(answer: str, count: int) -> int | None:
    """Accept a bare index or the model's unambiguous `Index: N` echo."""
    return parse_index(re.sub(r"^\s*Index:\s*", "", answer, flags=re.IGNORECASE), count)


def ollama_json(path: str, payload: dict | None = None) -> dict:
    """Talk only to the local Ollama daemon; no cloud model or private desktop data."""
    request = Request(
        "http://127.0.0.1:11434/api/" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Invalid Ollama response")
    return result


def ollama_model_digest(name: str) -> str:
    models = ollama_json("tags").get("models", [])
    for model in models:
        if model.get("name") == name and re.fullmatch(r"[0-9a-f]{64}", model.get("digest", "")):
            return model["digest"]
    raise ValueError("Requested local Ollama model is not installed with a valid digest")


def choose_ollama(
    goal: str, controls: list[dict], history: list[dict], model: str, max_prompt_tokens: int, prompt_lengths: list[int]
) -> str:
    result = ollama_json(
        "generate",
        {
            "model": model,
            "prompt": rollout_prompt(goal, controls, history),
            "stream": False,
            "think": False,
            "options": {"num_ctx": max_prompt_tokens, "num_predict": 12, "temperature": 0},
        },
    )
    tokens = result.get("prompt_eval_count")
    if not isinstance(tokens, int) or tokens > max_prompt_tokens:
        raise ValueError("Ollama prompt token count exceeded the research limit")
    if result.get("model") != model or not isinstance(result.get("response"), str):
        raise ValueError("Invalid Ollama model response")
    prompt_lengths.append(tokens)
    return result["response"].strip()


def navigation_baseline(goal: str, controls: list[dict], history: list[dict]) -> str:
    """Exact visible target, otherwise explore an unvisited navigation control."""
    quoted = re.findall(r'"([^"]+)"', goal)
    target = quoted[-1].casefold().strip() if quoted else ""
    clicked = {row["label"].casefold().strip() for row in history}
    for index, control in enumerate(controls):
        if (
            target
            and target not in clicked
            and control["gesture"] == "click"
            and str(control["label"]).casefold().strip() == target
        ):
            return str(index)
    visual_hints = re.findall(r"\[([^\]]+)\]", goal.casefold())
    if visual_hints:
        tokens = [token for token in visual_hints[-1].split() if token not in {"span", "svg", "div"}]
        if tokens:
            for index, control in enumerate(controls):
                if control["gesture"] == "click" and tokens[-1] in str(control.get("icon_hint", "")).casefold().split():
                    return str(index)
    mentioned = []
    for index, control in enumerate(controls):
        label = str(control["label"]).strip()
        if not label or control["gesture"] != "click" or label.casefold() in clicked:
            continue
        match = re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", goal, re.IGNORECASE)
        if match:
            mentioned.append((match.start(), index))
    if mentioned:
        return str(min(mentioned)[1])
    visited = {(row["label"], row["x"], row["y"], row["gesture"]) for row in history}
    for index, control in enumerate(controls):
        if (
            control.get("has_popup") == "true"
            and control["gesture"] == "hover"
            and (control["label"], control["x"], control["y"], "hover") not in visited
        ):
            return str(index)
    for index, control in enumerate(controls):
        if (control["tag"] in {"a", "button"} or control.get("role") == "tab") and (
            control["label"],
            control["x"],
            control["y"],
            control["gesture"],
        ) not in visited:
            return str(index)
    return "0"


def rollout_prompt(goal: str, controls: list[dict], history: list[dict]) -> str:
    visible = [
        {
            "index": index,
            **{
                key: value
                for key in (
                    "label",
                    "tag",
                    "type",
                    "role",
                    "width",
                    "height",
                    "background",
                    "fill",
                    "declared_color",
                    "icon_hint",
                    "has_popup",
                    "expanded",
                    "selected",
                    "gesture",
                )
                if (value := control.get(key)) not in (None, "")
            },
        }
        for index, control in enumerate(controls)
    ]
    recent = [
        {"label": row["label"], "gesture": row["gesture"], "screen_changed": row["screen_changed"]}
        for row in history[-6:]
    ]
    return (
        "Choose one currently visible control to advance the goal. "
        "You may need to reveal a hidden target first. Use prior actions to avoid loops. "
        "Hovering a control with has_popup=true can reveal nested options without selecting it. "
        "Indices restart from zero after every action. Return only one of the current "
        f"zero-based indices 0 through {len(visible) - 1}; do not reuse an old index.\nGoal: "
        + goal
        + "\nVisible controls: "
        + json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
        + "\nRecent actions: "
        + json.dumps(recent, ensure_ascii=False, separators=(",", ":"))
        + "\nIndex:"
    )


def available_choices(controls: list[dict], history: list[dict], state_sha256: str) -> list[dict]:
    attempted = {(row["state_sha256"], row["x"], row["y"], row["gesture"]) for row in history}
    choices = []
    for control in controls:
        if control.get("has_popup") == "true":
            choices.append({**control, "gesture": "hover"})
        choices.append({**control, "gesture": "click"})
    remaining = [
        control
        for control in choices
        if (state_sha256, control["x"], control["y"], control["gesture"]) not in attempted
    ]
    untried = {(control["x"], control["y"], control["gesture"]) for control in remaining}
    for row in history:
        untried.discard((row["x"], row["y"], row["gesture"]))
    return (
        [control for control in remaining if (control["x"], control["y"], control["gesture"]) in untried]
        if untried
        else remaining
    )


def run_episode(
    browser,
    assets: PinnedAssets,
    task: str,
    seed: str,
    max_steps: int,
    choose: Callable[[str, list[dict], list[dict]], str],
) -> dict:
    context, page, _, before_png, before_dom = _reset(browser, assets, task, seed)
    try:
        goal = goal_description(page)
        initial = json.dumps(
            {
                "dom": before_dom,
                "goal": goal,
                "controls": candidate_actions(page),
                "visual_sha256": hashlib.sha256(visual_feature(before_png).tobytes()).hexdigest(),
            },
            sort_keys=True,
        )
        history: list[dict] = []
        outcome = {"done": False, "reward": None}
        stop_reason = "step_limit"
        invalid_output = None
        for _ in range(max_steps):
            before_area = page.locator("#area").inner_html()
            state_sha256 = hashlib.sha256(before_area.encode()).hexdigest()
            controls = available_choices(candidate_actions(page), history, state_sha256)
            if not controls:
                stop_reason = "no_controls"
                break
            visible_history = [{key: value for key, value in row.items() if key != "state_sha256"} for row in history]
            raw_choice = choose(goal, controls, visible_history)
            choice = parse_action_index(raw_choice, len(controls))
            if choice is None:
                stop_reason = "invalid_index"
                invalid_output = raw_choice[:80]
                break
            control = controls[choice]
            if control["gesture"] == "hover":
                page.mouse.move(control["x"], control["y"])
            else:
                page.mouse.click(control["x"], control["y"])
            outcome = page.evaluate("({done: WOB_DONE_GLOBAL, reward: WOB_RAW_REWARD_GLOBAL})")
            if not outcome["done"]:
                observed = candidate_actions(page)
                stable = 0
                for poll in range(10):
                    page.wait_for_timeout(100)
                    current = candidate_actions(page)
                    stable = stable + 1 if current == observed else 0
                    observed = current
                    if poll >= 3 and stable >= 2:
                        break
                outcome = page.evaluate("({done: WOB_DONE_GLOBAL, reward: WOB_RAW_REWARD_GLOBAL})")
            history.append(
                {
                    "label": control["label"],
                    "tag": control["tag"],
                    "gesture": control["gesture"],
                    "x": control["x"],
                    "y": control["y"],
                    "state_sha256": state_sha256,
                    "screen_changed": page.locator("#area").inner_html() != before_area,
                }
            )
            if outcome["done"]:
                stop_reason = "terminal"
                break
        return {
            "initial_sha256": hashlib.sha256(initial.encode()).hexdigest(),
            "goal": goal,
            "done": bool(outcome["done"]),
            "reward": outcome["reward"],
            "success": bool(outcome["done"] and outcome["reward"] == 1),
            "steps": len(history),
            "history": history,
            "stop_reason": stop_reason,
            "invalid_output": invalid_output,
        }
    finally:
        context.close()


def main() -> None:
    # Optional dependency (pip install -e ".[browser]"); imported here so the helpers stay importable.
    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Comma-separated pinned MiniWoB pages")
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model-snapshot", type=Path)
    model_source.add_argument("--ollama-model", type=str)
    parser.add_argument("--report", type=Path, default=root / "miniwob_multistep_qwen_v1.json")
    args = parser.parse_args()
    from scripts.collect_miniwob_branches import parse_tasks

    tasks = parse_tasks(args.tasks)
    if (
        not 1 <= args.samples_per_task <= 100
        or not 0 <= args.seed_start <= 1000000
        or not 1 <= args.max_steps <= 20
        or not 256 <= args.max_prompt_tokens <= 4096
        or (
            args.model_snapshot is not None
            and (args.model_snapshot.name not in MODEL_REVISIONS or not args.model_snapshot.is_dir())
        )
        or (args.ollama_model is not None and not re.fullmatch(r"[A-Za-z0-9_.:/-]+", args.ollama_model))
        or args.report.exists()
        or args.report.with_suffix(args.report.suffix + ".tmp").exists()
    ):
        parser.error("Use bounded tasks, an existing pinned model snapshot, and a fresh report")

    prompt_lengths: list[int] = []
    if args.ollama_model is not None:
        model_name = args.ollama_model
        revision = ollama_model_digest(model_name)
        device = "local_ollama"

        def choose_model(goal: str, controls: list[dict], history: list[dict]) -> str:
            return choose_ollama(goal, controls, history, model_name, args.max_prompt_tokens, prompt_lengths)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this probe")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_snapshot,
            local_files_only=True,
            torch_dtype="auto",
            device_map="cuda:0",
            attn_implementation="eager",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True)
        model_name = MODEL_REVISIONS[args.model_snapshot.name]
        revision = args.model_snapshot.name
        device = "cuda:0"

        def choose_model(goal: str, controls: list[dict], history: list[dict]) -> str:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": rollout_prompt(goal, controls, history)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda:0")
            prompt_tokens = int(encoded["input_ids"].shape[1])
            if prompt_tokens > args.max_prompt_tokens:
                raise ValueError(f"Multi-step prompt exceeds {args.max_prompt_tokens} tokens")
            prompt_lengths.append(prompt_tokens)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
                torch.cuda.synchronize()
            return tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    assets = PinnedAssets(SOURCE_COMMIT)
    results = []

    def save_report(*, complete: bool) -> dict:
        report = {
            "purpose": "research_only_multistep_miniwob_rollout_probe",
            "complete": complete,
            "source_repo": SOURCE_REPO,
            "source_commit": SOURCE_COMMIT,
            "source_license": "MIT",
            "asset_sha256": assets.hashes(),
            "model": model_name,
            "revision": revision,
            "device": device,
            "tasks": tasks,
            "seed_start": args.seed_start,
            "samples_per_task": args.samples_per_task,
            "max_steps": args.max_steps,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_observed_prompt_tokens": max(prompt_lengths, default=0),
            "scores": {
                task: {
                    policy: sum(row[policy]["success"] for row in results if row["task"] == task)
                    for policy in ("baseline", "model")
                }
                for task in tasks
            },
            "results": results,
            "limitation": "Synthetic browser tasks and a frozen text-only model, not real desktop "
            "screen understanding, independent task suites, safety, or Jev parity. "
            "Baseline and prompt are exploratory; no runtime promotion.",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.report)
        return report

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for task in tasks:
                for index in range(args.seed_start, args.seed_start + args.samples_per_task):
                    seed = f"{SOURCE_COMMIT}:{task}:{index}"
                    baseline = run_episode(browser, assets, task, seed, args.max_steps, navigation_baseline)
                    model_run = run_episode(browser, assets, task, seed, args.max_steps, choose_model)
                    if baseline["initial_sha256"] != model_run["initial_sha256"]:
                        raise ValueError("Policies did not start from one exact measured state")
                    results.append(
                        {
                            "task": task,
                            "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
                            "initial_sha256": baseline["initial_sha256"],
                            "baseline": baseline,
                            "model": model_run,
                        }
                    )
                    save_report(complete=False)
                print(json.dumps({"task": task, "completed": args.samples_per_task}), flush=True)
        finally:
            browser.close()
    report = save_report(complete=True)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
