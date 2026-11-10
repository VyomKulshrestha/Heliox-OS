# Everyday planner eval

Measures whether Heliox's planner turns everyday requests into the smallest correct plan of typed
actions. It calls a real model, so it is opt-in and never runs in CI. It never executes a plan: the
runner calls `Planner.plan` in-process and scores the plan it returns.

## Files

- `daemon/benchmarks/planner_everyday_v1.json`: 51 requests. The first 28 are the correct cases of
  `gnomon_everyday_review_v1.json`; 23 more cover appending and prepending a line, replacing a file,
  rename, move, copy, folder delete, a new empty folder, volume, brightness, mute, clipboard, window
  chores, two- and three-part requests, removing a line, the launch-folder release job and the
  launch-folder sentence from `tests/test_multi_part_file_jobs.py`. Paths use `{home}`.
- `daemon/benchmarks/planner_everyday.py`: the acceptance checker (no model, no execution).
- `daemon/scripts/run_planner_everyday_eval.py`: the runner.
- `daemon/tests/test_planner_everyday_eval.py`: unit tests for the checker and the planner rules.

## What a case says

- `steps`: the steps a correct plan must contain, with their key parameters, in order. A parameter
  rule is a path (`path`, `endswith`), words the value contains, a regex, a number, `same_dir_as`
  another parameter (a rename stays in its folder) or `has_placeholder` (an append keeps the old text).
- `alternatives`: other step lists that are just as correct (mute with `volume_mute` or
  `volume_set` level 0; copy into `Documents` or to `Documents\screenshots`).
- `allowed`, `allow_lookup`: extra types a plan may use; `allow_lookup` admits read-only steps.
- `allow_ask`: the request does not say which file, so a question with no actions passes.
- `max_steps`, `code_ok` (shell or code types the job needs because no typed action does it),
  `not_undoable_ok` (types with no undoable alternative, such as `process_kill`) and `forbid`
  (steps that fail the plan, such as a shell script that deletes).

## Rules scored

| rule | passes when |
| --- | --- |
| types | every step uses a type the case allows |
| params | every required step is present with the right path, destination, text or level |
| order | the required steps appear in order, and a step that uses the previous output follows a step that makes one |
| max_steps | the plan is no longer than the case allows |
| no_code | no shell or code step unless the case needs one, and no forbidden step |
| undo | every step that changes something is one `undo_support` can reverse, unless the case exempts its type |

A plan passes when no rule fails. A planner error fails every rule.

## Running it

From `daemon/`:

```
.venv/Scripts/python.exe scripts/run_planner_everyday_eval.py --provider openrouter --model "~deepseek/deepseek-v4-flash-latest" --call-seconds 120 --concurrency 3 --out ../planner-eval-flash.json
.venv/Scripts/python.exe scripts/run_planner_everyday_eval.py --provider claude --model claude-opus-5-5 --sample opus15 --max-calls 20 --concurrency 2 --out ../planner-eval-opus.json
.venv/Scripts/python.exe scripts/run_planner_everyday_eval.py --rescore ../planner-eval-flash.json
```

`--ids` picks cases, `--repeat` plans each request several times, `--rescore` scores a saved run
again with the current cases and no model call, `--retry-errors` plans again only the requests a
saved run lost to provider errors, and `--call-seconds` raises the per-call time limit (the daemon
uses 35 seconds; a slow cheap model needs about 120 to finish a multi-part plan).

The model settings live in memory only; keys come from the key vault; usage is recorded in Heliox's budget, so the monthly cap applies, and the run
stops if the budget refuses a call. The LLM response cache is bypassed. A request that fails because
the provider timed out is retried twice after 35 seconds.

The planner sees what the daemon would give it: the system prompt, specialist hints, the skills
block and the listing of folders the request names (`request_folders`). It sees no memory, chat
history, screen text, plugins or MCP tools, so results describe a fresh request.

## Results on 10 November 2026

The cheap model ran with `--call-seconds 120` (before: 35 seconds, then the provider errors retried
at 120). A pass is no failed rule; an "asked" plan counts as a pass only where the case allows asking.

| model | requests | before | after | model calls |
| --- | --- | --- | --- | --- |
| `~deepseek/deepseek-v4-flash-latest` (OpenRouter) | 51 | 20/51 | 47/51 | 80 before, 48 after |
| `claude-opus-5-5` | 15 (`--sample opus15`) | 6/15 | 15/15 | 20 before, 15 after |

"Before" is commit c3bd4ed. The most common failures before:

- A setting or app chore done with a shell or code step (volume, brightness, mute, clipboard, quit an
  app, install): the prompt never listed `volume_set`, `brightness_set`, `volume_mute`,
  `clipboard_write`, the window actions, `process_kill` or `package_install`.
- An append that would have replaced the file. Both models planned `file_read` then a write of
  `{PREV_OUTPUT}` plus the new line, and the planner's post-processor deleted the leading `file_read`
  because a later step wrote the same path. That is the 10 November `checklist.txt` plan: the model
  returned `file_read, code_execute, file_write, file_read`, the post-processor removed the first
  step, and the plan card showed what was left. The UI lists steps in plan order.
- Window chores done by clicking detected buttons instead of `window_minimize`, `window_maximize` or `window_focus`.
- Plans that only search for a file whose location the request does not give.
- A sentence of prose before the JSON plan, which failed parsing twice and lost the request.

Still failing after: `g10` and `g11` list a folder and stop; `g21` (latest Node installer) and `g23`
(copy the meeting link) use code to find a URL, because `download_file` and `clipboard_write`
cannot take a value from an earlier step except through `{PREV_OUTPUT}`.
