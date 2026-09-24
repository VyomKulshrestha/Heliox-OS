"""Score a local Ollama model on frozen scope cases without executing actions.

The model is required to be fully GPU-resident. Its digest and the fixed
prompt are recorded so a report cannot silently mix different checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from scripts.evaluate_openjev_checkpoint import SUITES, scopejudge_cases
from scripts.evaluate_qwen_instruct_scope import (
    SYSTEM_PROMPT,
    case_signature,
    load_partial,
    messages,
    parse_answer,
    save_partial,
    summarize,
)

LOCAL_OPENER = build_opener(ProxyHandler({}))


def api(base_url: str, path: str, payload: dict | None = None, *, timeout: int = 180) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{base_url.rstrip('/')}{path}", data=body,
                      headers={"Content-Type": "application/json"})
    with LOCAL_OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def model_digest(tags: dict, name: str) -> str:
    matches = [item["digest"] for item in tags.get("models", []) if item.get("name") == name]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"Ollama model {name!r} is not uniquely installed")
    return matches[0]


def gpu_residency(processes: dict, name: str, digest: str) -> dict:
    matches = [item for item in processes.get("models", [])
               if item.get("name") == name and item.get("digest") == digest]
    if len(matches) != 1:
        raise RuntimeError("Expected the pinned model to be loaded once")
    model = matches[0]
    size = model.get("size")
    vram = model.get("size_vram")
    if not isinstance(size, int) or size <= 0 or vram != size:
        raise RuntimeError("Pinned model is not fully GPU-resident")
    return {"size_bytes": size, "size_vram_bytes": vram,
            "context_length": model.get("context_length")}


def possibly_truncated(prompt_tokens: int, context_tokens: int, output_tokens: int) -> bool:
    return prompt_tokens >= context_tokens - output_tokens


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Already-installed Ollama model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--suite", choices=(*SUITES, "scopejudge-test"), default="qwen-v1")
    parser.add_argument("--scopejudge-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/ollama_scope_baseline.json")
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--base-url must be a local HTTP endpoint")
    if args.num_ctx < 256:
        parser.error("--num-ctx must be at least 256")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.suite == "scopejudge-test":
        if args.scopejudge_path is None:
            parser.error("--scopejudge-path is required for scopejudge-test")
        cases, manifest = scopejudge_cases(args.scopejudge_path, "scope_test")
    else:
        cases, manifest = SUITES[args.suite], {"suite": args.suite}
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            parser.error("--limit must be within the selected suite")
        cases = cases[:args.limit]
    if args.report.exists():
        raise FileExistsError("Final report already exists")
    digest = model_digest(api(args.base_url, "/api/tags"), args.model)
    ollama_version = api(args.base_url, "/api/version")["version"]
    identity = {"model": args.model, "digest": digest, "suite": args.suite,
                "ollama_version": ollama_version,
                "case_signature": case_signature(cases),
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "num_ctx": args.num_ctx, "num_predict": 16, "temperature": 0,
                "think": False}
    partial = args.report.with_suffix(".partial.json")
    if partial.exists() and not args.resume:
        raise FileExistsError(f"Checkpoint exists at {partial}; pass --resume")
    rows = load_partial(partial, identity, cases) if partial.exists() else []
    started = time.monotonic()
    residency = None
    for index in range(len(rows), len(cases)):
        case = cases[index]
        try:
            response = api(args.base_url, "/api/chat", {
                "model": args.model, "messages": messages(case.request, case.action),
                "stream": False, "think": False, "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": 16, "num_ctx": args.num_ctx},
            })
        except HTTPError as exc:
            raise RuntimeError(f"Ollama returned HTTP {exc.code} at case {case.id}; "
                               f"completed rows remain in {partial}") from exc
        if response.get("model") != args.model:
            raise RuntimeError("Ollama served a different model")
        residency = gpu_residency(api(args.base_url, "/api/ps"), args.model, digest)
        raw = response["message"]["content"]
        answer = parse_answer(raw)
        input_tokens = int(response["prompt_eval_count"])
        may_be_truncated = possibly_truncated(input_tokens, args.num_ctx, identity["num_predict"])
        rows.append({"id": case.id, "gold_in_scope": case.in_scope,
                     "predicted_in_scope": None if may_be_truncated else
                     (answer if answer is not None else False),
                     "answer": answer, "raw": raw[:200], "input_tokens": input_tokens,
                     "generated_tokens": int(response["eval_count"]),
                     "total_duration_ns": int(response["total_duration"]),
                     "status": "possibly_truncated" if may_be_truncated else
                     ("ok" if answer is not None else "malformed")})
        save_partial(partial, identity, rows)
        if len(rows) % args.progress_every == 0 or len(rows) == len(cases):
            print(f"{len(rows)}/{len(cases)} {summarize(rows)}", flush=True)
    if residency is None:
        residency = gpu_residency(api(args.base_url, "/api/ps"), args.model, digest)
    report = {"purpose": "research_only_local_quantized_scope_baseline",
              "identity": identity, "gpu_residency": residency,
              "dataset": manifest,
              "summary": summarize(rows), "elapsed_seconds": time.monotonic() - started,
              "results": rows,
              "limitation": "Previously inspected synthetic cases; not a fresh holdout, Jev parity, or desktop safety proof."}
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    partial.unlink(missing_ok=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
