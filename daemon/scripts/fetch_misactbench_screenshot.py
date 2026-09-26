"""Fetch one pinned MisActBench screenshot with bounded ZIP HTTP ranges.

This avoids downloading the 2.33 GB archive. It treats archive names and
image bytes as untrusted data, checks ZIP metadata and per-entry CRC, and
never executes image content.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256  # noqa: E402

URL = "https://huggingface.co/datasets/osunlp/MisActBench/resolve/main/trajectories.zip"
ARCHIVE_BYTES = 2_327_521_887
CENTRAL_SHA256 = "99ecd76098dc1a542febf8f7bd09d36c9ce9567b868f1f95cd7b6c46cc84e9bb"
MAX_ENTRY_BYTES = 12 * 1024 * 1024


def get_range(start: int, end: int, *, max_bytes: int) -> bytes:
    if not 0 <= start <= end < ARCHIVE_BYTES or end - start + 1 > max_bytes:
        raise ValueError("Requested ZIP range exceeds the bounded archive access")
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        expected_range = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("Archive server ignored or changed the requested byte range")
        data = response.read(max_bytes + 1)
    if len(data) != end - start + 1:
        raise ValueError("Incomplete or oversized archive response")
    return data


def parse_central(data: bytes, expected_entries: int) -> dict[str, dict]:
    result = {}
    position = 0
    fmt = "<4s6H3L5H2L"
    header_size = struct.calcsize(fmt)
    for _ in range(expected_entries):
        if position + header_size > len(data):
            raise ValueError("ZIP central directory ended early")
        fields = struct.unpack_from(fmt, data, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central header")
        name_length, extra_length, comment_length = fields[10:13]
        next_position = position + header_size + name_length + extra_length + comment_length
        if next_position > len(data):
            raise ValueError("ZIP central entry exceeds directory")
        name = data[position + header_size:position + header_size + name_length].decode("utf-8")
        member = PurePosixPath(name)
        if (member.is_absolute() or ".." in member.parts or name in result
                or fields[3] & 1 or fields[4] not in (0, 8)):
            raise ValueError("Unsafe, duplicate, or unsupported ZIP member")
        result[name] = {"compression": fields[4], "crc32": fields[7],
                        "compressed": fields[8], "uncompressed": fields[9],
                        "offset": fields[16]}
        position = next_position
    if position != len(data):
        raise ValueError("ZIP central directory has trailing bytes")
    return result


def read_index() -> dict[str, dict]:
    tail_size = 65_557
    tail = get_range(ARCHIVE_BYTES - tail_size, ARCHIVE_BYTES - 1,
                     max_bytes=tail_size)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(tail):
        raise ValueError("ZIP end record not found")
    signature, disk, central_disk, disk_entries, entries, size, offset, comment = struct.unpack_from(
        "<4s4H2LH", tail, marker)
    if (signature != b"PK\x05\x06" or disk or central_disk or disk_entries != entries
            or entries == 0xffff or size == 0xffffffff or offset == 0xffffffff
            or marker + 22 + comment != len(tail)
            or offset + size != ARCHIVE_BYTES - tail_size + marker):
        raise ValueError("Unsupported or inconsistent ZIP end record")
    central = get_range(offset, offset + size - 1, max_bytes=2_000_000)
    if hashlib.sha256(central).hexdigest() != CENTRAL_SHA256:
        raise ValueError("MisActBench archive central directory changed")
    return parse_central(central, entries)


def member_bytes(name: str, metadata: dict) -> bytes:
    if metadata["compressed"] > MAX_ENTRY_BYTES or metadata["uncompressed"] > MAX_ENTRY_BYTES:
        raise ValueError("Screenshot exceeds the bounded per-entry size")
    start = metadata["offset"]
    local = get_range(start, start + 29, max_bytes=30)
    fields = struct.unpack("<4s5H3L2H", local)
    if fields[0] != b"PK\x03\x04" or fields[3] != metadata["compression"] or fields[2] & 1:
        raise ValueError("ZIP local header differs from central directory")
    name_length, extra_length = fields[9:11]
    local_name = get_range(start + 30, start + 29 + name_length,
                           max_bytes=4_096).decode("utf-8")
    if local_name != name:
        raise ValueError("ZIP local filename differs from central directory")
    payload_start = start + 30 + name_length + extra_length
    compressed = get_range(payload_start, payload_start + metadata["compressed"] - 1,
                           max_bytes=MAX_ENTRY_BYTES)
    if metadata["compression"] == 8:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        content = decoder.decompress(compressed, MAX_ENTRY_BYTES + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("ZIP deflate stream is invalid")
    else:
        content = compressed
    if (len(content) != metadata["uncompressed"] or len(content) > MAX_ENTRY_BYTES
            or zlib.crc32(content) != metadata["crc32"]
            or not content.startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Screenshot CRC, size, or PNG signature is invalid")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-id", help="Annotated trajectory UUID:step index")
    selection.add_argument("--manifest", type=Path, help="Pre-registered case ID list")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prior-steps", action="store_true",
                        help="Fetch the screenshot before each manifest action's preceding step")
    parser.add_argument("--list-first", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not args.case_id and not args.manifest and not args.list_first:
        parser.error("Choose --case-id, --manifest, or --list-first")
    if args.prior_steps and not args.manifest:
        parser.error("Prior-step extraction requires a manifest")
    if (args.case_id or args.manifest) and not args.output_dir:
        parser.error("Output directory is required for extraction")
    if args.list_first < 0 or args.list_first > 10:
        parser.error("List between zero and ten names")
    if args.progress_every < 1:
        parser.error("Progress interval must be positive")
    if not 1 <= args.workers <= 8:
        parser.error("Use between one and eight bounded range workers")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("MisActBench annotations differ from the pinned release")
    index = read_index()
    if args.list_first:
        print(json.dumps(list(index)[:args.list_first], indent=2))
    case_ids = [args.case_id] if args.case_id else []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        case_ids = manifest["case_ids"]
        if not 1 <= len(case_ids) <= 512 or len(set(case_ids)) != len(case_ids):
            raise ValueError("Manifest must contain 1-512 unique cases")
        if args.prior_steps:
            case_ids = [f"{trajectory_id}:{int(step_id) - 1}"
                        for trajectory_id, step_id in (case_id.split(":", 1) for case_id in case_ids)]
    if case_ids:
        records = json.loads(args.dataset.read_text(encoding="utf-8"))
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    def fetch_case(case_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f-]+:[0-9]+", case_id):
            raise ValueError("Invalid case ID")
        trajectory_id, step_id = case_id.split(":", 1)
        record = records[trajectory_id]["steps"][step_id]
        if not args.prior_steps and not isinstance(record["label"], bool):
            raise ValueError("Case has no expert action label")
        name = record["screenshot_path"].removeprefix("MisActBench/")
        if name not in index:
            raise ValueError("Annotated screenshot is missing from archive")
        metadata = index[name]
        target = output_dir / (case_id.replace(":", "_") + ".png")
        if target.exists():
            content = target.read_bytes()
            if (len(content) != metadata["uncompressed"]
                    or zlib.crc32(content) != metadata["crc32"]
                    or not content.startswith(b"\x89PNG\r\n\x1a\n")):
                raise FileExistsError("Existing screenshot fails pinned ZIP checks")
        else:
            content = member_bytes(name, metadata)
            target.write_bytes(content)
        return {"case_id": case_id, "member": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content), "path": str(target)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(fetch_case, case_ids)
        for position, result in enumerate(results, 1):
            if args.progress_every == 1:
                print(json.dumps(result), flush=True)
            elif position % args.progress_every == 0 or position == len(case_ids):
                print(json.dumps({"completed": position, "total": len(case_ids)}), flush=True)


if __name__ == "__main__":
    main()
