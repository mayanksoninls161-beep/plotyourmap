#!/usr/bin/env python3
"""
Load tester for the Adaptive Booth API.

Sends requests from an input folder, controls how many fire concurrently
(--instances) and how long to wait between batches (--interval). Prints a
live line the moment each request starts and finishes, plus a heartbeat
every 10 s so you can always see that work is in progress.

Usage
-----
    python load_test.py \\
        --input-folder /path/to/images \\
        --interval 2.0 \\
        --instances 4 \\
        --endpoint predict \\
        --host http://localhost:8000 \\
        --api-key YOUR_KEY

    # Stress test: repeat the file list 3×
    python load_test.py ... --repeat 3

Arguments
---------
  --input-folder / -i   Folder with input files (images / PDFs)
  --interval    / -t    Seconds between batches              (default 1.0)
  --instances   / -n    Concurrent requests per batch        (default 2)
  --endpoint    / -e    Endpoint name                        (default predict)
  --host                API base URL                         (default http://localhost:8000)
  --api-key             Authentication-API-Key header value  (required)
  --repeat              Times to loop the file list          (default 1)
  --timeout             Per-request timeout in seconds       (default 300)
  --heartbeat           Heartbeat print interval in seconds  (default 10)
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".tif", ".bmp"}
VALID_ENDPOINTS = ("predict", "hall_with_booth_predict", "debug_predict")

# ANSI colours (disabled on non-TTY)
_TTY   = sys.stdout.isatty()
_GREEN = "\033[32m" if _TTY else ""
_RED   = "\033[31m" if _TTY else ""
_CYAN  = "\033[36m" if _TTY else ""
_GRAY  = "\033[90m" if _TTY else ""
_BOLD  = "\033[1m"  if _TTY else ""
_RESET = "\033[0m"  if _TTY else ""


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def collect_files(folder: str) -> list:
    p = Path(folder)
    if not p.is_dir():
        sys.exit(f"Error: {folder!r} is not a directory")
    files = sorted(
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        sys.exit(
            f"No supported files found in {folder!r}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Single request  (with live start/finish prints)
# ──────────────────────────────────────────────────────────────────────────────

def _mime(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".pdf":                return "application/pdf"
    if s in (".jpg", ".jpeg"):     return "image/jpeg"
    if s == ".png":                return "image/png"
    return "application/octet-stream"


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    file_path: Path,
    req_id: int,
) -> dict:
    name = file_path.name
    print(f"  {_GRAY}[#{req_id:02d} START]{_RESET} {name}", flush=True)
    t0 = time.perf_counter()
    try:
        # Read in a thread so large files don't block the event loop
        raw   = await asyncio.to_thread(file_path.read_bytes)
        files = {"file": (name, raw, _mime(file_path))}
        resp  = await client.post(url, headers=headers, files=files)
        elapsed = time.perf_counter() - t0
        ok = resp.status_code == 200
        tag = f"{_GREEN}OK  {_RESET}" if ok else f"{_RED}FAIL{_RESET}"
        err_snippet = "" if ok else f"  ← {resp.text[:80]}"
        print(
            f"  [{tag}][#{req_id:02d}] {name:<38} "
            f"{resp.status_code}  {elapsed:.2f}s{err_snippet}",
            flush=True,
        )
        return {
            "file": name, "status": resp.status_code,
            "elapsed": elapsed, "success": ok,
            "error": None if ok else resp.text[:300],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        msg = str(exc)[:200]
        print(
            f"  [{_RED}ERR {_RESET}][#{req_id:02d}] {name:<38} "
            f"ERR   {elapsed:.2f}s  ← {msg}",
            flush=True,
        )
        return {
            "file": name, "status": None,
            "elapsed": elapsed, "success": False, "error": msg,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Heartbeat — prints a progress line every N seconds while batch runs
# ──────────────────────────────────────────────────────────────────────────────

async def _heartbeat(batch_start: float, interval: float, inflight: list):
    """Prints elapsed time and how many requests are still running."""
    while True:
        await asyncio.sleep(interval)
        elapsed = time.perf_counter() - batch_start
        remaining = sum(1 for done in inflight if not done)
        if remaining:
            print(
                f"  {_GRAY}... {elapsed:.0f}s elapsed, "
                f"{remaining} request(s) still in-flight ...{_RESET}",
                flush=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pct(sorted_data: list, p: float) -> float:
    if not sorted_data:
        return 0.0
    return sorted_data[max(0, int(len(sorted_data) * p / 100) - 1)]


def _stats_block(label: str, data: list):
    if not data:
        return
    print(f"\n  {_CYAN}{label}{_RESET} (n={len(data)}):")
    print(f"    Min    : {min(data):.2f}s")
    print(f"    Max    : {max(data):.2f}s")
    print(f"    Mean   : {statistics.mean(data):.2f}s")
    if len(data) >= 2:
        print(f"    Median : {statistics.median(data):.2f}s")
        print(f"    P95    : {_pct(sorted(data), 95):.2f}s")
        print(f"    P99    : {_pct(sorted(data), 99):.2f}s")


def print_summary(all_results: list, wall_time: float):
    total  = len(all_results)
    ok     = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    all_t  = [r["elapsed"] for r in all_results]
    ok_t   = [r["elapsed"] for r in ok]

    print(f"\n{'='*62}")
    print(f"{_BOLD}SUMMARY{_RESET}")
    print(f"{'='*62}")
    print(f"  Total requests  : {total}")
    print(f"  {_GREEN}Successful{_RESET}      : {len(ok)}")
    if failed:
        print(f"  {_RED}Failed{_RESET}          : {len(failed)}")
    print(f"  Total wall time : {wall_time:.2f}s")
    print(f"  Throughput      : {total / wall_time:.2f} req/s")

    _stats_block("All requests", all_t)
    if ok_t and len(ok_t) < len(all_t):
        _stats_block("Successful only", ok_t)

    if failed:
        print(f"\n  {_RED}Failures:{_RESET}")
        for r in failed:
            print(f"    {r['file']}: HTTP {r['status']} — {r['error']}")

    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────

async def run(
    input_folder: str,
    interval: float,
    instances: int,
    endpoint: str,
    host: str,
    api_key: str,
    repeat: int,
    timeout: float,
    heartbeat_interval: float,
):
    files     = collect_files(input_folder)
    all_files = files * repeat
    batches   = [all_files[i:i + instances] for i in range(0, len(all_files), instances)]
    url       = f"{host.rstrip('/')}/{endpoint}"
    headers   = {"Authentication-API-Key": api_key}
    req_counter = 0

    print(f"\n{_BOLD}Load test configuration{_RESET}")
    print(f"  Endpoint        : {_CYAN}{url}{_RESET}")
    print(f"  Input folder    : {input_folder}  ({len(files)} file(s))")
    print(f"  Instances/batch : {instances}")
    print(f"  Interval        : {interval}s between batches")
    print(f"  Repeat          : {repeat}x  →  {len(all_files)} total requests, {len(batches)} batch(es)")
    print(f"  Per-req timeout : {timeout}s")
    print(f"  Heartbeat every : {heartbeat_interval}s")
    print()

    all_results = []
    wall_start  = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        for idx, batch in enumerate(batches):

            if idx > 0 and interval > 0:
                print(f"  {_GRAY}Waiting {interval}s before next batch…{_RESET}", flush=True)
                await asyncio.sleep(interval)

            n = len(batch)
            print(
                f"\n{_BOLD}── Batch {idx + 1}/{len(batches)}"
                f"  ({n} concurrent){_RESET}",
                flush=True,
            )
            t_batch = time.perf_counter()

            # Track which coroutines have finished (for heartbeat)
            inflight = [False] * n

            async def _run_one(fp, slot, rid):
                result = await send_one(client, url, headers, fp, rid)
                inflight[slot] = True
                return result

            req_ids = list(range(req_counter + 1, req_counter + n + 1))
            req_counter += n

            coros = [_run_one(fp, slot, rid)
                     for slot, (fp, rid) in enumerate(zip(batch, req_ids))]

            hb = asyncio.create_task(
                _heartbeat(t_batch, heartbeat_interval, inflight)
            )
            try:
                results = await asyncio.gather(*coros)
            finally:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass

            elapsed_batch = time.perf_counter() - t_batch
            all_results.extend(results)
            ok_n = sum(1 for r in results if r["success"])
            print(
                f"  {_BOLD}Batch {idx + 1} done{_RESET}  "
                f"→ {elapsed_batch:.2f}s total  "
                f"({ok_n}/{n} OK)",
                flush=True,
            )

    print_summary(all_results, time.perf_counter() - wall_start)


def main():
    parser = argparse.ArgumentParser(
        description="Load tester for the Adaptive Booth API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-folder", "-i", required=True,
                        help="Folder of input files (images / PDFs)")
    parser.add_argument("--interval", "-t", type=float, default=1.0,
                        help="Seconds between batches")
    parser.add_argument("--instances", "-n", type=int, default=2,
                        help="Concurrent requests per batch")
    parser.add_argument("--endpoint", "-e", default="predict",
                        choices=VALID_ENDPOINTS,
                        help="API endpoint name")
    parser.add_argument("--host", default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--api-key", required=True,
                        help="Authentication-API-Key header value")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Times to repeat the file list")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds")
    parser.add_argument("--heartbeat", type=float, default=10.0,
                        dest="heartbeat",
                        help="Seconds between heartbeat prints while waiting")

    args = parser.parse_args()
    if args.instances < 1:
        parser.error("--instances must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    asyncio.run(run(
        input_folder       = args.input_folder,
        interval           = args.interval,
        instances          = args.instances,
        endpoint           = args.endpoint,
        host               = args.host,
        api_key            = args.api_key,
        repeat             = args.repeat,
        timeout            = args.timeout,
        heartbeat_interval = args.heartbeat,
    ))


if __name__ == "__main__":
    main()
