#!/usr/bin/env python3
"""
Staggered load tester for the Adaptive Booth API.

Difference from load_test.py
-----------------------------
  load_test.py   – fires ALL instances in a batch simultaneously.
  load_test_2.py – fires one request every --interval seconds within each batch.
                   Requests CAN overlap (previous one still running when next fires).
                   Each request's duration is measured from when IT was fired,
                   not from the batch start.

This simulates realistic user arrival patterns and lets you see how individual
response times change as concurrency builds up.

Usage
-----
    python load_test_2.py \\
        --input-folder /path/to/images \\
        --interval 5.0 \\
        --instances 4 \\
        --endpoint predict \\
        --host http://localhost:8000 \\
        --api-key YOUR_KEY

    # Stress: cycle through the folder 3×
    python load_test_2.py ... --repeat 3

Arguments
---------
  --input-folder / -i   Folder with input files (images / PDFs)
  --interval    / -t    Seconds between firing each request in a batch (default 5.0)
  --instances   / -n    Requests per batch                             (default 4)
  --batch-wait  / -b    Extra wait between batches (seconds)          (default 0.0)
  --endpoint    / -e    Endpoint name                                  (default predict)
  --host                API base URL                         (default http://localhost:8000)
  --api-key             Authentication-API-Key header value            (required)
  --repeat              Times to repeat the full file list             (default 1)
  --timeout             Per-request timeout in seconds                 (default 300)
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
# Single request
# ──────────────────────────────────────────────────────────────────────────────

def _mime(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".pdf":            return "application/pdf"
    if s in (".jpg", ".jpeg"): return "image/jpeg"
    if s == ".png":            return "image/png"
    return "application/octet-stream"


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    file_path: Path,
) -> dict:
    """POST one file. Returns timing dict (elapsed = from when THIS call started)."""
    t0 = time.perf_counter()
    try:
        raw   = await asyncio.to_thread(file_path.read_bytes)
        files = {"file": (file_path.name, raw, _mime(file_path))}
        resp  = await client.post(url, headers=headers, files=files)
        elapsed = time.perf_counter() - t0
        ok = resp.status_code == 200
        return {
            "file":    file_path.name,
            "status":  resp.status_code,
            "elapsed": elapsed,
            "success": ok,
            "error":   None if ok else resp.text[:300],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "file":    file_path.name,
            "status":  None,
            "elapsed": elapsed,
            "success": False,
            "error":   str(exc)[:300],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Staggered batch
# ──────────────────────────────────────────────────────────────────────────────

async def fire_staggered_batch(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    batch: list,          # list of Path
    interval: float,      # seconds between firing each request
    req_id_start: int,    # for labelling (#01, #02, …)
    wall_start: float,    # time.perf_counter() at test start
) -> list:
    """
    Fire `batch` requests one by one, `interval` seconds apart.
    All outstanding requests overlap.  Each result includes:
      fired_at  – seconds since test start when THIS request was sent
      done_at   – seconds since test start when response arrived
      elapsed   – individual duration (done_at - fired_at)
    """
    pending_tasks = []

    for slot, fp in enumerate(batch):
        # Stagger: wait `interval` before each request (except the first)
        if slot > 0:
            await asyncio.sleep(interval)

        fired_at = time.perf_counter() - wall_start
        req_id   = req_id_start + slot

        print(
            f"  {_GRAY}[t={fired_at:6.1f}s]{_RESET}"
            f"  {_BOLD}#{req_id:02d}{_RESET} → {fp.name}",
            flush=True,
        )

        # Capture loop variables for the closure
        async def _task(fp=fp, req_id=req_id, fired_at=fired_at):
            result   = await send_one(client, url, headers, fp)
            done_at  = time.perf_counter() - wall_start
            ok       = result["success"]
            tag      = f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"
            st       = str(result["status"]) if result["status"] else "ERR"
            err_hint = f"  ← {result['error'][:60]}" if result["error"] else ""
            print(
                f"  {_GRAY}[t={done_at:6.1f}s]{_RESET}  "
                f"{tag} {_BOLD}#{req_id:02d}{_RESET} {fp.name:<38} "
                f"{st}  individual={result['elapsed']:.2f}s{err_hint}",
                flush=True,
            )
            return {**result, "fired_at": fired_at, "done_at": done_at}

        pending_tasks.append(asyncio.create_task(_task()))

    # Wait for all requests in this batch to complete
    results = await asyncio.gather(*pending_tasks)
    return list(results)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def _pct(sorted_data: list, p: float) -> float:
    if not sorted_data:
        return 0.0
    return sorted_data[max(0, int(len(sorted_data) * p / 100) - 1)]


def print_summary(all_results: list, wall_time: float, interval: float):
    total   = len(all_results)
    ok      = [r for r in all_results if r["success"]]
    failed  = [r for r in all_results if not r["success"]]

    # elapsed = individual duration for each request
    all_t   = [r["elapsed"] for r in all_results]
    ok_t    = [r["elapsed"] for r in ok]

    print(f"\n{'='*64}")
    print(f"{_BOLD}SUMMARY  (staggered mode, interval={interval}s){_RESET}")
    print(f"{'='*64}")
    print(f"  Total requests  : {total}")
    print(f"  {_GREEN}Successful{_RESET}      : {len(ok)}")
    if failed:
        print(f"  {_RED}Failed{_RESET}          : {len(failed)}")
    print(f"  Total wall time : {wall_time:.2f}s")
    if total:
        print(f"  Throughput      : {total / wall_time:.3f} req/s")

    def _stats(label: str, data: list):
        if not data:
            return
        s = sorted(data)
        print(f"\n  {_CYAN}{label}{_RESET} (n={len(data)}):")
        print(f"    Min (fastest)  : {min(data):.2f}s")
        print(f"    Max (slowest)  : {max(data):.2f}s")
        print(f"    Mean           : {statistics.mean(data):.2f}s")
        if len(data) >= 2:
            print(f"    Median         : {statistics.median(data):.2f}s")
            print(f"    P95            : {_pct(s, 95):.2f}s")
            print(f"    P99            : {_pct(s, 99):.2f}s")
            print(f"    Std dev        : {statistics.stdev(data):.2f}s")

    _stats("Individual request duration (all)", all_t)
    if ok_t and len(ok_t) < len(all_t):
        _stats("Individual request duration (successful)", ok_t)

    # Per-request timeline table
    print(f"\n  {_CYAN}Request timeline{_RESET}")
    print(f"  {'#':>4}  {'File':<38}  {'Fired':>8}  {'Done':>8}  {'Duration':>10}  Status")
    print(f"  {'-'*4}  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*10}  ------")
    for r in all_results:
        ok_s   = f"{_GREEN}OK  {_RESET}" if r["success"] else f"{_RED}FAIL{_RESET}"
        req_id = r.get("req_id", "?")
        print(
            f"  {req_id:>4}  {r['file']:<38}  "
            f"{r.get('fired_at', 0):7.1f}s  "
            f"{r.get('done_at', 0):7.1f}s  "
            f"{r['elapsed']:9.2f}s  [{ok_s}]"
        )

    if failed:
        print(f"\n  {_RED}Failures:{_RESET}")
        for r in failed:
            print(f"    {r['file']}: HTTP {r['status']} — {r['error']}")

    print(f"{'='*64}")


# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────

async def run(
    input_folder: str,
    interval: float,
    instances: int,
    batch_wait: float,
    endpoint: str,
    host: str,
    api_key: str,
    repeat: int,
    timeout: float,
):
    files     = collect_files(input_folder)
    all_files = files * repeat
    batches   = [all_files[i:i + instances] for i in range(0, len(all_files), instances)]
    url       = f"{host.rstrip('/')}/{endpoint}"
    headers   = {"Authentication-API-Key": api_key}
    req_counter = 1

    total_reqs  = len(all_files)
    fire_span   = (instances - 1) * interval  # time to fire all reqs in one batch

    print(f"\n{_BOLD}Staggered load test{_RESET}")
    print(f"  Endpoint        : {_CYAN}{url}{_RESET}")
    print(f"  Input folder    : {input_folder}  ({len(files)} file(s))")
    print(f"  Instances/batch : {instances}  (one fired every {interval}s  → last fires at t={fire_span:.1f}s)")
    print(f"  Batch wait      : {batch_wait}s between batches")
    print(f"  Repeat          : {repeat}x  →  {total_reqs} total requests, {len(batches)} batch(es)")
    print(f"  Per-req timeout : {timeout}s")
    print()

    all_results: list = []
    wall_start  = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        for batch_idx, batch in enumerate(batches):

            if batch_idx > 0 and batch_wait > 0:
                print(f"\n  {_GRAY}Waiting {batch_wait}s before next batch…{_RESET}", flush=True)
                await asyncio.sleep(batch_wait)

            n = len(batch)
            print(
                f"\n{_BOLD}── Batch {batch_idx + 1}/{len(batches)}"
                f"  ({n} requests, staggered every {interval}s){_RESET}",
                flush=True,
            )
            t_batch = time.perf_counter()

            results = await fire_staggered_batch(
                client, url, headers, batch,
                interval, req_counter, wall_start,
            )

            # Tag each result with its request id for the timeline table
            for slot, r in enumerate(results):
                r["req_id"] = f"#{req_counter + slot:02d}"

            req_counter += n
            elapsed_batch = time.perf_counter() - t_batch
            all_results.extend(results)

            ok_n = sum(1 for r in results if r["success"])
            durations = [r["elapsed"] for r in results]
            mean_dur  = statistics.mean(durations) if durations else 0.0
            print(
                f"\n  {_BOLD}Batch {batch_idx + 1} done{_RESET}  "
                f"→ wall={elapsed_batch:.2f}s  "
                f"mean individual={mean_dur:.2f}s  "
                f"({ok_n}/{n} OK)",
                flush=True,
            )

    print_summary(all_results, time.perf_counter() - wall_start, interval)


def main():
    parser = argparse.ArgumentParser(
        description="Staggered load tester — fires requests one at a time within each batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-folder", "-i", required=True,
                        help="Folder of input files (images / PDFs)")
    parser.add_argument("--interval", "-t", type=float, default=5.0,
                        help="Seconds between firing each request within a batch")
    parser.add_argument("--instances", "-n", type=int, default=4,
                        help="Number of requests per batch")
    parser.add_argument("--batch-wait", "-b", type=float, default=0.0,
                        help="Extra seconds to wait between batches")
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

    args = parser.parse_args()
    if args.instances < 1:
        parser.error("--instances must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    asyncio.run(run(
        input_folder = args.input_folder,
        interval     = args.interval,
        instances    = args.instances,
        batch_wait   = args.batch_wait,
        endpoint     = args.endpoint,
        host         = args.host,
        api_key      = args.api_key,
        repeat       = args.repeat,
        timeout      = args.timeout,
    ))


if __name__ == "__main__":
    main()
