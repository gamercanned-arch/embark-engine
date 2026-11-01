"""
EMBARK Engine -- Live Server TPS & TTFT Benchmark
Sends real prompts via OpenAI-compatible SSE streaming to http://127.0.0.1:8080/v1
Measures:
  - TTFT  : time from request sent to first token received (ms)
  - TPS   : decode tokens per second (tokens / (last_token_t - first_token_t))
  - ITL   : per-token inter-token latency (P50, P95, P99) in ms
"""

import json
import time
import statistics
import socket
import sys

BASE_HOST = "127.0.0.1"
BASE_PORT = 8080
PATH      = "/v1/chat/completions"
ROUNDS    = 3
MAX_TOKENS = 200

MODELS = ["lfm2-350m", "lfm3-vl-2b", "groot-19b"]

PROMPTS = [
    ("short",  "What is 2 + 2?"),
    ("medium", "Explain the difference between RAM and storage in 3 sentences."),
    ("long",   "Write a Python function that implements binary search on a sorted list, "
               "with proper docstring and error handling."),
]


def pct(data, p):
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def safe_mean(lst):
    v = [x for x in lst if x == x]  # drop NaN
    return statistics.mean(v) if v else float("nan")


def safe_std(lst):
    v = [x for x in lst if x == x]
    return statistics.stdev(v) if len(v) > 1 else 0.0


def stream_completion_raw(model, content, max_tokens=MAX_TOKENS):
    """
    Low-level raw socket SSE reader.
    Returns (ttft_ms, tps, itl_list_ms, token_count, full_text).
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }).encode("utf-8")

    http_req = (
        f"POST {PATH} HTTP/1.1\r\n"
        f"Host: {BASE_HOST}:{BASE_PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii") + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect((BASE_HOST, BASE_PORT))

    t_send = time.perf_counter()
    sock.sendall(http_req)

    # Receive everything
    buf = b""
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        except socket.timeout:
            break
    sock.close()

    # Split HTTP headers from body
    header_end = buf.find(b"\r\n\r\n")
    if header_end == -1:
        return float("nan"), float("nan"), [], 0, ""
    body = buf[header_end + 4:]

    # Decode body (handle chunked transfer encoding)
    body_str = body.decode("utf-8", errors="replace")
    # Strip chunked size lines (hex digits followed by \r\n)
    import re
    body_str = re.sub(r"^[0-9a-fA-F]+\r\n", "", body_str, flags=re.MULTILINE)

    ttft_ms = None
    token_times = []
    full_text = ""

    # Parse SSE lines
    # We don't have real per-token timestamps from a buffered bulk read,
    # so we reprocess with a streaming socket approach below.
    # But first, try to count tokens from bulk body.
    lines = body_str.splitlines()
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {})
        piece = delta.get("content", "")
        if piece:
            full_text += piece

    n = len(full_text.split())  # rough token count from text
    # We'll return NaN for timing since bulk recv loses timing resolution
    return float("nan"), float("nan"), [], n, full_text


def stream_completion_timed(model, content, max_tokens=MAX_TOKENS):
    """
    Streaming SSE with per-line timing via raw socket recv loop.
    Returns (ttft_ms, tps, itl_list_ms, token_count, full_text).
    """
    import re

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }).encode("utf-8")

    http_req = (
        f"POST {PATH} HTTP/1.1\r\n"
        f"Host: {BASE_HOST}:{BASE_PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii") + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect((BASE_HOST, BASE_PORT))

    t_send = time.perf_counter()
    sock.sendall(http_req)

    ttft_ms = None
    token_times = []
    full_text = ""
    buf = b""
    past_headers = False

    done = False
    while not done:
        try:
            chunk = sock.recv(256)  # small recv = finer timing granularity
        except socket.timeout:
            break
        if not chunk:
            break

        buf += chunk

        if not past_headers:
            idx = buf.find(b"\r\n\r\n")
            if idx == -1:
                continue
            buf = buf[idx + 4:]
            past_headers = True

        # Process complete lines
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            # Skip chunked transfer size lines
            if re.fullmatch(r"[0-9a-fA-F]+", line):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t_send) * 1000.0
                token_times.append(now)
                full_text += piece

    sock.close()

    n = len(token_times)
    if n < 2:
        tps = float("nan")
        itl = []
    else:
        elapsed = token_times[-1] - token_times[0]
        tps = (n - 1) / elapsed if elapsed > 0 else float("nan")
        itl = [(token_times[i] - token_times[i-1]) * 1000.0 for i in range(1, n)]

    return ttft_ms if ttft_ms is not None else float("nan"), tps, itl, n, full_text


def run_bench():
    print("=" * 80)
    print("  EMBARK ENGINE -- LIVE SERVER BENCHMARK")
    print(f"  Endpoint : http://{BASE_HOST}:{BASE_PORT}{PATH}")
    print(f"  Rounds   : {ROUNDS} per (model x prompt)")
    print(f"  Max tok  : {MAX_TOKENS}")
    print("=" * 80)

    results = {}

    for model in MODELS:
        results[model] = {}
        print(f"\n{'-'*80}")
        print(f"  MODEL: {model}")
        print(f"{'-'*80}")

        for tag, prompt in PROMPTS:
            ttfts, tpss, all_itl, token_counts = [], [], [], []
            print(f"\n  Prompt [{tag}]: \"{prompt[:60]}{'...' if len(prompt)>60 else ''}\"")

            for r in range(1, ROUNDS + 1):
                try:
                    ttft, tps, itl, n, text = stream_completion_timed(model, prompt)
                    ttfts.append(ttft)
                    tpss.append(tps)
                    all_itl.extend(itl)
                    token_counts.append(n)
                    ttft_str = f"{ttft:7.1f}" if ttft == ttft else "    N/A"
                    tps_str  = f"{tps:6.2f}"  if tps == tps  else "   N/A"
                    preview  = text[:70].replace('\n', ' ')
                    print(f"    Round {r}: TTFT={ttft_str}ms  TPS={tps_str} tok/s  "
                          f"tokens={n:3d}  \"{preview}\"")
                except Exception as e:
                    print(f"    Round {r}: ERROR -- {e}")

            if ttfts:
                valid_tps = [x for x in tpss if x == x]
                results[model][tag] = {
                    "ttft_mean":  safe_mean(ttfts),
                    "ttft_std":   safe_std(ttfts),
                    "tps_mean":   safe_mean(tpss),
                    "tps_std":    safe_std(pss := [x for x in tpss if x == x]) if (pss := [x for x in tpss if x == x]) else 0,
                    "itl_p50":    pct(all_itl, 50),
                    "itl_p95":    pct(all_itl, 95),
                    "itl_p99":    pct(all_itl, 99),
                    "tokens_avg": safe_mean(token_counts),
                }

    # Summary table
    print(f"\n\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    hdr = f"{'Model':<14} {'Prompt':<8} {'TTFT ms':>8} {'TPS':>7} {'P50 ITL':>9} {'P95 ITL':>9} {'P99 ITL':>9} {'Tok':>5}"
    print(hdr)
    print("-" * len(hdr))

    for model in MODELS:
        for tag, _ in PROMPTS:
            d = results.get(model, {}).get(tag)
            if d:
                ttft_s = f"{d['ttft_mean']:8.1f}" if d['ttft_mean'] == d['ttft_mean'] else "     N/A"
                tps_s  = f"{d['tps_mean']:7.2f}"  if d['tps_mean']  == d['tps_mean']  else "    N/A"
                p50_s  = f"{d['itl_p50']:9.1f}"   if d['itl_p50']   == d['itl_p50']   else "      N/A"
                p95_s  = f"{d['itl_p95']:9.1f}"   if d['itl_p95']   == d['itl_p95']   else "      N/A"
                p99_s  = f"{d['itl_p99']:9.1f}"   if d['itl_p99']   == d['itl_p99']   else "      N/A"
                tok_s  = f"{d['tokens_avg']:5.0f}" if d['tokens_avg'] == d['tokens_avg'] else "  N/A"
                print(f"{model:<14} {tag:<8}{ttft_s}{tps_s}{p50_s}{p95_s}{p99_s}{tok_s}")
        print()

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_bench()
