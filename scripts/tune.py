#!/usr/bin/env python3
"""Compare server configurations on the same workload, without touching the device.

For each configuration: reload the model through the admin API, wait for the device to
cool to `nominal`, run the workload a number of times, and record what each response
reports. Results go to a CSV, one row per request, and a summary table is printed with
medians per configuration.

    python scripts/tune.py --base http://192.168.1.50:8080 --image menu.jpg
    python scripts/tune.py --base ... --configs my_configs.json --repeats 3 --out run.csv

Only the standard library is used, so it runs wherever Python 3.8+ does.

Why it waits for the device to cool: a fanless iPad runs measurably slower once warm, and
comparing a configuration measured cold with one measured hot measures the temperature,
not the configuration. Why it repeats: one run tells nothing about the spread, and without
the spread a difference between configurations cannot be told from noise.
"""
import argparse, base64, csv, json, statistics, sys, time, urllib.error, urllib.request

# The reference configuration first, then one change at a time.
DEFAULT_CONFIGS = [
    {"label": "reference",         "context_length": 32768, "batch_size": 512,  "reuse_kv_cache": False, "kv_cache_type": "f16"},
    {"label": "batch 1024",        "context_length": 32768, "batch_size": 1024, "reuse_kv_cache": False, "kv_cache_type": "f16"},
    {"label": "batch 2048",        "context_length": 32768, "batch_size": 2048, "reuse_kv_cache": False, "kv_cache_type": "f16"},
    {"label": "reuse, batch 2048", "context_length": 32768, "batch_size": 2048, "reuse_kv_cache": True,  "kv_cache_type": "f16"},
    {"label": "kv q8_0",           "context_length": 32768, "batch_size": 512,  "reuse_kv_cache": False, "kv_cache_type": "q8_0"},
]


def http(base, method, path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer tune"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def wait_until_cool(base, limit_s):
    start = time.time()
    while time.time() - start < limit_s:
        _, stats = http(base, "GET", "/v1/stats", timeout=15)
        if stats.get("thermal_state") == "nominal":
            return "nominal"
        print(f"    cooling: thermal={stats.get('thermal_state')}, waiting...", flush=True)
        time.sleep(30)
    return "not cooled within limit"


def build_workload(args):
    """A long shared system prompt with different questions, as a multi-call pipeline stage
    sends, plus an image transcription if an image is given."""
    system = " ".join(f"Rule {i}: be precise, quote names and prices exactly as given, never invent items."
                      for i in range(1, args.system_rules + 1))
    workload = []
    for i, question in enumerate(["List three spirit categories.", "List three wine styles.",
                                  "List three beer styles."], 1):
        workload.append(("text %d" % i, {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "max_completion_tokens": args.max_tokens}))
    if args.image:
        image = base64.b64encode(open(args.image, "rb").read()).decode()
        mime = "image/png" if args.image.lower().endswith(".png") else "image/jpeg"
        workload.append(("image", {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
                {"type": "text", "text": "Transcribe every item on this menu with its price."}]}],
            "max_completion_tokens": args.max_tokens}))
    return workload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="server root, e.g. http://192.168.1.50:8080")
    parser.add_argument("--model", default="gemma-3-4b-it-Q4_K_M")
    parser.add_argument("--configs", help="JSON file: a list of objects with admin load settings and a 'label'")
    parser.add_argument("--image", help="a real menu photo to include as an image request")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--system-rules", type=int, default=250, help="length of the shared system prompt")
    parser.add_argument("--cool-limit", type=int, default=900, help="seconds to wait for nominal thermal state")
    parser.add_argument("--out", default="tune_results.csv")
    args = parser.parse_args()

    configs = json.load(open(args.configs)) if args.configs else DEFAULT_CONFIGS
    workload = build_workload(args)
    rows = []

    for config in configs:
        label = config.get("label", json.dumps(config))
        settings = {k: v for k, v in config.items() if k != "label"}
        print(f"\n=== {label} ===", flush=True)
        status, loaded = http(args.base, "POST", "/v1/admin/load",
                              {"model": args.model, "projector": bool(args.image), **settings})
        if status != 200:
            print(f"  load failed ({status}): {loaded.get('error')}")
            rows.append({"config": label, "request": "(load)", "error": loaded.get("error")})
            continue
        print(f"  loaded: {loaded.get('configuration')}")

        for repeat in range(1, args.repeats + 1):
            thermal_before = wait_until_cool(args.base, args.cool_limit)
            for name, body in workload:
                started = time.time()
                status, response = http(args.base, "POST", "/v1/chat/completions",
                                        {"model": "tune", "temperature": 0, "stream": False, **body})
                wall = time.time() - started
                _, stats = http(args.base, "GET", "/v1/stats", timeout=15)
                usage, timings = response.get("usage", {}), response.get("timings", {})
                row = {
                    "config": label, "repeat": repeat, "request": name, "status": status,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": timings.get("cached_tokens"),
                    "prefill_ms": timings.get("prefill_ms"), "decode_ms": timings.get("decode_ms"),
                    "prefill_tps": round(timings.get("prefill_tps", 0), 1),
                    "decode_tps": round(timings.get("decode_tps", 0), 1),
                    "wall_s": round(wall, 1),
                    "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                    "peak_footprint_mb": round(stats.get("peak_footprint_bytes", 0) / 2**20),
                    "thermal_before": thermal_before, "thermal_after": stats.get("thermal_state"),
                    "server_config": timings.get("config"),
                }
                rows.append(row)
                print(f"  #{repeat} {name:<7} prefill {row['prefill_ms']}ms ({row['prefill_tps']} tok/s, "
                      f"{row['cached_tokens']} cached)  decode {row['decode_tps']} tok/s  "
                      f"{row['wall_s']}s  {row['thermal_after']}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary (medians), all rows in {args.out}\n")
    print(f"{'config':<20} {'request':<8} {'prefill tok/s':>13} {'decode tok/s':>12} {'wall s':>7} {'cached':>7}")
    for config in dict.fromkeys(row["config"] for row in rows):
        for name in dict.fromkeys(row["request"] for row in rows if row["config"] == config):
            sample = [r for r in rows if r["config"] == config and r["request"] == name and r.get("status") == 200]
            if not sample:
                continue
            median = lambda key: statistics.median(r[key] or 0 for r in sample)
            print(f"{config:<20} {name:<8} {median('prefill_tps'):>13.1f} {median('decode_tps'):>12.1f} "
                  f"{median('wall_s'):>7.1f} {median('cached_tokens'):>7.0f}")

    # Leave the device as it was found: on the reference configuration.
    http(args.base, "POST", "/v1/admin/load",
         {"model": args.model, "projector": True, **{k: v for k, v in DEFAULT_CONFIGS[0].items() if k != "label"}})
    print("\nDevice restored to the reference configuration.")


if __name__ == "__main__":
    main()
