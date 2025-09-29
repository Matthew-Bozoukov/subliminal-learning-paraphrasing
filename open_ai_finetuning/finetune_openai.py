#!/usr/bin/env python3
"""
Fine-tune an OpenAI model from a single training JSONL (no validation file)
with real-time polling + failure diagnostics.

Usage:
  python finetune.py \
    --train ./train_chat.jsonl \
    --model gpt-4o-mini \
    --suffix my-run \
    --n-epochs 3 \
    --poll \
    --metrics-csv metrics.csv \
    --tail-on-fail 200 \
    --dump-events fine_tune_events.jsonl

Env:
  export OPENAI_API_KEY=sk-...
"""

import argparse, os, sys, time, csv, shutil, json
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------- ASCII plotting helpers ---------------------- #

BARS = "▁▂▃▄▅▆▇█"

def _minmax(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (0.0, 1.0)
    mn = min(xs); mx = max(xs)
    if mx == mn:
        eps = 1e-9
        return (mn - eps, mx + eps)
    return (mn, mx)

def sparkline(xs, width=60):
    if not xs: return ("", 0.0, 1.0)
    n = len(xs)
    if n > width:
        xs = xs[-width:]
    mn, mx = _minmax(xs)
    span = mx - mn if mx > mn else 1.0
    out = []
    for x in xs:
        v = 0 if x is None else int(round((x - mn) / span * (len(BARS) - 1)))
        v = max(0, min(v, len(BARS) - 1))
        out.append(BARS[v])
    return ("".join(out), mn, mx)

def print_live(title: str, series: List[float], width: int = 60):
    s, mn, mx = sparkline(series, width=width)
    if s:
        print(f"{title:>12}  {s}  [{mn:.4g} … {mx:.4g}]")

# ---------------------- metrics parsing ---------------------- #

def extract_step_and_metrics(ev: Any) -> Optional[Dict[str, Any]]:
    data = getattr(ev, "data", None) or {}
    step = None
    for k in ("step", "global_step", "iteration", "epoch", "iter"):
        v = data.get(k)
        if isinstance(v, (int, float)):
            step = int(v); break
    metrics: Dict[str, float] = {}
    m = data.get("metrics")
    if isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
    for k, v in data.items():
        if k in ("step", "global_step", "iteration", "epoch", "iter"):
            continue
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    if not metrics:
        return None
    return {"step": step, "metrics": metrics}

def append_csv(path: str, header_keys: List[str], row: Dict[str, Any], wrote_header_once: List[bool]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if (not exists) and (not wrote_header_once[0]):
            w.writerow(["created_at", "step"] + header_keys)
            wrote_header_once[0] = True
        w.writerow([row.get("created_at", ""), row.get("step", "")] + [row.get(k, "") for k in header_keys])

# ---------------------- failure triage helpers ---------------------- #

def summarize_failure(job, events_tail: List[Any], train_file_meta) -> None:
    """Prints the likely reason and concrete next steps."""
    print("\n========== FAILURE DIAGNOSTICS ==========")

    # 1) Primary error object if present
    err = getattr(job, "error", None)
    if err:
        # err can be dict-like with 'message' and 'code'
        code = getattr(err, "code", None) or (err.get("code") if isinstance(err, dict) else None)
        msg  = getattr(err, "message", None) or (err.get("message") if isinstance(err, dict) else str(err))
        print(f"job.error.code:    {code}")
        print(f"job.error.message: {msg}")

    # 2) Training file ingestion/validation
    if train_file_meta:
        print(f"\nTraining file status: {train_file_meta.status} | bytes={train_file_meta.bytes} | name={train_file_meta.filename}")
        if hasattr(train_file_meta, "purpose"):
            print(f"purpose: {getattr(train_file_meta,'purpose',None)}")
        # Sometimes validator messages appear in events only, but if status != 'processed' it's a strong hint
        if getattr(train_file_meta, "status", "") not in ("processed", "uploaded"):
            print("Note: training file not fully processed. There may be schema or encoding issues.")

    # 3) Tail of WARN/ERROR events
    if events_tail:
        print("\nRecent WARN/ERROR events:")
        for e in events_tail:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.created_at))
            level = getattr(e, "level", "info").upper()
            msg = getattr(e, "message", "")
            print(f"  [{ts}] {level:7} {msg}")

    # 4) Heuristic hints based on messages
    joined = " ".join(getattr(e, "message", "") for e in events_tail).lower()
    hints = []

    if "invalid json" in joined or "invalid line" in joined or "could not parse json" in joined:
        hints += [
            "- Your training file has a malformed JSON line. Check the line number in the event message.",
            "- Each line must be a single JSON object like: {\"messages\":[{\"role\":\"user\",\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"...\"}]}"
        ]
    if "messages" in joined and "role" in joined and "content" in joined and "missing" in joined:
        hints += [
            "- Some examples are missing required fields. Every example needs a 'messages' array with 'user' then 'assistant', each having 'content'."
        ]
    if "too long" in joined or "max tokens" in joined or "exceeds" in joined:
        hints += [
            "- One or more examples are too long. Trim extremely long prompts/answers.",
        ]
    if "rate limit" in joined or "quota" in joined or "billing" in joined:
        hints += [
            "- The run may have hit rate limits or quota. Re-run later or reduce parallelism; check the account's usage limits."
        ]
    if "file not found" in joined or "training_file" in joined and "not found" in joined:
        hints += [
            "- The training_file reference became unavailable. Re-upload the file and retry."
        ]

    if hints:
        print("\nLikely next steps:")
        for h in hints:
            print(h)

    print("========================================\n")

# ---------------------- main ---------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to training .jsonl")
    ap.add_argument("--model", default="gpt-4.1-nano-2025-04-14", help="Base model")
    ap.add_argument("--suffix", default="", help="Model suffix (optional)")
    ap.add_argument("--n-epochs", type=int, default=10, help="n_epochs (optional)")
    ap.add_argument("--poll", action="store_true", help="Poll until job completes")
    ap.add_argument("--seed", default='42', help="seed for finetuning")
    ap.add_argument("--lr", default='2', help="learning rate multiplier")
    ap.add_argument("--batch_size", default='4', help="batch size")
    # metrics
    ap.add_argument("--metrics-csv", default=None, help="Optional CSV to log metrics as they stream")
    ap.add_argument("--plot-width", type=int, default=70, help="ASCII plot width")
    # failure diagnostics
    ap.add_argument("--tail-on-fail", type=int, default=200, help="Show last N WARN/ERROR events when a job fails")
    ap.add_argument("--dump-events", default=None, help="If set, append all polled events to this JSONL")
    args = ap.parse_args()

    if not os.path.exists(args.train) or not args.train.endswith(".jsonl"):
        sys.exit("Training file must be an existing .jsonl")

    client = OpenAI(api_key='')  # uses OPENAI_API_KEY

    # 1) Upload training file
    with open(args.train, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print("Uploaded training file:", train_file.id)

    # 2) Create job
    params = {
        "model": args.model,
        "training_file": train_file.id,
    }
    if args.suffix:
        params["suffix"] = args.suffix
    if args.n_epochs is not None:
        params["hyperparameters"] = {
            "n_epochs": args.n_epochs,
            "learning_rate_multiplier": args.lr,
            "batch_size": args.batch_size
        }
    job = client.fine_tuning.jobs.create(**params, seed=args.seed)
    print(f"Created job: {job.id} | status={job.status}")
    print("job.status =", job.status)
    print("training_file =", job.training_file, "validation_file =", getattr(job, "validation_file", None))

    # initial validator messages
    ev = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=100)
    for e in reversed(ev.data):
        print(e.level, e.message)

    tf_meta = client.files.retrieve(job.training_file)
    print("file.status =", tf_meta.status, " | purpose =", tf_meta.purpose, " | filename =", tf_meta.filename, " | bytes =", tf_meta.bytes)

    # Prepare event dump / rolling tail caches
    last_event = None
    dump_fp = open(args.dump_events, "a", encoding="utf-8") if args.dump_events else None
    tail_errs: deque = deque(maxlen=args.tail_on_fail or 200)

    # 3) Poll
    if args.poll:
        print("\nPolling… (Ctrl+C to stop)\n")

        metric_series: Dict[str, List[float]] = defaultdict(list)
        metric_header: List[str] = []
        wrote_header_once = [False]

        term_width = shutil.get_terminal_size((100, 20)).columns
        plot_width = max(30, min(args.plot_width, term_width - 20))

        try:
            while True:
                job = client.fine_tuning.jobs.retrieve(job.id)
                events = client.fine_tuning.jobs.list_events(
                    fine_tuning_job_id=job.id,
                    after=last_event
                )

                # chronological
                for e in reversed(events.data):
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.created_at))
                    last_event = e.id
                    msg = e.message
                    level = e.level.upper()

                    # dump raw event if requested
                    if dump_fp:
                        dump_fp.write(json.dumps({
                            "id": e.id,
                            "created_at": e.created_at,
                            "level": e.level,
                            "message": e.message,
                            "data": getattr(e, "data", None)
                        }, ensure_ascii=False) + "\n")

                    # print live stream
                    print(f"[{ts}] {level:7} {msg}")

                    # keep a tail of warnings/errors in memory
                    if level in ("WARN", "ERROR"):
                        tail_errs.append(e)

                    # parse metrics if any
                    parsed = extract_step_and_metrics(e)
                    if parsed:
                        step = parsed["step"]
                        numbers = parsed["metrics"]
                        for k in numbers.keys():
                            if k not in metric_header:
                                metric_header.append(k)
                        for k in metric_header:
                            v = numbers.get(k)
                            if v is not None:
                                metric_series[k].append(float(v))
                            else:
                                if len(metric_series[k]) < len(metric_series[metric_header[0]]):
                                    metric_series[k].append(None)
                        if args.metrics_csv:
                            row = {"created_at": ts, "step": step or ""}
                            for k in metric_header:
                                val_list = metric_series[k]
                                row[k] = val_list[-1] if val_list else ""
                            append_csv(args.metrics_csv, metric_header, row, wrote_header_once)
                        keys_to_plot = [k for k in metric_header if "loss" in k.lower()] or (metric_header[:1] if metric_header else [])
                        for k in keys_to_plot:
                            series = [x for x in metric_series[k] if x is not None]
                            print_live(k, series, width=plot_width)

                if job.status in ("succeeded", "failed", "cancelled"):
                    print("\nJob finished:", job.status)
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped polling. You can check later with the job id:", job.id)
        finally:
            if dump_fp:
                dump_fp.flush()
                dump_fp.close()

    # 4) End-of-run report
    job = client.fine_tuning.jobs.retrieve(job.id)

    if getattr(job, "fine_tuned_model", None):
        print("\nFine-tuned model:", job.fine_tuned_model)
        print("\nQuick test:\n")
        print(f"""from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(
    model="{job.fine_tuned_model}",
    messages=[{{"role":"user","content":"Hello!"}}]
)
print(resp.choices[0].message["content"])""")
    else:
        print("No model yet. Current status:", job.status)
        if job.status == "failed":
            # fetch a fresh tail of events (WARN/ERROR only)
            ev_all = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=1000)
            warn_err_tail: List[Any] = []
            for e in ev_all.data[::-1]:  # chronological
                if e.level.upper() in ("WARN", "ERROR"):
                    warn_err_tail.append(e)
                if len(warn_err_tail) >= (args.tail_on_fail or 200):
                    break
            summarize_failure(job, warn_err_tail, tf_meta)

if __name__ == "__main__":
    main()
