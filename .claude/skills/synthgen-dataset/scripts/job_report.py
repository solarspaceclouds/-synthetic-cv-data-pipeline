"""Report on a synthgen job's output directory: what was planned, generated, accepted and rejected.

Prints the config essentials, counts, rejection reasons, per-task table (target, prompt,
reference, verdict, scores, occluder share, degradation), timings, and the files to look at.

    .venv/bin/python .claude/skills/synthgen-dataset/scripts/job_report.py out/<job>
    ... --tasks 0    # summary only
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", type=int, default=50, help="max per-task rows (0 = none)")
    a = ap.parse_args()
    out = Path(a.out)
    if not (out / "job.json").exists():
        raise SystemExit(f"{out}/job.json not found: not a synthgen output dir (or not planned yet)")

    cfg = json.loads((out / "job.json").read_text())["config"]
    tasks = jsonl(out / "tasks.jsonl")
    gens = {g["task_id"]: g for g in jsonl(out / "generation.jsonl")}
    verdicts = {v["task_id"]: v for v in jsonl(out / "verdicts.jsonl")}

    g = cfg["generator"]
    print(f"job {cfg['name']}  ({cfg['domain']}, {cfg['operation']})")
    print(f"  source    {cfg['source'].get('annotations') or ''} {cfg['source']['images']}  split={cfg['source']['split']}")
    print(f"  targets   {cfg['targets']['classes']}  change_to={cfg.get('change_to') or '-'}")
    print(f"  generator {g['adapter']} res={g.get('res')} steps={g.get('steps')}  labeler={cfg['labeler']['adapter']}")
    cond = cfg["conditioning"]
    print(f"  condition {cond['type']}" + (f"  match={cond['reference']['match']}" if cond["type"] == "reference_object" else
                                          f"  prompt='{cond['prompt']}'" if cond["type"] == "text" else ""))
    extras = [k for k in ("occlusion", "postprocess", "placement") if cfg.get(k) and
              (cfg[k].get("enabled") or cfg[k].get("ops") or cfg[k].get("ground_phrases"))]
    print(f"  extras    {', '.join(extras) or '-'}")

    print(f"\nplanned {len(tasks)} / requested {cfg['n_samples']}; generated {len(gens)}; verified {len(verdicts)}")
    acc = sum(v["accepted"] for v in verdicts.values())
    if verdicts:
        print(f"accepted {acc}/{len(verdicts)} ({acc / len(verdicts):.0%})")
        reasons = Counter(v["reason"] for v in verdicts.values() if not v["accepted"])
        for r, n in reasons.most_common():
            print(f"  rejected {n:3d}  {r}")
    secs = [g["seconds"] for g in gens.values() if "seconds" in g]
    if secs:
        print(f"generation {statistics.median(secs):.2f} s/image median, {sum(secs) / 60:.1f} min total")
    occ = [t.get("occluder_frac", 0) for t in tasks if "occluder_frac" in t]
    if occ:
        print(f"occlusion: occluders restored in {sum(o > 0 for o in occ)}/{len(occ)} tasks")
    if tasks and tasks[0].get("reference"):
        refs = Counter((t["reference"]["image_id"], t["reference"].get("ann_id")) for t in tasks if t.get("reference"))
        print(f"references: {len(refs)} distinct for {len(tasks)} tasks (max reuse {max(refs.values())})")

    if a.tasks and tasks:
        print(f"\n{'task':>5s}  {'image':>8s}  {'edit':18s} {'verdict':8s} {'score':>5s} {'inside':>6s} {'fill':>5s} {'occ':>5s}  prompt / reason")
        for t in tasks[: a.tasks]:
            v = verdicts.get(t["task_id"], {})
            edit = f"+{t['target_class']}" if t.get("insert_box") else f"{t['original_class']}->{t['target_class']}"
            status = "ACCEPT" if v.get("accepted") else ("reject" if v else "-")
            fill = v.get("fill_frac", v.get("area_ratio"))
            note = t.get("prompt") or (f"ref {t['reference']}" if t.get("reference") else "")
            if v and not v["accepted"]:
                note = v["reason"]
            print(f"{t['task_id'].rsplit('-', 1)[-1]:>5s}  {t['image_id']:>8}  {edit:18s} {status:8s} "
                  f"{v.get('score') if v.get('score') is not None else '':>5} {v.get('inside_frac', ''):>6} "
                  f"{fill if fill is not None else '':>5} {t.get('occluder_frac', ''):>5}  {note[:70]}")

    print("\nlook at:")
    for f in ("review.jpg", "summary.json", "annotations.json", "face_labels.json", "manifest.jsonl"):
        if (out / f).exists():
            print(f"  {out / f}")
    if not (out / "review.jpg").exists() and verdicts:
        print(f"  (no review sheet yet: .venv/bin/python -m synthgen.review {out} --max 12)")


if __name__ == "__main__":
    main()
