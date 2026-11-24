"""
run_accuracy.py: accuracy check (H3). Grader holistic scores vs official
Smarter Balanced scores on 8 grade 11 explanatory responses (Spearman).
"""

import os
import random

import ai_professor as ap
from stats_utils import fmt, fmt_p, spearman

SEED = 2026


def main():
    rng = random.Random(SEED)
    passages = ap.load_passages()
    task = {"text": passages["smart_task"], "key_points": []}

    import pandas as pd
    rows = pd.read_csv(os.path.join(ap.MATERIALS, "smart_responses.csv")).to_dict("records")

    print(f"Grader: {ap.MODEL}   responses: {len(rows)}   runs each: {ap.N_RUNS}\n")

    order = rows[:]
    rng.shuffle(order)  # blinding: random scoring order
    for r in order:
        med, runs = ap.score_text(task, r["text"], "explanatory", n_runs=ap.N_RUNS)
        r["grader"] = med["holistic"]
        r["runs"] = [x["holistic"] for x in runs]

    rows.sort(key=lambda r: -r["official_score"])
    print(f"  {'id':<8}{'official':>9}{'grader':>8}   {'all runs':<16}{'words':>7}")
    for r in rows:
        print(f"  {r['response_id']:<8}{r['official_score']:>6}/10{r['grader']:>6}/5   "
              f"{str(r['runs']):<16}{len(str(r['text']).split()):>7}")

    sp = spearman([r["grader"] for r in rows], [r["official_score"] for r in rows])
    same = sum(len(set(r["runs"])) == 1 for r in rows) / len(rows)

    print(f"\n  Spearman rho = {fmt(sp['rho'])}, "
          f"p = {fmt_p(sp['p'])}")
    print(f"  Identical score on all {ap.N_RUNS} runs: {same:.0%} of responses")

    out = os.path.join(ap.HERE, "results", "accuracy_check.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "official_score", "grader_median", "runs"])
        for r in rows:
            w.writerow([r["response_id"], r["official_score"], r["grader"], r["runs"]])
    print("  Wrote results/accuracy_check.csv")


if __name__ == "__main__":
    main()
