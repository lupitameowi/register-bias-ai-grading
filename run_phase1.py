"""
run_phase1.py: Phase 1, grader validation and controlled register test (H1-H4).
No human participants.

Steps
  0. Materials check   every matched pair: same key points, lengths within 10%,
                        and a real register difference (objective formality).
  1. Stability (H1)     each text scored 5x with prompt A, 1x with reworded B and C.
                        Sets the smallest score difference treated as meaningful.
  2. Order effect (H2)  order-test texts scored after strong summaries, after weak
                        summaries, and alone in a fresh session.
  3. Accuracy (H3)      grader vs official scores on SmART responses (Spearman).
  4. Register test (H4) casual vs formal versions of each matched pair.
"""

import os
import random
import statistics

import pandas as pd

import ai_professor as ap
from stats_utils import friedman, fmt, fmt_p, paired_summary, qwk, spearman

SEED = 2026
ORDER_PASSAGE = "caves"
RESULTS = os.path.join(ap.HERE, "results")


def read(name):
    return pd.read_csv(os.path.join(ap.MATERIALS, name)).to_dict("records")


def banner(title):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def main():
    rng = random.Random(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    passages = ap.load_passages()
    pairs = read("matched_pairs.csv")
    neighbors = read("order_neighbors.csv")
    smart = read("smart_responses.csv")

    print(f"Grader: {ap.MODEL}   runs per text: {ap.N_RUNS}   seed: {SEED}")

    # ------------------------------------------------------------------ 0
    banner("0. MATERIALS CHECK (matched pairs)")
    texts, problems = [], 0
    print(f"  {'pair':<6}{'key pts (c/f)':>15}{'words (c/f)':>14}{'FK grade (c/f)':>17}")
    for p in pairs:
        passage = passages[p["passage_id"]]
        row = {}
        for reg in ("casual", "formal"):
            t = p[f"{reg}_text"]
            m = ap.text_measures(t)
            row[reg] = {"cov": ap.key_point_coverage(t, passage), **m}
            texts.append({"text_id": f"{p['pair_id']}-{reg}", "pair_id": p["pair_id"],
                          "passage_id": p["passage_id"], "register": reg, "text": t, **m})
        c, f = row["casual"], row["formal"]
        ratio = abs(c["word_count"] - f["word_count"]) / max(c["word_count"], f["word_count"])
        flags = []
        if c["cov"] != f["cov"]:
            flags.append("KEY POINTS DIFFER")
        if ratio > 0.10:
            flags.append(f"LENGTH OFF BY {ratio:.0%}")
        problems += bool(flags)
        print(f"  {p['pair_id']:<6}{c['cov']:>8}/{f['cov']:<6}{c['word_count']:>8}/{f['word_count']:<5}"
              f"{c['fk_grade']:>10.1f}/{f['fk_grade']:<6.1f} {'  '.join(flags)}")
    fk = paired_summary([t["fk_grade"] for t in texts if t["register"] == "casual"],
                        [t["fk_grade"] for t in texts if t["register"] == "formal"])
    inf = paired_summary([t["informal_per_100"] for t in texts if t["register"] == "casual"],
                         [t["informal_per_100"] for t in texts if t["register"] == "formal"])
    print(f"\n  Register really differs?  FK grade casual-formal = {fmt(fk['mean_diff'])} (p {fmt_p(fk['p_t'])});"
          f"  informal markers/100 words = {fmt(inf['mean_diff'])} (p {fmt_p(inf['p_t'])})")
    print(f"  Pairs flagged: {problems}")

    # ------------------------------------------------------------------ 1
    banner("1. STABILITY (H1) — how much does the grader disagree with itself?")
    order = texts[:]
    rng.shuffle(order)  # blinding: random scoring order
    for t in order:
        passage = passages[t["passage_id"]]
        t["A_med"], t["A_runs"] = ap.score_text(passage, t["text"], "analytic", "A")
        t["B"] = ap.score_text(passage, t["text"], "analytic", "B", n_runs=1)[0]
        t["C"] = ap.score_text(passage, t["text"], "analytic", "C", n_runs=1)[0]
        t["H_med"], t["H_runs"] = ap.score_text(passage, t["text"], "holistic")

    print(f"  {'dimension':<20}{'identical 5/5 runs':>20}{'mean run range':>16}{'QWK A vs B':>12}{'QWK A vs C':>12}")
    for dim in ap.DIMENSIONS:
        same = statistics.mean(len({r[dim] for r in t["A_runs"]}) == 1 for t in texts)
        rng_ = statistics.mean(max(r[dim] for r in t["A_runs"]) - min(r[dim] for r in t["A_runs"]) for t in texts)
        a = [round(t["A_med"][dim]) for t in texts]
        print(f"  {dim:<20}{same:>19.0%}{rng_:>16.2f}{fmt(qwk(a, [t['B'][dim] for t in texts])):>12}"
              f"{fmt(qwk(a, [t['C'][dim] for t in texts])):>12}")
    same_h = statistics.mean(len({r['holistic'] for r in t["H_runs"]}) == 1 for t in texts)
    print(f"  {'holistic':<20}{same_h:>19.0%}")

    within = statistics.mean(statistics.median(abs(r["comprehension"] - t["A_med"]["comprehension"]) for r in t["A_runs"])
                             for t in texts)
    across = statistics.mean((abs(t["B"]["comprehension"] - t["A_med"]["comprehension"]) +
                              abs(t["C"]["comprehension"] - t["A_med"]["comprehension"])) / 2 for t in texts)
    floor = max(within, across)
    print(f"\n  Typical comprehension wobble: within-prompt {within:.2f}, across reworded prompts {across:.2f}.")
    print(f"  STABILITY THRESHOLD = {floor:.2f} points. Smaller differences are not treated as meaningful.")

    # ------------------------------------------------------------------ 2
    banner(f"2. ORDER EFFECT (H2) — passage '{ORDER_PASSAGE}'")
    targets = [t for t in texts if t["passage_id"] == ORDER_PASSAGE]
    strong = [n["text"] for n in neighbors if n["passage_id"] == ORDER_PASSAGE and n["quality"] == "strong"]
    weak = [n["text"] for n in neighbors if n["passage_id"] == ORDER_PASSAGE and n["quality"] == "weak"]
    order_rows = []
    for t in targets:
        p = passages[ORDER_PASSAGE]
        after_strong = ap.score_text(p, t["text"], "analytic", "A", context=strong)[0]["comprehension"]
        after_weak = ap.score_text(p, t["text"], "analytic", "A", context=weak)[0]["comprehension"]
        order_rows.append({"text_id": t["text_id"], "register": t["register"], "after_strong": after_strong,
                           "after_weak": after_weak, "alone": t["A_med"]["comprehension"]})
    od = pd.DataFrame(order_rows)
    fr = friedman(od["after_strong"], od["after_weak"], od["alone"])
    print(f"  Mean comprehension score:  after strong {od['after_strong'].mean():.2f} | "
          f"after weak {od['after_weak'].mean():.2f} | alone {od['alone'].mean():.2f}")
    print(f"  Friedman chi2 = {fmt(fr['chi2'])}, p = {fmt_p(fr['p'])}   (n = {len(od)} texts)")
    od["shift"] = od["after_weak"] - od["after_strong"]
    for reg in ("casual", "formal"):
        print(f"  Exploratory: weak-minus-strong shift for {reg} texts = {od.loc[od.register == reg, 'shift'].mean():.2f}")
    od.to_csv(os.path.join(RESULTS, "phase1_order_test.csv"), index=False)

    # ------------------------------------------------------------------ 3
    banner("3. ACCURACY (H3) — grader vs official human scores")
    task = {"text": passages["smart_task"], "key_points": []}
    grader = [ap.score_text(task, s["text"], "explanatory")[0]["holistic"] for s in smart]
    sp = spearman(grader, [s["official_score"] for s in smart])
    for s, g in zip(smart, grader):
        print(f"  {s['response_id']}: official {s['official_score']:>2}/10   grader {g}/5")
    print(f"  Spearman rho = {fmt(sp['rho'])}, p = {fmt_p(sp['p'])}   "
          f"(SmART grade 11 responses, official scores 2-10)")

    # ------------------------------------------------------------------ 4
    banner("4. CONTROLLED REGISTER TEST (H4) — same content, different register")
    by_pair = {}
    for t in texts:
        by_pair.setdefault(t["pair_id"], {})[t["register"]] = t
    pids = sorted(by_pair)
    outcomes = [("comprehension", "content: TEST OF BIAS"), ("organization", "content: test of bias"),
                ("holistic", "holistic: TEST OF BIAS"), ("grammar_mechanics", "style: manipulation check"),
                ("word_choice", "style: manipulation check"), ("academic_tone", "style: manipulation check")]
    print(f"  {'outcome':<19}{'role':<27}{'casual':>7}{'formal':>7}{'diff':>7}{'95% CI':>16}{'p(t)':>7}{'p(W)':>7}{'dz':>7}")
    for dim, role in outcomes:
        get = (lambda t: t["H_med"]["holistic"]) if dim == "holistic" else (lambda t, d=dim: t["A_med"][d])
        s = paired_summary([get(by_pair[p]["casual"]) for p in pids], [get(by_pair[p]["formal"]) for p in pids])
        flag = ""
        if "bias" in role.lower():
            flag = "  <- beyond threshold" if abs(s["mean_diff"]) > floor else "  (within threshold)"
        print(f"  {dim:<19}{role:<27}{s['mean_casual']:>7.2f}{s['mean_formal']:>7.2f}{s['mean_diff']:>7.2f}"
              f"   [{fmt(s['ci_low'])},{fmt(s['ci_high'])}]{fmt_p(s['p_t']):>7}{fmt_p(s['p_wilcoxon']):>7}{fmt(s['dz']):>7}{flag}")
    print(f"\n  n = {len(pids)} pairs.")

    flat = [{"text_id": t["text_id"], "pair_id": t["pair_id"], "passage_id": t["passage_id"],
             "register": t["register"], "word_count": t["word_count"], "fk_grade": t["fk_grade"],
             "informal_per_100": t["informal_per_100"],
             **{f"ai_{d}": t["A_med"][d] for d in ap.DIMENSIONS}, "ai_holistic": t["H_med"]["holistic"]}
            for t in texts]
    pd.DataFrame(flat).to_csv(os.path.join(RESULTS, "phase1_scores.csv"), index=False)
    print(f"\n  Wrote results/phase1_scores.csv and results/phase1_order_test.csv")


if __name__ == "__main__":
    main()
