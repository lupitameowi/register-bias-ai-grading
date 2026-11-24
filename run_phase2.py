"""
run_phase2.py: Phase 2, first-generation student summaries (H5-H7).

Input: materials/phase2_summaries.csv (not included; see README), one row per summary:
    participant_id, passage_id, register (casual|formal), order (1|2), text,
    checklist_count (hand-scored key points), quiz_score
"""

import os
import random
import warnings

import pandas as pd
from statsmodels.stats.power import TTestPower

import ai_professor as ap
from stats_utils import fmt, fmt_p, paired_summary

SEED = 2026
RESULTS = os.path.join(ap.HERE, "results")
REAL_FILE = os.path.join(ap.MATERIALS, "phase2_summaries.csv")


def banner(title):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def mixed_model(df, outcome, add_word_count, add_order=False):
    import statsmodels.formula.api as smf
    formula = f"{outcome} ~ C(register, Treatment('formal')) + checklist_count + C(passage_id)"
    if add_word_count:
        formula += " + word_count"
    if add_order:
        formula += " + C(order)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm(formula, df, groups=df["participant_id"]).fit(reml=True)
        except Exception as exc:  # small pilot samples can fail to converge
            return {"error": str(exc)}
    term = next(k for k in res.params.index if "casual" in k)
    lo, hi = res.conf_int().loc[term]
    return {"estimate": float(res.params[term]), "ci_low": float(lo), "ci_high": float(hi), "p": float(res.pvalues[term])}


def main():
    rng = random.Random(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    passages = ap.load_passages()
    df = pd.read_csv(REAL_FILE)
    print(f"Grader: {ap.MODEL}   participants: {df.participant_id.nunique()}")

    # Score every summary blind, in random order, each in a fresh session.
    records = df.to_dict("records")
    rng.shuffle(records)
    for r in records:
        passage = passages[r["passage_id"]]
        analytic, _ = ap.score_text(passage, r["text"], "analytic", "A")
        holistic, _ = ap.score_text(passage, r["text"], "holistic")
        r.update({f"ai_{k}": v for k, v in analytic.items()}, ai_holistic=holistic["holistic"], **ap.text_measures(r["text"]))
    df = pd.DataFrame(records).sort_values(["participant_id", "register"]).reset_index(drop=True)
    df.drop(columns=["text"]).to_csv(os.path.join(RESULTS, "phase2_scores.csv"), index=False)

    wide = df.pivot(index="participant_id", columns="register")

    def paired(col):
        return paired_summary(wide[(col, "casual")], wide[(col, "formal")])

    def show(label, col, note=""):
        s = paired(col)
        print(f"  {label:<26}{s['mean_casual']:>8.2f}{s['mean_formal']:>8.2f}{s['mean_diff']:>8.2f}"
              f"   [{fmt(s['ci_low'])},{fmt(s['ci_high'])}]{fmt_p(s['p_t']):>8}{fmt(s['dz']):>7}  {note}")

    header = f"  {'measure':<26}{'casual':>8}{'formal':>8}{'diff':>8}{'95% CI':>16}{'p':>8}{'dz':>7}"

    banner("H5. MANIPULATION CHECK — did students actually write casually?")
    print(header)
    show("Flesch-Kincaid grade", "fk_grade")
    show("Informal markers /100 wds", "informal_per_100")
    show("AI academic tone", "ai_academic_tone")
    show("AI word choice", "ai_word_choice")
    show("AI grammar/mechanics", "ai_grammar_mechanics")

    banner("H6. CONTENT CHECK — same ideas in both summaries?")
    print(header)
    show("Key points (checklist)", "checklist_count")
    show("Quiz score", "quiz_score")
    show("Word count", "word_count")
    wc = paired("word_count")
    add_wc = abs(wc["mean_diff"]) > 0.10 * df["word_count"].mean()
    print(f"  Word count {'DOES' if add_wc else 'does not'} differ by >10%, so it "
          f"{'is' if add_wc else 'is not'} added as a covariate below.")

    firsts = df[df["order"] == 1]["register"].value_counts().to_dict()
    n_cas, n_for = firsts.get("casual", 0), firsts.get("formal", 0)
    print(f"\n  Order balance: {n_cas} wrote casual first, {n_for} wrote formal first.")

    banner("H7. MAIN TEST — does casual writing lose comprehension credit?")
    print("  Mixed model: score ~ register + key points + passage (+ word count), random intercept per student")
    print(f"  {'outcome':<20}{'model':<14}{'casual effect':>15}{'95% CI':>18}{'p':>8}")
    for outcome in ("ai_comprehension", "ai_organization", "ai_holistic"):
        for label, with_order in (("main", False), ("+ order", True)):
            m = mixed_model(df, outcome, add_wc, add_order=with_order)
            name = outcome if label == "main" else ""
            if "error" in m:
                print(f"  {name:<20}{label:<14}  did not fit: {m['error'][:44]}")
            else:
                print(f"  {name:<20}{label:<14}{m['estimate']:>15.2f}"
                      f"   [{fmt(m['ci_low'])},{fmt(m['ci_high'])}]{fmt_p(m['p']):>8}")
    print("\n  Simpler paired check:")
    print(header)
    show("AI comprehension", "ai_comprehension")
    show("AI holistic", "ai_holistic")

    same = wide[wide[("checklist_count", "casual")] == wide[("checklist_count", "formal")]]
    print(f"\n  Conditional comparison (Chen et al., 2026): students whose two summaries had the SAME key-point count")
    if len(same):
        print(f"  n = {len(same)}   comprehension: casual {same[('ai_comprehension', 'casual')].mean():.2f} vs "
              f"formal {same[('ai_comprehension', 'formal')].mean():.2f}   "
              f"holistic: casual {same[('ai_holistic', 'casual')].mean():.2f} vs formal {same[('ai_holistic', 'formal')].mean():.2f}")
    else:
        print("  n = 0")

    n = df.participant_id.nunique()
    dz_min = TTestPower().solve_power(nobs=n, alpha=0.05, power=0.80, alternative="two-sided")
    print(f"\n  Sensitivity: n = {n}, minimum detectable dz = {dz_min:.2f} (alpha .05, power .80, two-sided)")
    print("  Wrote results/phase2_scores.csv")


if __name__ == "__main__":
    main()
