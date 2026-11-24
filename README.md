# Register Bias in AI Grading

Code and materials for a two-phase study testing whether an AI grader mistakes casual, code-switched writing for weaker understanding.

Guadalupe Rodriguez · PSY 9: Research Methods in Psychology · Hartnell College · Faculty mentor: Prof. Kristi Sadler

## Design

**Phase 1: grader validation (no participants)**
- H1 Stability: each matched-pair text scored 5 times with the main prompt and once with each of two reworded prompts.
- H2 Order: 20 caves texts (10 casual, 10 formal) scored after 3 strong summaries, after 3 weak summaries, and alone.
- H3 Accuracy: grader vs. official human scores on 8 Smarter Balanced grade 11 explanatory responses (Spearman).
- H4 Register: casual vs. formal versions of 20 matched pairs with identical key points (paired t-test, Wilcoxon).

**Phase 2: student writing (17 first-generation students)**
- H5 Manipulation check: formality (Flesch–Kincaid, informal markers, AI style scores).
- H6 Content check: hand-scored key points and quiz scores.
- H7 Main test: mixed model of AI comprehension score on register, key points and passage, with a random intercept per student; refit with order as a covariate.

**Rubric** (1–5): comprehension and organization are content dimensions (test of bias). Grammar/mechanics, word choice and academic tone are style dimensions (manipulation check). A separate holistic prompt asks only how well the summary captures the passage.

## Grader settings

- Model: `claude-sonnet-5-5` (Anthropic API), default sampling. The model does not accept a temperature setting.
- Each text scored 5 times in fresh sessions; the median is used.
- Scoring runs: November 23, 2026.

## Files

| File | Contents |
|---|---|
| `ai_professor.py` | Rubric, prompts, grader, text measures |
| `stats_utils.py` | QWK, paired tests, Friedman, Spearman |
| `run_phase1.py` | H1–H4 |
| `run_phase2.py` | H5–H7 |
| `run_accuracy.py` | H3 only |
| `materials/passages.json` | Caves and geysers passages, key points, quizzes |
| `materials/matched_pairs.csv` | 20 matched casual/formal pairs |
| `materials/order_neighbors.csv` | Strong and weak summaries for the order test |
| `materials/smart_responses.csv` | Response IDs and official scores (texts not included) |
| `materials/phase2_summaries_TEMPLATE.csv` | Column format for Phase 2 data |
| `results/` | Phase 1 outputs |

## Running

```
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your-key      (Windows; use export on Mac/Linux)
python run_accuracy.py
python run_phase1.py
python run_phase2.py
```

## Not included

- **Participant data.** Student summaries, quiz answers and Phase 2 scores are not shared, to protect participant privacy.
- **Smarter Balanced materials.** The task prompt and response texts belong to Smarter Balanced. Response IDs and official scores are listed; the texts are available through the Smarter Annotated Response Tool (https://smart.smarterbalanced.org) and the task at https://sampleitems.smarterbalanced.org/item/200-57418.

## Sources

Passages were rewritten by the researcher from U.S. National Park Service pages (public domain): "Solution Caves" and "Geysers." The matched pairs and order-test summaries were written by the researcher.

Code was developed with assistance from Claude (Anthropic).
