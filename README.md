# Resolving Disagreement Between Four Sentiment Models at Scale

**A sentiment analysis case study — 498,147 Reddit comments, r/LosAngeles**

Graduate text mining project · Individual contribution: sentiment analysis component

---

| | | | |
|---|---|---|---|
| **498,147** | **4** | **39.9 pts** | **70.0%** |
| comments analysed | sentiment methods compared | spread on identical text | best model accuracy |

> [!IMPORTANT]
> **Google Drive environment:** The programming work in this analysis was performed in a Google Drive file environment and folder structure (via Google Colab). The setup for file inputs and outputs in the notebooks was therefore based on working in Google Drive. To re-run the notebooks, either recreate that Drive folder structure or update the file paths to match your own environment.

---

## Overview

As part of a graduate team project analysing community discourse on r/LosAngeles, I owned the sentiment analysis component: applying multiple sentiment classification methods to a corpus of 498,147 Reddit comments, determining which method could actually be trusted, and using the result to show how sentiment varies across the discussion themes a teammate's topic model had identified.

Two things distinguish this from a standard "run four sentiment models and report percentages" exercise. First, the four methods did not agree — not at the margins, but fundamentally, disagreeing by up to forty percentage points on how negative the same comments were. Second, resolving that disagreement required building an independent, bias-resistant evaluation process rather than trusting any model's self-reported confidence.

## The Problem

Sentiment analysis is often treated as solved: pick a library, run it, report the numbers. That assumption broke down almost immediately. Scoring the same 498,147 comments with two lexicon-based methods (TextBlob, VADER) and two transformer-based methods (Twitter-RoBERTa, multilingual BERT) produced four different pictures of the same community.

- Negative-comment share ranged from **22.7% to 62.6%** — a 40-point spread on identical text
- All four methods agreed on a shared label for only **19.5%** of comments
- For nearly a quarter of the corpus, the four methods collectively produced every possible label

No model's output could resolve which one was correct, because the outputs were the thing being evaluated. The core of the project became building the evidence needed to choose responsibly — and being explicit about what that evidence could and could not support.

## Approach

### 1. Preparing text for sentiment, not just for word counts

The dataset's standard cleaned text was lowercased with emoji stripped — appropriate for topic modelling, but sentiment methods rely on exactly what that strips. VADER treats capitalisation as an intensity signal; transformer models tokenise emoji as meaningful input. I derived a second text representation preserving both, then measured the effect directly rather than assuming it mattered: **0.04%** of TextBlob's labels changed, **0.80%** of VADER's, and **5.08%** of RoBERTa's — a result that later shaped how much weight the decision carried in the final analysis.

### 2. Establishing a lexicon baseline — and finding a hidden flaw

Running TextBlob and VADER first (cheap, CPU-only, fast to validate the pipeline end to end) surfaced a specific and consequential bug in how "neutral" was being measured. Both libraries return a score of exactly zero when they recognise none of the words in a comment. That non-answer was being recorded as "neutral," indistinguishable from a genuine judgment of balance.

> **Finding:** 75.4% of TextBlob's "neutral" labels and 92.6% of VADER's were unrecognised text, not genuine neutral judgments. Actual neutral sentiment was closer to 8.5% and 1.7% of the corpus respectively.

### 3. Adding transformer models — and catching a second bug

Twitter-RoBERTa (social-media-trained, natively 3-class) and multilingual BERT (general-purpose, 5-star output) were added to test whether domain-matched training actually improves accuracy. Converting BERT's 5-star rating into three sentiment classes seemed straightforward: map the most likely star to a class. Comparing that against a corrected version — summing probability mass across each class rather than taking the single most likely star — turned up an 11.9% disagreement rate and revealed the naive approach was structurally biased toward neutral, since negative or positive probability could split across two adjacent star ratings and lose to a three-star plurality.

> **Finding:** Correcting the conversion changed 11.9% of BERT's labels and nearly halved its reported neutral share, from 19.8% to 8.1%.

### 4. Scaling to 498,147 comments on a GPU budget

Initial development ran on a 100,000-comment sample; once actual throughput was measured, I extended all four methods to the full corpus. Transformer inference was implemented as resumable, checkpointed chunks of 25,000 comments on a Colab T4 GPU, so a session disconnect — a real risk on free-tier Colab — cost at most one chunk rather than the entire run. Comparing sample-based results against the full-corpus results afterward showed a maximum deviation of 0.26 percentage points, confirming the earlier sampling decision had been sound.

### 5. Comparing four methods honestly

Simple pairwise agreement is misleading when class distributions are skewed — two methods that both lean negative will agree by coincidence. I used **Cohen's kappa** instead, which discounts chance agreement, and the correction changed the story: RoBERTa and BERT had the highest raw agreement of any pair (56.2%), but the two lexicon methods corresponded more closely once chance was discounted. Ordering all four methods by how negative they ran revealed the real pattern — agreement tracked position on that negativity spectrum, not model architecture. A lexicon and a transformer sitting close together on the spectrum agreed with each other more than two transformers at opposite ends did.

### 6. Building a benchmark that could be trusted

Because the four methods contradicted each other, I built an independent, hand-labelled reference: 100 comments sampled at random and labelled blind, plus a second batch of 25 comments — selected specifically because all four methods disagreed — reserved for qualitative error analysis only. Two design choices protected the benchmark from bias: both batches were shuffled together so labelling quality stayed uniform across each, and no model predictions were visible during annotation, which would otherwise anchor judgment toward whatever the models had already guessed. A written coding rule set was fixed before annotation began, the most important of which distinguished a comment's *expressed attitude* from its *subject matter* — a factual, neutral statement about a bad situation is not the same as a comment expressing negativity.

Against that benchmark, **Twitter-RoBERTa scored 70.0% accuracy and 70.3 macro-F1**, roughly twenty points ahead of the other three methods, which proved statistically indistinguishable from one another once tested with pairwise significance tests on their points of disagreement (p = 0.000–0.003 against each alternative). Macro-F1 — chosen in advance specifically because it does not let a large class dominate — proved necessary: two methods with identical overall accuracy (50.0%) differed by five points on macro-F1 because one of them almost never identified neutral comments correctly.

### 7. Connecting sentiment to topic

The final deliverable combined the selected sentiment model with a teammate's topic model (five LDA topics, chosen on coherence and diversity). Rather than trusting an exported assignment file, I reproduced the topic model independently within my own pipeline — partly to verify topic labels attached to the correct comment IDs, since the original workflow used a train/test split that could have reordered rows, and partly to confirm the result was reproducible at all. Topic sizes matched the original run to within **0.06%**.

Reviewing the topic labels against their actual content — sampling documents across the full range of assignment confidence rather than only the highest-confidence examples — surfaced one mislabelled topic and identified two of the five as semantically incoherent groupings rather than genuine themes. Both findings were incorporated into how the final results were interpreted and reported.

## Key Findings

| Finding | Detail |
|---|---|
| Four methods, one corpus, wildly different answers | Negative-comment share ranged from 22.7% to 62.6% on identical text |
| "Neutral" often meant "unrecognised" | Two lexicon libraries recorded dictionary misses as neutral, inflating that class by 4–50x |
| A default model-conversion choice was quietly biased | Correcting a 5-star-to-3-class collapse changed 11.9% of labels and nearly halved the neutral share |
| Model choice was settled with evidence | A blind human benchmark found one method significantly more accurate (70.0% vs. ~50%) |
| Sentiment differs sharply across themes | Politics and housing ran 27.8 points more negative than neighbourhood/daily-life discussion |
| Engagement does not follow negativity | The most-discussed themes were not the most negative ones |

## Answering the Research Questions

**How does sentiment differ across discussion themes?** — ✅ **Fully answered.** Sentiment varies by 27.8 percentage points between the most and least critical themes, and the ranking survives every robustness check applied.

**Do the most negative themes also see the highest engagement?** — ❌ **Not supported.** The most negative theme ranked only third by comment volume; the least negative theme ranked second. Engagement appears to track how central a theme is to daily life rather than how critically it is discussed.

## Tools & Technologies

| Category | Tools |
|---|---|
| Languages & core libraries | Python, pandas, NumPy, SciPy |
| Modelling | scikit-learn, Hugging Face Transformers (PyTorch) |
| Sentiment methods | TextBlob, VADER, Twitter-RoBERTa, multilingual BERT |
| Compute | Google Colab (T4 GPU), resumable checkpointed inference |
| Data formats | Parquet / PyArrow, Excel (annotation workflow) |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebooks |

Models: `cardiffnlp/twitter-roberta-base-sentiment-latest` · `nlptown/bert-base-multilingual-uncased-sentiment`

## Skills Demonstrated

- End-to-end NLP pipeline design and execution on a real-world dataset at scale (498K+ records)
- Comparative evaluation of lexicon-based and transformer-based sentiment methods
- Root-cause diagnosis of subtle ML bugs — a biased label conversion and a mislabelled dictionary-miss artifact — found by investigating unexpected results rather than assuming correctness
- Design of bias-resistant human evaluation protocols: blind annotation, pre-registered coding rules, held-out qualitative samples
- Applied statistical methodology — chance-corrected agreement (Cohen's kappa), macro-averaged F1 for imbalanced classes, pairwise significance testing
- Reproducibility practices in a team setting, including independently re-deriving a teammate's model output to verify data integrity before building on it
- Cloud GPU resource management — chunked, checkpointed inference resilient to session interruption
- Technical writing for multiple audiences: internal methodology documentation, results reporting mapped to research questions, and stakeholder presentation

## Reflection

The most useful habit this project reinforced was treating an unexpected result as a prompt to investigate rather than a nuisance to explain away. The lexicon abstention issue, the bias in the BERT star-conversion, and the mislabelled topic were all found because a number looked slightly off and warranted a closer look rather than a shrug. None of these findings were part of the original plan — all three ended up materially shaping the final analysis and its conclusions.
