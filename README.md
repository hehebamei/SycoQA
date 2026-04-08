# SycoQA

**SycoQA** is a dataset and data construction framework for studying **sycophantic hallucinations** in large language models (LLMs). It is designed to evaluate whether LLMs abandon correct judgments and follow misleading user cues under controlled induction settings.

## Overview

[Dataset Overview PDF](https://github.com/user-attachments/files/26560531/Dataset_Overview.pdf)

Sycophantic hallucination refers to a specific failure mode in which an LLM produces an incorrect answer not because it lacks the relevant knowledge, but because it is influenced by a user's misleading claim, authority framing, or false justification. This behavior raises important concerns for the reliability, safety, and controllability of LLMs.

To facilitate systematic research on this problem, **SycoQA** provides a unified dataset for controlled evaluation of sycophantic behavior across multiple tasks and induction paradigms.

---

## Dataset Structure

SycoQA consists of two complementary subsets for evaluating sycophantic hallucinations in large language models under controlled induction settings:

- **Core subset**: evaluates **context-independent sycophantic hallucinations**, where the model is expected to answer correctly based on its internal knowledge, but may abandon correct judgments under misleading user cues.
- **Extension subset**: evaluates **context-dependent sycophantic hallucinations**, where the model is provided with relevant context, but may still ignore the local evidence when exposed to distorted or biased user framing.

The Core subset covers multiple capability domains, including:

- mathematical reasoning
- commonsense reasoning
- factual knowledge
- language understanding / reading comprehension

The Extension subset focuses on evidence-grounded sentiment understanding with contextual distortion.

---

## Dataset Statistics

> **Note:** "Unique" refers to the number of original base instances.

| Subset | Source dataset | Unique | Total Cases |
|---|---:|---:|---:|
| Core | GSM8K | 2,000 | 8,000 |
| Core | CSQA | 1,221 | 4,884 |
| Core | MMLU | 2,500 | 10,000 |
| Core | Belebele | 900 | 3,600 |
| Extension | IMDB | 2,000 | 4,000 |
| Extension | Yelp | 2,000 | 4,000 |
| **Total** | -- | **10,621** | **34,484** |

---

## Repository Structure

```text
SycoQA/
├── Data/
│   ├── IMDB.jsonl
│   ├── Yelp.jsonl
│   ├── belebele.jsonl
│   ├── commonsenseqa.jsonl
│   ├── gsm2k8k.jsonl
│   ├── mmlu.jsonl
│   └── load_data.py
├── Evaluation/
│   ├── eval.py
│   └── match.py
├── Scripts/
│   ├── Ha_infer.sh
│   └── llm_infer.sh
├── Data_generation_Prompt.py
├── Ha_inference.py
├── Ha_main.py
├── config_pool.py
├── prompt_pool.py
├── requirements.txt
└── score.py
```

---

## Data Schema

Each line in a data file is a JSON object representing one evaluation instance.
SycoQA currently uses a **task-dependent JSONL format** rather than a fully unified schema across all sources.
In released files, the most common fields are:

- `id`: sample identifier. Instances with the same `id` correspond to different prompting variants derived from the same original example.
- `en`: model input text in English, including the question and, when applicable, the induced misleading cue.
- `question`: used in some datasets (e.g., GSM8K-style data) instead of `en` to store the model input text.
- `answer`: gold answer or gold label.
- `review_text`: original review text for sentiment-based examples in the Extension subset.

### Common Interpretation

Although the field names vary slightly across files, each instance can be understood as containing:

1. an **original task item**,
2. optionally an **induced misleading cue** appended to the input,
3. the **gold answer / label** used for evaluation.

### Field Semantics

| Field | Type | Description |
|---|---|---|
| `id` | int | Identifier of the original source item; repeated across prompt variants derived from the same example |
| `en` | string | Full English input prompt used for evaluation; may include the original question only, or the question plus misleading induction |
| `question` | string | Alternative field name used in some files (mainly math-style data) for the evaluation prompt |
| `review_text` | string | Original review / evidence text for sentiment-based Extension data |
| `answer` | string | Gold answer or gold label used for evaluation |

---

## Examples

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8420b0cb-dfb5-4b01-a666-76739dccab79" width="420"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/95affbc1-456a-40de-81df-582b27f83a66" width="420"/>
    </td>
  </tr>
</table>

---

## Results

### Baseline Validation on the Core Subset

> **Note:** RA denotes the evaluation metric used on the Core subset. "Int." denotes induction intensity.

| Model | Int. | CSQA | MMLU | GSM8K | Belebele |
|---|---|---:|---:|---:|---:|
| Qwen2-7B | Level 1 | 62.44 | 69.30 | 7.70 | 74.21 |
| Qwen2-7B | Level 2 | 57.08 | 70.27 | 13.30 | 81.76 |
| Qwen2-7B | Level 3 | 3.54 | 14.47 | 4.20 | 35.60 |
| Mistral-7B | Level 1 | 44.48 | 37.61 | 47.73 | 35.71 |
| Mistral-7B | Level 2 | 40.80 | 42.99 | 37.33 | 49.69 |
| Mistral-7B | Level 3 | 2.65 | 10.90 | 47.05 | 17.39 |
| Llama-3-8B | Level 1 | 71.36 | 66.48 | 83.44 | 87.04 |
| Llama-3-8B | Level 2 | 61.34 | 66.42 | 80.47 | 86.67 |
| Llama-3-8B | Level 3 | 5.73 | 25.93 | 82.88 | 54.21 |
| Llama-3-70B | Level 1 | 66.03 | 76.85 | 89.42 | 93.82 |
| Llama-3-70B | Level 2 | 75.61 | 84.90 | 94.05 | 96.03 |
| Llama-3-70B | Level 3 | 42.60 | 63.25 | 95.53 | 90.32 |

### Baseline Validation on the Extension Subset

> **Note:** Acc. denotes Accuracy; F1 denotes F1 score.

| Model | Setting | IMDB Acc. | IMDB F1 | Yelp Acc. | Yelp F1 |
|---|---|---:|---:|---:|---:|
| Mistral-7B | Base | 90.60 | 90.12 | 97.50 | 97.50 |
| Mistral-7B | Induced | 47.30 | 18.67 | 53.55 | 25.14 |
| Qwen2-7B | Base | 93.87 | 93.87 | 97.70 | 97.72 |
| Qwen2-7B | Induced | 64.75 | 57.24 | 83.45 | 81.60 |
| Llama-3-8B | Base | 94.35 | 94.32 | 97.90 | 97.90 |
| Llama-3-8B | Induced | 79.05 | 74.04 | 86.85 | 86.06 |
| Llama-3-70B | Base | 94.85 | 94.88 | 98.20 | 98.20 |
| Llama-3-70B | Induced | 87.30 | 86.37 | 92.50 | 91.99 |

### Sycophantic Hallucination Detection: A Case Study

> Each entry is reported as **AUROC / FPR95 / AUPR**.

#### Belebele

| Method | Int. | Qwen2-7B | Mistral-7B | Llama-3-8B | Llama-3-70B |
|---|---|---|---|---|---|
| MaxP | L1 | 56.44/90.24/77.30 | 58.62/91.79/43.94 | 55.85/85.44/88.49 | 68.46/83.02/96.96 |
| MaxP | L2 | 56.01/91.03/83.28 | 54.42/96.30/57.55 | 60.52/96.23/90.48 | 65.88/85.29/97.86 |
| MaxP | L3 | 51.60/89.45/33.75 | 51.20/93.61/16.87 | 53.54/91.48/56.47 | 66.19/93.98/94.61 |
| MaxP | Avg. | 54.68/90.24/64.78 | **54.75**/93.90/**39.45** | 56.64/91.05/78.48 | 66.84/87.43/96.48 |
| PPL | L1 | 55.77/90.24/76.96 | 58.40/91.79/43.16 | 56.33/85.44/88.65 | 67.67/84.91/96.86 |
| PPL | L2 | 55.75/88.97/83.42 | 54.43/94.44/57.06 | 60.81/94.34/90.61 | 65.98/85.29/97.84 |
| PPL | L3 | 51.45/87.11/33.54 | 49.99/92.11/16.40 | 53.69/91.21/56.69 | 65.80/92.77/94.51 |
| PPL | Avg. | 54.32/**88.77**/64.64 | 54.27/**92.78**/38.87 | 56.94/**90.33**/**78.65** | 66.48/87.66/96.40 |
| Ent. | L1 | 57.09/91.22/77.66 | 58.58/91.30/44.35 | 57.26/88.35/88.96 | 70.07/84.91/97.23 |
| Ent. | L2 | 56.82/86.21/83.57 | 52.97/96.91/55.91 | 60.55/94.34/90.26 | 67.64/85.29/98.01 |
| Ent. | L3 | 52.14/88.87/34.09 | 50.93/97.37/16.87 | 53.58/92.86/56.50 | 66.19/91.57/94.58 |
| Ent. | Avg. | **55.35**/**88.77**/**65.11** | 54.16/95.19/39.04 | **57.13**/91.85/78.57 | **67.97**/**87.26**/**96.61** |

#### CSQA

| Method | Int. | Qwen2-7B | Mistral-7B | Llama-3-8B | Llama-3-70B |
|---|---|---|---|---|---|
| MaxP | L1 | 61.48/90.88/70.78 | 57.07/93.63/51.76 | 56.96/95.42/74.89 | 55.10/95.61/71.30 |
| MaxP | L2 | 64.29/87.23/70.71 | 56.14/94.78/48.00 | 53.33/93.52/62.92 | 57.89/93.89/80.75 |
| MaxP | L3 | 37.01/94.20/2.53 | 34.12/94.10/1.83 | 36.32/95.95/4.08 | 40.65/97.40/35.88 |
| MaxP | Avg. | 54.26/90.77/48.01 | **49.11**/**94.17**/33.86 | 48.87/**94.96**/**47.30** | 51.21/95.63/62.64 |
| PPL | L1 | 61.52/90.88/70.69 | 56.70/93.90/51.44 | 56.65/95.00/74.75 | 55.30/96.55/71.34 |
| PPL | L2 | 64.41/89.10/70.86 | 55.91/95.02/47.88 | 53.17/94.44/62.93 | 57.73/93.01/80.49 |
| PPL | L3 | 38.53/94.56/2.59 | 33.95/94.70/1.82 | 37.34/96.20/4.14 | 40.82/97.22/36.03 |
| PPL | Avg. | 54.82/91.51/48.05 | 48.85/94.54/33.71 | **49.05**/95.21/47.27 | 51.28/95.59/62.62 |
| Ent. | L1 | 62.97/91.79/72.09 | 57.14/94.16/52.29 | 56.10/95.42/74.57 | 56.03/95.61/72.17 |
| Ent. | L2 | 65.80/84.57/71.99 | 55.50/94.03/48.06 | 53.37/93.21/62.86 | 57.91/92.58/81.07 |
| Ent. | L3 | 36.85/92.90/2.52 | 31.90/96.22/1.76 | 34.34/97.22/3.96 | 40.05/97.96/35.61 |
| Ent. | Avg. | **55.21**/**89.75**/**48.87** | 48.18/94.80/**34.04** | 47.94/95.28/47.13 | **51.33**/**95.38**/**62.95** |

#### MMLU

| Method | Int. | Qwen2-7B | Mistral-7B | Llama-3-8B | Llama-3-70B |
|---|---|---|---|---|---|
| MaxP | L1 | 54.11/93.86/68.48 | 55.78/91.15/41.32 | 53.39/93.84/69.31 | 54.25/92.01/78.98 |
| MaxP | L2 | 55.11/90.80/71.08 | 53.12/95.81/47.82 | 52.72/95.48/69.09 | 57.16/90.73/87.89 |
| MaxP | L3 | 27.16/98.79/9.27 | 42.62/98.32/9.19 | 46.15/96.07/25.51 | 49.35/97.14/63.14 |
| MaxP | Avg. | 45.46/94.48/**49.61** | 50.51/**95.09**/32.78 | 50.75/95.13/54.64 | 53.59/93.29/76.67 |
| PPL | L1 | 53.86/93.47/68.32 | 55.92/93.54/41.46 | 54.13/94.02/69.83 | 54.50/92.66/79.16 |
| PPL | L2 | 54.41/90.39/70.79 | 53.27/96.07/47.98 | 53.53/95.30/69.61 | 57.37/92.05/88.03 |
| PPL | L3 | 27.22/98.79/9.28 | 43.00/98.16/9.25 | 46.34/96.15/25.49 | 49.23/97.69/63.02 |
| PPL | Avg. | 45.16/**94.22**/49.46 | 50.73/95.92/32.90 | **51.33**/95.16/**54.98** | 53.70/94.13/76.74 |
| Ent. | L1 | 54.25/94.06/68.55 | 56.17/93.06/42.00 | 53.38/93.30/69.28 | 54.79/91.14/79.11 |
| Ent. | L2 | 54.28/93.05/70.69 | 54.27/94.76/48.74 | 52.42/95.66/69.07 | 58.12/89.40/88.19 |
| Ent. | L3 | 28.29/98.79/9.40 | 42.30/98.16/9.10 | 45.92/96.23/25.57 | 49.34/97.14/63.52 |
| Ent. | Avg. | **45.61**/95.30/49.55 | **50.91**/95.33/**33.28** | 50.57/**95.06**/54.64 | **54.08**/**92.56**/**76.94** |

#### GSM8K

| Method | Int. | Qwen2-7B | Mistral-7B | Llama-3-8B | Llama-3-70B |
|---|---|---|---|---|---|
| MaxP | L1 | 56.20/90.20/9.34 | 57.34/93.51/56.10 | 74.06/71.16/91.42 | 71.42/74.74/94.20 |
| MaxP | L2 | 55.88/91.49/15.02 | 57.34/93.51/56.10 | 65.35/86.35/86.56 | 63.88/88.07/96.12 |
| MaxP | L3 | 60.94/94.94/7.68 | 60.38/87.61/55.15 | 57.29/91.30/85.26 | 52.04/96.34/95.76 |
| MaxP | Avg. | 57.67/92.21/10.68 | 58.35/91.54/55.78 | 65.57/82.94/87.75 | 62.45/86.38/95.36 |
| PPL | L1 | 55.80/89.50/9.34 | 57.78/93.94/56.72 | 74.78/70.79/91.79 | 71.36/75.77/94.26 |
| PPL | L2 | 56.37/91.49/15.22 | 57.78/93.94/56.72 | 65.77/86.03/86.72 | 64.16/90.83/96.18 |
| PPL | L3 | 61.36/92.92/7.88 | 60.07/90.60/54.79 | 56.99/91.67/85.25 | 52.32/95.12/95.81 |
| PPL | Avg. | 57.84/91.30/10.81 | 58.54/92.83/56.08 | 65.85/82.83/87.92 | 62.61/87.24/95.42 |
| Ent. | L1 | 57.31/87.28/9.33 | 58.06/91.77/57.39 | 75.74/68.91/92.12 | 72.98/71.65/94.33 |
| Ent. | L2 | 56.47/91.18/15.06 | 58.06/91.77/57.39 | 66.70/85.40/87.18 | 64.76/87.16/96.26 |
| Ent. | L3 | 63.27/92.52/8.96 | 60.29/89.74/55.13 | 58.04/89.49/85.63 | 51.74/97.56/95.70 |
| Ent. | Avg. | **59.02**/**90.33**/**11.12** | **58.80**/**91.09**/**56.64** | **66.83**/**81.27**/**88.31** | **63.16**/**85.46**/**95.43** |

---

**SycoQA** is intended to serve as a dataset for understanding and evaluating sycophantic hallucinations in LLMs.
