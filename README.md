# SycoQA

**SycoQA** is a dataset and data construction framework for studying **sycophantic hallucinations** in large language models (LLMs). It is designed to evaluate whether LLMs abandon correct judgments and follow misleading user cues under controlled induction settings.

## Overview

Sycophantic hallucination refers to a specific failure mode in which an LLM produces an incorrect answer not because it lacks the relevant knowledge, but because it is influenced by a user’s misleading claim, authority framing, or false justification. This behavior raises important concerns for the reliability, safety, and controllability of LLMs.

To facilitate systematic research on this problem, **SycoQA** provides a unified dataset for controlled evaluation of sycophantic behavior across multiple tasks and induction paradigms.


## Benchmark Design

SycoQA is organized under a shared QA-style interface and supports multiple settings for evaluating sycophantic failures.

### 1. Context-independent setting

This setting evaluates whether a model changes its answer under misleading user cues **without external evidence**. It tests whether the model abandons an internally available correct judgment.

### 2. Context-dependent setting

This setting evaluates whether a model follows misleading user interpretations **even when explicit local evidence is provided in the context**. It is used to test grounded sycophantic hallucination, where the model may ignore evidence and align with the user instead.

### 3. Progressive induction paradigm

SycoQA supports multiple induction levels with **increasing sycophantic pressure**, such as:

* Misleading statement induction
* fabricated supporting details
* authority-based framing
* reasoning-based false justification

This design enables fine-grained analysis of how model behavior shifts under progressively stronger user-induced pressure.

## Repository Structure

```text
SycoQA/
├── Data/                         # Datasets and processed samples
├── Evaluation/                   # Evaluation scripts and outputs
├── Scripts/                      # Utility scripts for running experiments
├── Data_generation_Prompt.py     # Prompt templates or generation utilities
├── Ha_inference.py               # Inference pipeline
├── Ha_main.py                    # Main entry for experiments
├── config_pool.py                # Configuration settings
├── prompt_pool.py                # Prompt templates / induction pool
├── score.py                      # Scoring and metric computation
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/hehebamei/SycoQA.git
cd SycoQA
pip install -r requirements.txt
```




---

**SycoQA** is intended to serve as a standardized testbed for understanding and evaluating sycophantic hallucinations in LLMs.
