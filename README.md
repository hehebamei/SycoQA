# SycoQA

**SycoQA** is a dataset and data construction framework for studying **sycophantic hallucinations** in large language models (LLMs). It is designed to evaluate whether LLMs abandon correct judgments and follow misleading user cues under controlled induction settings.

## Overview

Sycophantic hallucination refers to a specific failure mode in which an LLM produces an incorrect answer not because it lacks the relevant knowledge, but because it is influenced by a user’s misleading claim, authority framing, or false justification. This behavior raises important concerns for the reliability, safety, and controllability of LLMs.

To facilitate systematic research on this problem, **SycoQA** provides a unified benchmark for controlled evaluation of sycophantic behavior across multiple tasks and induction paradigms.

This project is motivated by three observations:

* Existing studies often focus on **task-specific** or **prompt-specific** settings.
* Recent work has started to explore downstream objectives such as **detection** and **mitigation**, but lacks a common benchmark.
* There is still no widely adopted resource for **standardized evaluation** and **behavioral analysis** of sycophantic hallucinations.

## What SycoQA Provides

SycoQA is designed as a reusable benchmark resource for:

* **Controlled evaluation** of sycophantic hallucination in LLMs
* **Behavioral analysis** under progressively increasing misleading pressure
* **Benchmarking downstream methods**, such as detection and mitigation
* **Studying the distinction** between ordinary hallucination and sycophantic hallucination

## Benchmark Design

SycoQA is organized under a shared QA-style interface and supports multiple settings for evaluating sycophantic failures.

### 1. Context-independent setting

This setting evaluates whether a model changes its answer under misleading user cues **without external evidence**. It tests whether the model abandons an internally available correct judgment.

### 2. Context-dependent setting

This setting evaluates whether a model follows misleading user interpretations **even when explicit local evidence is provided in the context**. It is used to test grounded sycophantic hallucination, where the model may ignore evidence and align with the user instead.

### 3. Progressive induction paradigm

SycoQA supports multiple induction levels with **increasing sycophantic pressure**, such as:

* misleading user claims
* fabricated supporting details
* authority-based framing
* reasoning-based false justification

This design enables fine-grained analysis of how model behavior shifts under progressively stronger user-induced pressure.

## Repository Structure

```text
SycoQA/
├── Data/                         # Benchmark data and processed samples
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

## Quick Start

A typical workflow includes three stages.

### 1. Prepare data

Place or generate benchmark data under the `Data/` directory.

### 2. Run inference

Use the inference or main script to obtain model predictions:

```bash
python Ha_main.py
```

or

```bash
python Ha_inference.py
```

### 3. Evaluate results

Compute benchmark scores using:

```bash
python score.py
```

> The exact arguments and configuration may depend on your experimental setup. Please modify `config_pool.py` and `prompt_pool.py` accordingly.

## Research Goals

This project aims to support research on the following questions:

* When do LLMs exhibit sycophantic hallucinations?
* How does sycophantic pressure affect model behavior across tasks?
* How can we distinguish **ordinary hallucination** from **sycophantic hallucination**?
* Can sycophantic failures be detected before the final wrong answer is produced?
* How can such failures be mitigated in a robust and generalizable way?

## Potential Applications

SycoQA can be used for:

* evaluating the robustness of instruction-following LLMs
* benchmarking hallucination detection methods
* studying alignment failures under misleading prompts
* testing mitigation strategies for user-induced hallucinations
* analyzing behavior differences across model scales and families

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@misc{sycoqa,
  title={SycoQA: A Dataset for Evaluating Sycophantic Hallucinations in Large Language Models},
  author={Your Name(s)},
  year={2026},
  note={GitHub repository}
}
```

## TODO

* [ ] Release the full benchmark data
* [ ] Release evaluation scripts and standardized metrics
* [ ] Add detailed examples for context-independent and context-dependent settings
* [ ] Provide baseline results for open-source LLMs
* [ ] Add support for detection and mitigation benchmarks

## Contact

For questions, suggestions, or collaboration, please open an issue or contact the repository maintainer.

---

**SycoQA** is intended to serve as a standardized testbed for understanding and evaluating sycophantic hallucinations in LLMs.
