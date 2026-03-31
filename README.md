# SycoQA

**SycoQA** is a dataset and data construction framework for studying **sycophantic hallucinations** in large language models (LLMs). It is designed to evaluate whether LLMs abandon correct judgments and follow misleading user cues under controlled induction settings.

## Overview

Sycophantic hallucination refers to a specific failure mode in which an LLM produces an incorrect answer not because it lacks the relevant knowledge, but because it is influenced by a user’s misleading claim, authority framing, or false justification. This behavior raises important concerns for the reliability, safety, and controllability of LLMs.

To facilitate systematic research on this problem, **SycoQA** provides a unified dataset for controlled evaluation of sycophantic behavior across multiple tasks and induction paradigms.

---

## Examples

<p align="center">
  <img src="https://github.com/user-attachments/assets/8420b0cb-dfb5-4b01-a666-76739dccab79" width="82%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/95affbc1-456a-40de-81df-582b27f83a66" width="82%" />
</p>

---

## Results

### Baseline Validation on the Core Subset

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc61ba85-316e-41fe-8a70-d5e6938ce263" width="82%" />
</p>

### Baseline Validation on the Extension Subset

<p align="center">
  <img src="https://github.com/user-attachments/assets/f9741751-76aa-4eba-b93c-b1971594157a" width="76%" />
</p>

### Sycophantic Hallucination Detection: A Case Study

<table align="center">
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/0142f63e-73f9-42df-bf45-7b8b64001435" width="380"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/ed9ea716-c111-4aa0-8504-b63183baa7e3" width="380"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/b4ed29e7-27f6-4974-958d-a27d899f1f13" width="380"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/38ccc783-bd39-4085-b25d-549f81b6844f" width="380"/></td>
  </tr>
</table>

---

**SycoQA** is intended to serve as a dataset for understanding and evaluating sycophantic hallucinations in LLMs.
