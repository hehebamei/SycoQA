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

> **Note:** Standard Markdown does not natively support subtables. The layouts below use HTML so they can be displayed as grouped subtables in README-style renderers.

### Baseline Validation

<table>
  <tr>
    <td valign="top" width="50%">
      <p align="center"><strong>(a) Core Subset</strong></p>
      <p><em>RA denotes the evaluation metric used on the Core subset. "Int." denotes induction intensity.</em></p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Int.</th>
            <th>CSQA</th>
            <th>MMLU</th>
            <th>GSM8K</th>
            <th>Belebele</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Qwen2-7B</td><td>Level 1</td><td>62.44</td><td>69.30</td><td>7.70</td><td>74.21</td></tr>
          <tr><td>Qwen2-7B</td><td>Level 2</td><td>57.08</td><td>70.27</td><td>13.30</td><td>81.76</td></tr>
          <tr><td>Qwen2-7B</td><td>Level 3</td><td>3.54</td><td>14.47</td><td>4.20</td><td>35.60</td></tr>
          <tr><td>Mistral-7B</td><td>Level 1</td><td>44.48</td><td>37.61</td><td>47.73</td><td>35.71</td></tr>
          <tr><td>Mistral-7B</td><td>Level 2</td><td>40.80</td><td>42.99</td><td>37.33</td><td>49.69</td></tr>
          <tr><td>Mistral-7B</td><td>Level 3</td><td>2.65</td><td>10.90</td><td>47.05</td><td>17.39</td></tr>
          <tr><td>Llama-3-8B</td><td>Level 1</td><td>71.36</td><td>66.48</td><td>83.44</td><td>87.04</td></tr>
          <tr><td>Llama-3-8B</td><td>Level 2</td><td>61.34</td><td>66.42</td><td>80.47</td><td>86.67</td></tr>
          <tr><td>Llama-3-8B</td><td>Level 3</td><td>5.73</td><td>25.93</td><td>82.88</td><td>54.21</td></tr>
          <tr><td>Llama-3-70B</td><td>Level 1</td><td>66.03</td><td>76.85</td><td>89.42</td><td>93.82</td></tr>
          <tr><td>Llama-3-70B</td><td>Level 2</td><td>75.61</td><td>84.90</td><td>94.05</td><td>96.03</td></tr>
          <tr><td>Llama-3-70B</td><td>Level 3</td><td>42.60</td><td>63.25</td><td>95.53</td><td>90.32</td></tr>
        </tbody>
      </table>
    </td>
    <td valign="top" width="50%">
      <p align="center"><strong>(b) Extension Subset</strong></p>
      <p><em>Acc. denotes Accuracy; F1 denotes F1 score.</em></p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Setting</th>
            <th>IMDB Acc.</th>
            <th>IMDB F1</th>
            <th>Yelp Acc.</th>
            <th>Yelp F1</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Mistral-7B</td><td>Base</td><td>90.60</td><td>90.12</td><td>97.50</td><td>97.50</td></tr>
          <tr><td>Mistral-7B</td><td>Induced</td><td>47.30</td><td>18.67</td><td>53.55</td><td>25.14</td></tr>
          <tr><td>Qwen2-7B</td><td>Base</td><td>93.87</td><td>93.87</td><td>97.70</td><td>97.72</td></tr>
          <tr><td>Qwen2-7B</td><td>Induced</td><td>64.75</td><td>57.24</td><td>83.45</td><td>81.60</td></tr>
          <tr><td>Llama-3-8B</td><td>Base</td><td>94.35</td><td>94.32</td><td>97.90</td><td>97.90</td></tr>
          <tr><td>Llama-3-8B</td><td>Induced</td><td>79.05</td><td>74.04</td><td>86.85</td><td>86.06</td></tr>
          <tr><td>Llama-3-70B</td><td>Base</td><td>94.85</td><td>94.88</td><td>98.20</td><td>98.20</td></tr>
          <tr><td>Llama-3-70B</td><td>Induced</td><td>87.30</td><td>86.37</td><td>92.50</td><td>91.99</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

### Sycophantic Hallucination Detection: A Case Study

> Each entry is reported as <strong>AUROC / FPR95 / AUPR</strong>.

<table>
  <tr>
    <td valign="top" width="50%">
      <p align="center"><strong>(a) Belebele</strong></p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Int.</th>
            <th>Qwen2-7B</th>
            <th>Mistral-7B</th>
            <th>Llama-3-8B</th>
            <th>Llama-3-70B</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>MaxP</td><td>L1</td><td>56.44/90.24/77.30</td><td>58.62/91.79/43.94</td><td>55.85/85.44/88.49</td><td>68.46/83.02/96.96</td></tr>
          <tr><td>MaxP</td><td>L2</td><td>56.01/91.03/83.28</td><td>54.42/96.30/57.55</td><td>60.52/96.23/90.48</td><td>65.88/85.29/97.86</td></tr>
          <tr><td>MaxP</td><td>L3</td><td>51.60/89.45/33.75</td><td>51.20/93.61/16.87</td><td>53.54/91.48/56.47</td><td>66.19/93.98/94.61</td></tr>
          <tr><td>MaxP</td><td>Avg.</td><td>54.68/90.24/64.78</td><td><strong>54.75</strong>/93.90/<strong>39.45</strong></td><td>56.64/91.05/78.48</td><td>66.84/87.43/96.48</td></tr>
          <tr><td>PPL</td><td>L1</td><td>55.77/90.24/76.96</td><td>58.40/91.79/43.16</td><td>56.33/85.44/88.65</td><td>67.67/84.91/96.86</td></tr>
          <tr><td>PPL</td><td>L2</td><td>55.75/88.97/83.42</td><td>54.43/94.44/57.06</td><td>60.81/94.34/90.61</td><td>65.98/85.29/97.84</td></tr>
          <tr><td>PPL</td><td>L3</td><td>51.45/87.11/33.54</td><td>49.99/92.11/16.40</td><td>53.69/91.21/56.69</td><td>65.80/92.77/94.51</td></tr>
          <tr><td>PPL</td><td>Avg.</td><td>54.32/<strong>88.77</strong>/64.64</td><td>54.27/<strong>92.78</strong>/38.87</td><td>56.94/<strong>90.33</strong>/<strong>78.65</strong></td><td>66.48/87.66/96.40</td></tr>
          <tr><td>Ent.</td><td>L1</td><td>57.09/91.22/77.66</td><td>58.58/91.30/44.35</td><td>57.26/88.35/88.96</td><td>70.07/84.91/97.23</td></tr>
          <tr><td>Ent.</td><td>L2</td><td>56.82/86.21/83.57</td><td>52.97/96.91/55.91</td><td>60.55/94.34/90.26</td><td>67.64/85.29/98.01</td></tr>
          <tr><td>Ent.</td><td>L3</td><td>52.14/88.87/34.09</td><td>50.93/97.37/16.87</td><td>53.58/92.86/56.50</td><td>66.19/91.57/94.58</td></tr>
          <tr><td>Ent.</td><td>Avg.</td><td><strong>55.35</strong>/<strong>88.77</strong>/<strong>65.11</strong></td><td>54.16/95.19/39.04</td><td><strong>57.13</strong>/91.85/78.57</td><td><strong>67.97</strong>/<strong>87.26</strong>/<strong>96.61</strong></td></tr>
        </tbody>
      </table>
    </td>
    <td valign="top" width="50%">
      <p align="center"><strong>(b) CSQA</strong></p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Int.</th>
            <th>Qwen2-7B</th>
            <th>Mistral-7B</th>
            <th>Llama-3-8B</th>
            <th>Llama-3-70B</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>MaxP</td><td>L1</td><td>61.48/90.88/70.78</td><td>57.07/93.63/51.76</td><td>56.96/95.42/74.89</td><td>55.10/95.61/71.30</td></tr>
          <tr><td>MaxP</td><td>L2</td><td>64.29/87.23/70.71</td><td>56.14/94.78/48.00</td><td>53.33/93.52/62.92</td><td>57.89/93.89/80.75</td></tr>
          <tr><td>MaxP</td><td>L3</td><td>37.01/94.20/2.53</td><td>34.12/94.10/1.83</td><td>36.32/95.95/4.08</td><td>40.65/97.40/35.88</td></tr>
          <tr><td>MaxP</td><td>Avg.</td><td>54.26/90.77/48.01</td><td><strong>49.11</strong>/<strong>94.17</strong>/33.86</td><td>48.87/<strong>94.96</strong>/<strong>47.30</strong></td><td>51.21/95.63/62.64</td></tr>
          <tr><td>PPL</td><td>L1</td><td>61.52/90.88/70.69</td><td>56.70/93.90/51.44</td><td>56.65/95.00/74.75</td><td>55.30/96.55/71.34</td></tr>
          <tr><td>PPL</td><td>L2</td><td>64.41/89.10/70.86</td><td>55.91/95.02/47.88</td><td>53.17/94.44/62.93</td><td>57.73/93.01/80.49</td></tr>
          <tr><td>PPL</td><td>L3</td><td>38.53/94.56/2.59</td><td>33.95/94.70/1.82</td><td>37.34/96.20/4.14</td><td>40.82/97.22/36.03</td></tr>
          <tr><td>PPL</td><td>Avg.</td><td>54.82/91.51/48.05</td><td>48.85/94.54/33.71</td><td><strong>49.05</strong>/95.21/47.27</td><td>51.28/95.59/62.62</td></tr>
          <tr><td>Ent.</td><td>L1</td><td>62.97/91.79/72.09</td><td>57.14/94.16/52.29</td><td>56.10/95.42/74.57</td><td>56.03/95.61/72.17</td></tr>
          <tr><td>Ent.</td><td>L2</td><td>65.80/84.57/71.99</td><td>55.50/94.03/48.06</td><td>53.37/93.21/62.86</td><td>57.91/92.58/81.07</td></tr>
          <tr><td>Ent.</td><td>L3</td><td>36.85/92.90/2.52</td><td>31.90/96.22/1.76</td><td>34.34/97.22/3.96</td><td>40.05/97.96/35.61</td></tr>
          <tr><td>Ent.</td><td>Avg.</td><td><strong>55.21</strong>/<strong>89.75</strong>/<strong>48.87</strong></td><td>48.18/94.80/<strong>34.04</strong></td><td>47.94/95.28/47.13</td><td><strong>51.33</strong>/<strong>95.38</strong>/<strong>62.95</strong></td></tr>
        </tbody>
      </table>
    </td>
  </tr>
  <tr>
    <td valign="top" width="50%">
      <p align="center"><strong>(c) MMLU</strong></p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Int.</th>
            <th>Qwen2-7B</th>
            <th>Mistral-7B</th>
            <th>Llama-3-8B</th>
            <th>Llama-3-70B</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>MaxP</td><td>L1</td><td>54.11/93.86/68.48</td><td>55.78/91.15/41.32</td><td>53.39/93.84/69.31</td><td>54.25/92.01/78.98</td></tr>
          <tr><td>MaxP</td><td>L2</td><td>55.11/90.80/71.08</td><td>53.12/95.81/47.82</td><td>52.72/95.48/69.09</td><td>57.16/90.73/87.89</td></tr>
          <tr><td>MaxP</td><td>L3</td><td>27.16/98.79/9.27</td><td>42.62/98.32/9.19</td><td>46.15/96.07/25.51</td><td>49.35/97.14/63.14</td></tr>
          <tr><td>MaxP</td><td>Avg.</td><td>45.46/94.48/<strong>49.61</strong></td><td>50.51/<strong>95.09</strong>/32.78</td><td>50.75/95.13/54.64</td><td>53.59/93.29/76.67</td></tr>
          <tr><td>PPL</td><td>L1</td><td>53.86/93.47/68.32</td><td>55.92/93.54/41.46</td><td>54.13/94.02/69.83</td><td>54.50/92.66/79.16</td></tr>
          <tr><td>PPL</td><td>L2</td><td>54.41/90.39/70.79</td><td>53.27/96.07/47.98</td><td>53.53/95.30/69.61</td><td>57.37/92.05/88.03</td></tr>
          <tr><td>PPL</td><td>L3</td><td>27.22/98.79/9.28</td><td>43.00/98.16/9.25</td><td>46.34/96.15/25.49</td><td>49.23/97.69/63.02</td></tr>
          <tr><td>PPL</td><td>Avg.</td><td>45.16/<strong>94.22</strong>/49.46</td><td>50.73/95.92/32.90</td><td><strong>51.33</strong>/95.16/<strong>54.98</strong></td><td>53.70/94.13/76.74</td></tr>
          <tr><td>Ent.</td><td>L1</td><td>54.25/94.06/68.55</td><td>56.17/93.06/42.00</td><td>53.38/93.30/69.28</td><td>54.79/91.14/79.11</td></tr>
          <tr><td>Ent.</td><td>L2</td><td>54.28/93.05/70.69</td><td>54.27/94.76/48.74</td><td>52.42/95.66/69.07</td><td>58.12/89.40/88.19</td></tr>
          <tr><td>Ent.</td><td>L3</td><td>28.29/98.79/9.40</td><td>42.30/98.16/9.10</td><td>45.92/96.23/25.57</td><td>49.34/97.14/63.52</td></tr>
          <tr><td>Ent.</td><td>Avg.</td><td><strong>45.61</strong>/95.30/49.55</td><td><strong>50.91</strong>/95.33/<strong>33.28</strong></td><td>50.57/<strong>95.06</strong>/54.64</td><td><strong>54.08</strong>/<strong>92.56</strong>/<strong>76.94</strong></td></tr>
        </tbody>
      </table>
    </td>
    <td valign="top" width="50%">
      <p align="center"><strong>(d) GSM8K</strong></p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Int.</th>
            <th>Qwen2-7B</th>
            <th>Mistral-7B</th>
            <th>Llama-3-8B</th>
            <th>Llama-3-70B</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>MaxP</td><td>L1</td><td>56.20/90.20/9.34</td><td>57.34/93.51/56.10</td><td>74.06/71.16/91.42</td><td>71.42/74.74/94.20</td></tr>
          <tr><td>MaxP</td><td>L2</td><td>55.88/91.49/15.02</td><td>57.34/93.51/56.10</td><td>65.35/86.35/86.56</td><td>63.88/88.07/96.12</td></tr>
          <tr><td>MaxP</td><td>L3</td><td>60.94/94.94/7.68</td><td>60.38/87.61/55.15</td><td>57.29/91.30/85.26</td><td>52.04/96.34/95.76</td></tr>
          <tr><td>MaxP</td><td>Avg.</td><td>57.67/92.21/10.68</td><td>58.35/91.54/55.78</td><td>65.57/82.94/87.75</td><td>62.45/86.38/95.36</td></tr>
          <tr><td>PPL</td><td>L1</td><td>55.80/89.50/9.34</td><td>57.78/93.94/56.72</td><td>74.78/70.79/91.79</td><td>71.36/75.77/94.26</td></tr>
          <tr><td>PPL</td><td>L2</td><td>56.37/91.49/15.22</td><td>57.78/93.94/56.72</td><td>65.77/86.03/86.72</td><td>64.16/90.83/96.18</td></tr>
          <tr><td>PPL</td><td>L3</td><td>61.36/92.92/7.88</td><td>60.07/90.60/54.79</td><td>56.99/91.67/85.25</td><td>52.32/95.12/95.81</td></tr>
          <tr><td>PPL</td><td>Avg.</td><td>57.84/91.30/10.81</td><td>58.54/92.83/56.08</td><td>65.85/82.83/87.92</td><td>62.61/87.24/95.42</td></tr>
          <tr><td>Ent.</td><td>L1</td><td>57.31/87.28/9.33</td><td>58.06/91.77/57.39</td><td>75.74/68.91/92.12</td><td>72.98/71.65/94.33</td></tr>
          <tr><td>Ent.</td><td>L2</td><td>56.47/91.18/15.06</td><td>58.06/91.77/57.39</td><td>66.70/85.40/87.18</td><td>64.76/87.16/96.26</td></tr>
          <tr><td>Ent.</td><td>L3</td><td>63.27/92.52/8.96</td><td>60.29/89.74/55.13</td><td>58.04/89.49/85.63</td><td>51.74/97.56/95.70</td></tr>
          <tr><td>Ent.</td><td>Avg.</td><td><strong>59.02</strong>/<strong>90.33</strong>/<strong>11.12</strong></td><td><strong>58.80</strong>/<strong>91.09</strong>/<strong>56.64</strong></td><td><strong>66.83</strong>/<strong>81.27</strong>/<strong>88.31</strong></td><td><strong>63.16</strong>/<strong>85.46</strong>/<strong>95.43</strong></td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

---

**SycoQA** is intended to serve as a dataset for understanding and evaluating sycophantic hallucinations in LLMs.
