DATASET_PROMPTS = {
    "mmmlu": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion:\n{input_data}\n",
    "belebele": "Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering.\n\n{input_data}\n",
    "commonsenseqa": "Answer the following multiple choice common-sense reasoning question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step and output the reasoning process before answering.\n\n{input_data}",
    "answer": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\answer{{}}\n",
    "gsm2k8k": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
}






