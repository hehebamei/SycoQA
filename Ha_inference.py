
import os
import sys
import time
import random

import pickle
import argparse
import scipy.spatial
import math
import json
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
import seaborn as sns
from collections import Counter

import numpy as np
import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

# project_root_path = os.environ["PROJECT_PATH"]

sys.path.append(project_root_path)
from Data.load_data import DatasetInfo
from prompt_pool import *
from score import OutputScoreInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference:
    def __init__(self, model_info: dict, dataset_info: dict, verbose: dict):
        self.model_info = model_info
        self.dataset_info = dataset_info
        self.verbose = verbose
        self.is_debug = dataset_info.get("is_debug", False)
        self.model = self.model_info["model_ckpt"]
        self.model_name = self.model_info["model_name"]
        self.generation_config = self.model_info["generation_config"]
        self.tokenizer = self.model_info["tokenizer"]
        self.max_output_token = self.model_info["max_output_token"]

        self.dataset_name = self.dataset_info["dataset_name"]
        self.data_loader = DatasetInfo(self.dataset_name)
        self.data_all = self.data_loader.data
        self.data_size = self.data_loader.data_size
        self.language = self.dataset_info["language"]

        self.sample_info = {}

    def dataset_inference(self):
        self.greedy_inference()
        # self.greedy_inference(1, step)


    def greedy_inference(self):
        if self.is_debug:
            self.data_size = 8
            self.data_all = self.data_all[:self.data_size]
            print("Debug mode is on, only process 5 samples.")
        for i in tqdm(range(self.data_size)):
            print("*" * 30 + f" index {str(i)} " + "*" * 30)
            sample = self.data_all[i]
            input_data, output_data, model_input, input_ids = self.parse_input(sample)
            self.sample_info = {
                "input": {
                    "raw_input_data": input_data,
                    "model_input": model_input,
                    "model_input_ids": input_ids,
                },
                "output": {
                    "raw_output_data": output_data,
                }
            }

            # with torch.no_grad():
            with torch.inference_mode():
                generation_output = self.model_inference()
                self.sample_info["output"]["output_scores"] = generation_output.scores
                self.sample_info["output"]["output_seq"] = generation_output.sequences
                self.sample_info["output"]["attentions"] = generation_output.attentions
                self.sample_info["output"][
                    "all_token_hidden_states"] = generation_output.hidden_states  # output_len x layer_num x sampling_num x beam_search x hidden_dim
                self.sample_info["output"]["output_len"] = min(self.max_output_token, len(generation_output.scores))

                input_seq, output_seq, maxprob, ppl, entropy = self.print_output()
                output = {'id': i,
                          'answer_type': sample["answer_type"] if self.dataset_name == "theoremqa" else "",
                          'input_seq': self.sample_info["input"]["model_input"],
                          'output_seq': output_seq,
                          'input_seq': input_seq,
                          'maxprob': maxprob,
                          'ppl': ppl,
                          'entropy': entropy}
                print(f"verbose{self.verbose}")
                if self.verbose["save_output"]: self.save_output(output, i)

                hidden_states = self.print_hidden_states()
                if self.verbose["save_hidden_states"]: self.save_hidden_states(hidden_states, i)

                attentions = generation_output.attentions
                self.save_attentions(attentions, i)



    def model_inference(self):
        input_ids = self.sample_info["input"]["model_input_ids"]
        self.model.eval()
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] \
            if "Llama" in self.model_name else self.tokenizer.eos_token_id

        time_start = time.time()
        generation_output = self.model.generate(
            input_ids=input_ids.to(device),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            max_new_tokens=self.max_output_token,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            do_sample=False,
        )
        time_end = time.time()
        print(f'inference time: {round(time_end - time_start, 4)}')

        return generation_output

    def parse_input(self, sample):
        if self.dataset_name =="answer":
            input_data = sample["prompt"][0]["content"]
            output_data = sample["base"]["correct_answer"]
        elif "commonsense" in self.dataset_name:
            input_data = sample["en"]
            index = input_data.find(":")
            input_data = input_data[index + 1:]
            output_data = sample["answer"]
        elif "belebele" in self.dataset_name:
            input_data = sample["en"]
            output_data = sample["answer"]
        elif "mmmlu" in self.dataset_name:
            input_data = sample["en"]
            output_data = sample["answer"]
        elif "mathqq" in self.dataset_name:
            input_data = sample["en"]
            output_data = sample["answer"]
        elif "mgsmqq" in self.dataset_name:
            input_data = sample[self.language]
            output_data = sample["answer"]
        else:
            input_data = sample["question"]
            output_data = sample["answer"]

        # input_data = sample[self.language]
        # output_data = sample["answer"]

        model_input = DATASET_PROMPTS[self.dataset_name].replace("{input_data}", input_data)
        # if self.dataset_name == "theoremqa":
        #     model_input = model_input.replace("{answer_type}", sample["answer_type"])
        # model_input = input_data              #这里没有写过多的prompt
        input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": model_input}],
                                                       tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_len = len(input_ids[0])

        print(f"********** Input Text (length: {input_len}) **********\n{input_data}\n")
        print(f"********** Input ID **********\n{input_ids}\n")

        return input_data, output_data, model_input, input_ids

    def print_output(self):
        output_scores = self.sample_info["output"]["output_scores"]
        output_seq = self.sample_info["output"]["output_seq"]
        true_output = self.sample_info["output"]["raw_output_data"]
        output_len = self.sample_info["output"]["output_len"]

        
        input_seq = self.tokenizer.decode(output_seq[0][:-output_len])
        output_seq = self.tokenizer.decode(output_seq[0][-output_len:])
        # print(f"********** Model-generated Text ({input_seq}\n")
        print(f"********** Model-generated Text (length: {output_len}) **********\n{output_seq}\n")
        print(f"********** True Output Text **********\n{true_output}\n")

        outputinfo = OutputScoreInfo(output_scores)
        maxprob = outputinfo.compute_maxprob()
        ppl = outputinfo.compute_ppl()
        entropy = outputinfo.compute_entropy()
        print(f"********** Output Info: **********\nmaxprob {maxprob}; perplexity {ppl}; entropy {entropy}\n")

        return input_seq, output_seq, maxprob, ppl, entropy

    def save_output(self, output, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Output', self.model_name,
                               self.dataset_name)

        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            print(f"saved to {filedir}")
            pickle.dump(output, file)

    def save_attentions(self, attentions, i):        
        all_steps = []

        for step in range(len(attentions)):
            step_attn = attentions[step]  # list of layers
            step_layers = []

            for layer_attn in step_attn:
                # layer_attn: [1, num_heads, seq_len, seq_len]
                att = layer_attn.cpu().numpy()
                # 对 num_heads 维平均: axis=1 -> [1, seq_len, seq_len]
                # att_mean = att.mean(axis=1)
                # 去掉 batch 维度 -> [seq_len, seq_len]
                att_mean = att[0]
                step_layers.append(att_mean)

            all_steps.append(step_layers)

        # 这里不要 np.array(all_steps)，直接保存 list
        att_info = {"attentions": all_steps}


        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Attentions', self.model_name,
                               self.dataset_name)

        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            print(f"saved to {filedir}")
            pickle.dump(att_info, file)    

    def print_hidden_states(self):
        hidden_states = self.sample_info["output"]["all_token_hidden_states"]
        output_len = self.sample_info["output"]["output_len"]
        all_hs = []
        for pos in range(output_len):
            layer_hs = []
            for layer in range(len(hidden_states[pos])):

                hs = hidden_states[pos][layer][0][0].cpu().numpy()
                layer_hs.append(hs)
            all_hs.append(layer_hs)

        all_hs = np.array(all_hs)
        hidden_states = all_hs

        print(f"********** Hidden State Size: **********\n{all_hs.shape}\n")

        return hidden_states

    def save_hidden_states(self, hidden_states, i):
        hs = {'hidden_states': hidden_states}
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/HiddenStates', self.model_name,
                               self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(hs, file)


