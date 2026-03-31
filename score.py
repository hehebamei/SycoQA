import os
import sys
import time

import scipy.spatial
from scipy.stats import entropy
import math
import json

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class OutputScoreInfo:
    def __init__(self, output_scores):
        self.output_scores = output_scores
        self.all_token_re = []
        self.all_token_max_re = []
        for token in range(len(self.output_scores)):
            re = self.output_scores[token][0].tolist()
            re = F.softmax(torch.tensor(re).to(device), 0).cpu().tolist()
            self.all_token_re.append(re)
            self.all_token_max_re.append(max(re))

    def compute_maxprob(self):
        seq_prob_list = self.all_token_max_re
        max_prob = np.mean(seq_prob_list)
        return max_prob

    def compute_ppl(self):
        seq_ppl_list = [math.log(max_re) for max_re in self.all_token_max_re]
        ppl = -np.mean(seq_ppl_list)
        return ppl

    def compute_entropy(self):
        seq_entropy_list = [entropy(re, base=2) for re in self.all_token_re]
        seq_entropy = np.mean(seq_entropy_list)
        return seq_entropy




