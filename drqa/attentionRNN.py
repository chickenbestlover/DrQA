# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import layers_RN_q as layers

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

import cuda_functional as MF
import numpy as np


class AttentionRNN(nn.Module):
    def __init__(self, opt, doc_input_size):
        super(AttentionRNN, self).__init__()
        self.doc_rnns = nn.ModuleList()
        self.question_rnns = nn.ModuleList()

        self.doc_attns = nn.ModuleList()
        self.question_self_attns = nn.ModuleList()
        self.num_layers = opt['doc_layers']
        for i in range(self.num_layers):

            doc_input_size = doc_input_size if i == 0 else 2 * opt['hidden_size']
            question_input_size = opt['embedding_dim'] if i == 0 else 2 * opt['hidden_size']

            self.doc_rnns.append(layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            ))
            self.question_rnns.append(layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            ))

            self.doc_attns.append(layers.BilinearSeqAttn_norm(2 * opt['hidden_size'], 2 * opt['hidden_size']))
            self.question_self_attns.append(layers.LinearSeqAttn(2 * opt['hidden_size']))

    def forward(self, x1, x1_mask, x2, x2_mask):

        # Encode all layers

        for i in range(self.num_layers):
            # Forward
            #print('doc_rnn_input:',doc_rnn_input.size())
            x1 = self.doc_rnns[i](x1,x1_mask)
            x2 = self.question_rnns[i](x2,x2_mask)

            q_merge_weights = self.question_self_attns[i](x2, x2_mask)
            question_hidden = layers.weighted_avg(x2, q_merge_weights)
            doc_attn_scores = self.doc_attns[i](x1,question_hidden,x1_mask)
            x1 = doc_attn_scores.unsqueeze(2).expand_as(x1) * x1

        return x1.contiguous(),question_hidden.contiguous()

