# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers_multiattn1 as layers

from drqa.Layers import EncoderLayer, DecoderLayer, DecoderLayer_end
import drqa.Constants as Constants
import numpy as np
# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

def normalize_emb_(data):
    print (data.size(), data[:10].norm(2,1))
    norms = data.norm(2,1) + 1e-8
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    print (data.size(), data[:10].norm(2,1))

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            if normalize_emb: normalize_emb_(embedding)
            self.embedding.weight.data = embedding

            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if normalize_emb: normalize_emb_(self.pos_embedding.weight.data)
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            if normalize_emb: normalize_emb_(self.ner_embedding.weight.data)
        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        '''
        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        '''

        self.decoder_start = DecoderLayer_end(doc_hidden_size,
                                              opt['embedding_dim'] // 2,
                                              n_head=4,
                                              d_k=32,
                                              d_v=32,
                                              dropout=0.1)
        self.decoder_end = DecoderLayer_end(doc_hidden_size,
                                            opt['embedding_dim'] // 2,
                                            n_head=4,
                                            d_k=32,
                                            d_v=32,
                                            dropout=0.1)


    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            if self.opt['dropout_emb'] > 0:
                x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            if self.opt['dropout_emb'] > 0:
                x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_ner_emb)
        drnn_input = torch.cat(drnn_input_list, 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        #if self.opt['question_merge'] == 'avg':
        #    q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        #elif self.opt['question_merge'] == 'self_attn':
        #    q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        #question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        enc_slf_attn_pad_mask = get_attn_padding_mask(x1, x1)
        enc_slf_attn_sub_mask = get_attn_subsequent_mask(x1)
        enc_slf_attn_mask = torch.gt(enc_slf_attn_pad_mask + enc_slf_attn_sub_mask, 0)
        enc_dec_attn_pad_mask = get_attn_padding_mask(x1, x2)
        start_scores, *_ = self.decoder_start.forward(dec_input=doc_hiddens,
                                                enc_output=question_hiddens,
                                                slf_attn_mask=enc_slf_attn_mask,
                                                dec_enc_attn_mask=enc_dec_attn_pad_mask
                                                )
        end_scores, *_ = self.decoder_end.forward(dec_input=doc_hiddens,
                                                enc_output=question_hiddens,
                                                slf_attn_mask=enc_slf_attn_mask,
                                                dec_enc_attn_mask=enc_dec_attn_pad_mask

                                                )
        if self.training:
            start_scores = nn.functional.log_softmax(start_scores)
            end_scores = nn.functional.log_softmax(end_scores)

        else:
            start_scores = nn.functional.softmax(start_scores)
            end_scores = nn.functional.softmax(end_scores)

        return start_scores, end_scores
