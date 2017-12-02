# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from drqa import layers_transformer1 as layers
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
from drqa.Modules import BottleLinear as Linear

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


__author__ = "Yu-Hsiang Huang"

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

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

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers=6, n_head=8, d_k=64, d_v=64,
            doc_hidden_size=512, conv_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()
        '''
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = doc_hidden_size

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        '''
        self.layer_stack = nn.ModuleList([
            EncoderLayer(doc_hidden_size, conv_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x1, enc_slf_attn_mask,return_attns=False):
        # Word embedding look up
        #enc_input = self.src_word_emb(src_seq)

        # Position Encoding addition
        #enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = x1
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_layers=6, n_head=8, d_k=64, d_v=64,
            doc_hidden_size=512, conv_inner_size=1024, dropout=0.1):

        super(Decoder, self).__init__()
        '''
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        '''

        self.layer_stack = nn.ModuleList([
            DecoderLayer(doc_hidden_size, conv_inner_size, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x2, x1, enc_output,
                dec_slf_attn_pad_mask,
                dec_slf_attn_sub_mask,
                dec_slf_attn_mask,
                dec_enc_attn_pad_mask,
                return_attns=False):
        # Word embedding look up
        #dec_input = self.tgt_word_emb(x2)

        # Position Encoding addition
        #dec_input += self.position_enc(tgt_pos)

        # Decode

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = x2
        for dec_layer in self.layer_stack:
            dec_output, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_enc_attns
        else:
            return dec_output,




class DocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False,embedding_order=True):
        super(DocReader, self).__init__()
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
        '''
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if normalize_emb: normalize_emb_(self.pos_embedding.weight.data)
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            if normalize_emb: normalize_emb_(self.ner_embedding.weight.data)
        '''
        if embedding_order:
            self.embedding_order = nn.Embedding(num_embeddings=1000, embedding_dim=opt['embedding_dim'],
                                                padding_idx=padding_idx)
            self.embedding_order.weight.data = layers.position_encoding_init(n_position=1000, d_pos_vec=opt['embedding_dim'])
            self.embedding_order.weight.requires_grad = False
        '''
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

        doc_hidden_size = doc_input_size
        '''
        doc_hidden_size = opt['embedding_dim']
        n_head=8
        n_layers=3
        d_k=32
        d_v=32
        self.encoder = Encoder(
            n_layers=n_layers, n_head=n_head,
            doc_hidden_size=doc_hidden_size,
            conv_inner_hid=doc_hidden_size //2, dropout=0.1, d_k=d_k, d_v=d_v)
        self.decoder1 = Decoder(
            n_layers=n_layers, n_head=n_head,
            doc_hidden_size=opt['embedding_dim'],
            conv_inner_size=opt['embedding_dim']//2, dropout=0.1, d_k=d_k, d_v=d_v)

        self.decoder_start = DecoderLayer_end(doc_hidden_size,
                                              opt['embedding_dim'] // 2,
                                              n_head=n_head,
                                              d_k=d_k,
                                              d_v=d_v,
                                              dropout=0.1)
        self.decoder_end = DecoderLayer_end(doc_hidden_size,
                                              opt['embedding_dim'] // 2,
                                              n_head=n_head,
                                              d_k=d_k,
                                              d_v=d_v,
                                              dropout=0.1)
        '''
        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
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
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )
        self.num_object=opt['num_objects']
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
        self.relationNet = layers.RelationNetwork(hidden_size=2 * doc_hidden_size, output_size=doc_hidden_size)
        # doc_attention for maxpooling
        #self.doc_attn = layers.SeqAttnMatch(input_size=doc_hidden_size)
        self.num_head = opt['num_head']
        self.doc_attn = MultiHeadAttention(n_head=self.num_head,
                                           d_model=question_hidden_size,
                                           d_k=32,
                                           d_v=32,
                                           dropout=opt['dropout_emb'])
        self.doc_pos_ffn = PositionwiseFeedForward(d_hid=question_hidden_size,d_inner_hid=2*question_hidden_size,dropout=opt['dropout_emb'])
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



    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask,x1_order,x2_order):
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
        x1_emb += self.embedding_order(x1_order)
        x2_emb = self.embedding(x2)
        x2_emb += self.embedding_order(x2_order)

        '''
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
        '''

        '''
        enc_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation

        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            enc_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            if self.opt['dropout_emb'] > 0:
                x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            enc_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            if self.opt['dropout_emb'] > 0:
                x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            enc_input_list.append(x1_ner_emb)
        enc_input = torch.cat(enc_input_list, 2)
        '''
        enc_input = x2_emb
        enc_slf_attn_mask = get_attn_padding_mask(x2, x2)
        #print('enc_slf_attn_mask:',enc_slf_attn_mask.size())
        enc_output, *_ = self.encoder.forward(enc_input,enc_slf_attn_mask)
        #print('enc_output:',enc_output.size())
        dec_slf_attn_pad_mask = get_attn_padding_mask(x1, x1)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(x1)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(x1, x2)
        dec_output, *_ = self.decoder1.forward(x2= x1_emb, x1= x2_emb, enc_output= enc_output,
                                               dec_slf_attn_pad_mask=dec_slf_attn_pad_mask,
                                               dec_slf_attn_sub_mask=dec_slf_attn_sub_mask,
                                               dec_slf_attn_mask=dec_slf_attn_mask,
                                               dec_enc_attn_pad_mask=dec_enc_attn_pad_mask
                                               )
        #print('dec_output:',dec_output.size())

        #dec_slf_attn_pad_mask = get_attn_padding_mask(x1, x1)
        #dec_slf_attn_sub_mask = get_attn_subsequent_mask(x1)
        #dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        #dec_enc_attn_pad_mask = get_attn_padding_mask(x1, x2)
        start_scores, *_ = self.decoder_start.forward(dec_input=dec_output,
                                                enc_output=enc_output,
                                                slf_attn_mask=dec_slf_attn_mask,
                                                dec_enc_attn_mask=dec_enc_attn_pad_mask
                                                )
        end_scores, *_ = self.decoder_end.forward(dec_input=dec_output,
                                                enc_output=enc_output,
                                                slf_attn_mask=dec_slf_attn_mask,
                                                dec_enc_attn_mask=dec_enc_attn_pad_mask

                                                )
        if self.training:
            start_scores = nn.functional.log_softmax(start_scores)
            end_scores = nn.functional.log_softmax(end_scores)

        else:
            start_scores = nn.functional.softmax(start_scores)
            end_scores = nn.functional.softmax(end_scores)

        #print('start_scores:',start_scores)
        #print('end_scores:', end_scores)

        return start_scores, end_scores


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.embedding_order.parameters()))
        dec_freezed_param_ids = set(map(id, self.embedding_order.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)