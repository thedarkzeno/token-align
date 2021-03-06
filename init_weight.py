#init_weight.py
# based on amazon's ramen
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import torch
import argparse
import torch.nn as nn
from transformers import  AutoTokenizer, AutoModelForMaskedLM
import numpy as np
parser = argparse.ArgumentParser('generate target embeddings from alignments')
parser.add_argument('--tgt_tokenizer', default='', help='path to target tokenizer')
parser.add_argument('--src_tokenizer', default='', help='path to source tokenizer')
parser.add_argument('--src_model', default='pytorch.bin', help='source pre-trained file')
parser.add_argument('--prob', default='', help='word translation probability')
parser.add_argument('--tgt_path', default='', help='save the target model')
params = parser.parse_args()
print(params)

if 'roberta' in params.src_model:
    MAP = {
        'word_embeddings': 'roberta.embeddings.word_embeddings.weight',
        'output_weight': 'lm_head.decoder.weight',
        'output_bias': 'lm_head.bias'
    }

elif 'bert' in params.src_model:
    MAP = {
        'word_embeddings': 'bert.embeddings.word_embeddings.weight',
        'output_weight': 'cls.predictions.decoder.weight',
        'output_bias': 'cls.predictions.bias'
    }



def guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=None):
    emb_dim = src_embs.size(1)
    num_tgt = len(tgt_tokenizer.get_vocab())

    # init with zero
    tgt_embs = src_embs.new_empty(num_tgt, emb_dim)
    tgt_bias = src_bias.new_zeros(num_tgt)
    nn.init.normal_(tgt_embs, mean=0, std=emb_dim ** -0.5)

    # initialize randomly
    if prob is None:
        print('| INITIALIZE EMBEDDINGS AND BIAS RANDOMLY')
        return tgt_embs, tgt_bias


    num_src_per_tgt = np.array([len(x) for x in prob.values()]).mean()
    print(f'| # aligned src / tgt: {num_src_per_tgt:.5}')

    for t, ws in prob.items():
        if not tgt_tokenizer.convert_tokens_to_ids(t): continue

        px, ix = [], []
        for e, p in ws.items():
            # get index of the source word e
            j = src_tokenizer.convert_tokens_to_ids(e)
            ix.append(j)
            px.append(p)
        px = torch.tensor(px).to(src_embs.device)
        # get index of target word t
        ti = tgt_tokenizer.convert_tokens_to_ids(t)
        tgt_embs[ti] = px @ src_embs[ix]
        tgt_bias[ti] = px.dot(src_bias[ix])

    return tgt_embs, tgt_bias


def init_tgt(params):
    """
    Initialize the parameters of the target model
    """
    prob = None
    if params.prob:
        print(' | load word translation probs!')
        prob = torch.load(params.prob)

    print(f'| load English pre-trained model: {params.src_model}')
    model = AutoModelForMaskedLM.from_pretrained(params.src_model)
    config = model.config
    model.save_pretrained(params.src_model+"_")
    model = torch.load(params.src_model+"_/pytorch_model.bin")
    src_tokenizer = AutoTokenizer.from_pretrained(params.src_tokenizer)

    # get English word-embeddings and bias
    src_embs = model[MAP['word_embeddings']]
    src_bias = model[MAP['output_bias']]

    # initialize target tokenizer, we always use BertWordPieceTokenizer for the target language
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        params.tgt_tokenizer
    )

    tgt_embs, tgt_bias = guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=prob)

    # checksum for debugging purpose
    print(' checksum src | embeddings {:.5f} - bias {:.5f}'.format(
        src_embs.norm().item(), src_bias.norm().item()))
    model[MAP['word_embeddings']] = tgt_embs
    model[MAP['output_bias']] = tgt_bias
    model[MAP['output_weight']] = model[MAP['word_embeddings']]
    print(' checksum tgt | embeddings {:.5f} - bias {:.5f}'.format(
        model[MAP['word_embeddings']].norm().item(),
        model[MAP['output_bias']].norm().item()))

    # save the model
    # model.save_pretrained(params.tgt_model)
    torch.save(model, params.tgt_path+"/pytorch_model.bin")
    config.vocab_size = len(tgt_tokenizer.get_vocab())
    config.save_pretrained(params.tgt_path)
    tgt_tokenizer.save_pretrained(params.tgt_path)

    #fixing possible mismatch
    model = AutoModelForMaskedLM.from_pretrained(params.tgt_path, ignore_mismatched_sizes=True)
    model.save_pretrained(params.tgt_path)



if __name__ == '__main__':
    init_tgt(params)
