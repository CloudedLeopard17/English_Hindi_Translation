# English-Hindi Neural Machine Translation

A Transformer-based sequence-to-sequence model trained from scratch
for English to Hindi translation.

## Overview
- Architecture  : Transformer (encoder-decoder)
- Dataset       : IITB English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)
- Tokenizer     : SentencePiece BPE — shared vocabulary (32k tokens)
- Training      : From scratch with fixed LR for 10 epochs. 20 Epochs for DDP. 
- Evaluation    : BLEU score

## Features
- Custom BPE tokenizer trained on 1.49M parallel sentence pairs
- Label smoothing and gradient clipping for stable training
- Pytorch DDP training for multi-GPU setup.

## Results
| Metric | Score |
|--------|-------|
| BLEU   | .1718  |

## Dataset
[CFILT/IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi) — 1.49M high-quality
sentence pairs covering news, government, and literary domains.
