import os
import sentencepiece as spm
from datasets import load_dataset
import numpy as np

ds = load_dataset("cfilt/iitb-english-hindi", split="train")

save_path = "./sentencepiece_tok/"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

with open(os.path.join(save_path, "train_corpus.txt"), "w", encoding="utf-8") as f:
    for item in ds:
        en = item['translation']["en"]
        hi = item['translation']["hi"]
        f.write(en.strip() + "\n")
        f.write(hi.strip() + "\n")


spm.SentencePieceTrainer.train(
    input=os.path.join(save_path, "train_corpus.txt"),
    model_prefix=os.path.join(save_path, "en_hi_shared"),
    vocab_size=32000,           
    model_type="bpe",
    character_coverage=0.9995,  # critical for Devanagari coverage
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    shuffle_input_sentence=True,
    num_threads=os.cpu_count()
)
