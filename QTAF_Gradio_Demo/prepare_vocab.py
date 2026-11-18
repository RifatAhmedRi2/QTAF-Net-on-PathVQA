"""Reconstruct mappings exactly as in notebook 01 from the cleaned training split."""
import json,re,hashlib
from collections import Counter
from pathlib import Path
from datasets import load_dataset
import numpy as np

def image_hash(image):
    image=image.convert('RGB')
    h=hashlib.sha256()
    h.update(str(image.size).encode())
    h.update(image.tobytes())
    return h.hexdigest()

def tokenize(text):
    return re.findall(r'\w+|[^\w\s]',str(text).lower().strip())

def main():
    dataset=load_dataset('flaviagiammarino/path-vqa')
    # See README: verify exact training cleaning and ordering against the original notebook.
    validation_hashes={image_hash(im) for im in dataset['validation']['image']}
    clean_train=dataset['train'].filter(lambda row: image_hash(row['image']) not in validation_hashes)
    word_counts=Counter()
    for question in clean_train['question']:word_counts.update(tokenize(question))
    word2idx={'<PAD>':0,'<UNK>':1}
    for word in word_counts:word2idx[word]=len(word2idx)
    open_answers=[str(a).lower().strip() for a in clean_train['answer'] if str(a).lower().strip() not in ('yes','no')]
    answer2idx={a:i for i,(a,_) in enumerate(Counter(open_answers).most_common(1000))}
    assert len(clean_train)==19648, f'Unexpected clean train size: {len(clean_train)}'
    assert len(word2idx)==3972, f'Unexpected vocabulary size: {len(word2idx)}'
    assert len(answer2idx)==1000
    output=Path(__file__).parent/'vocab.json'
    output.write_text(json.dumps({'word2idx':word2idx,'open_answer2idx':answer2idx},ensure_ascii=False))
    print('Saved',output)
if __name__=='__main__':main()
