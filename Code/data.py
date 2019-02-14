import io
import pandas as pd
import numpy as np

def load_development_data(fname):
    data = pd.read_csv(fname, header=0)
    subject_ids = set(data['id'])
    d = {}
    for s in subject_ids:
        d_subj = {}
        d_subj["words"] = data.loc[data['id'] == s].iloc[:, 4::]
        d_subj["min"] = data.loc[data['id'] == s].iloc[:, 1]
        d_subj["female"] = data.loc[data['id'] == s].iloc[:, 2]
        d_subj["activity"] = data.loc[data['id'] == s].iloc[:, 3]
        d[s] = d_subj

    return subject_ids, d

def load_vectors(fname, vocab_size = 50000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    words = []; data = []; i = 0
    for line in fin:
        if i < vocab_size:
            tokens = line.rstrip().split(' ')
            words.append(tokens[0])
            data.append(list(map(float, tokens[1:])))
            i += 1
    return words, np.vstack(data), n, d