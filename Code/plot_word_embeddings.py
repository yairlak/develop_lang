import os, argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data import *
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Plot word embeddings')
parser.add_argument('-f1', '--filename-wordvecs', type=str, default='../Data/word_embeddings/cc.en.300.vec')
parser.add_argument('-f2', '--filename-develop', type=str, default='../Data/Types Per Min EA Sample.csv')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prob', type=int, default=1, help='Plot only fraction of words')
args=parser.parse_args()

np.random.seed(args.seed)

# LOAD DATA
subject_ids, develop_data = load_development_data(args.filename_develop)
vocab, wordvecs, n, d = load_vectors(args.filename_wordvecs, vocab_size = 10000)

# PCA
pca = PCA(n_components=2)
X_r = pca.fit(wordvecs).transform(wordvecs)
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

# PLOT
fig, ax = plt.subplots(figsize=(40, 30))
lw = 4


for i, word in tqdm(enumerate(vocab)):
    if np.random.rand() < args.prob:
        plt.text(X_r[i, 0], X_r[i, 1], word, alpha=0.5)

for i, subject_id in enumerate(subject_ids):
    color = [(subject_id - 513) / 60, (570 - subject_id) / 50, 1 / (subject_id - 512)]
    words_subject_heard = list(develop_data[subject_id]["words"].columns[np.sum(develop_data[subject_id]["words"].values[:, :-4], axis=0)>0])
    for w, word_heard in enumerate(words_subject_heard):
        try:
            IX = vocab.index(word_heard)
            dx = (np.random.rand()*2-1)*0.01
            dy = (np.random.rand()*2-1)*0.01
            if w == 0:
                ax.scatter(X_r[IX, 0]+dx, X_r[IX, 1]+dy, edgecolor=color, s=10, lw=lw, label=str(subject_id))
            else:
                ax.scatter(X_r[IX, 0] + dx, X_r[IX, 1] + dy, edgecolor=color, s=10, lw=lw)
        except:
            print('Subject %s, word heard not in vocabulary: %s' % (subject_id, word_heard))


plt.legend()
ax.axis('off')
ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])

fig.savefig('../Figures/word_embeddings_zoomin2.png')
