import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.spatial import distance

def nr_of_cluster_distribution():
    # read text file into pandas DataFrame
    df = pd.read_csv("files/nr_clusters.txt", sep="\t", index_col=False)
    seeds = np.array(df['seed'])

    nr_clusters = np.array(df['nr_clusters'])
    mx_clusters = np.max(nr_clusters)
    print(seeds)
    cnt_dict = {}
    for i in range(1, mx_clusters+1):
        if i not in cnt_dict:
            cnt_dict[i] = np.count_nonzero(nr_clusters == i)
        else:
            cnt_dict[i] += 1

    print(cnt_dict)
    sns.histplot(data=df, x="nr_clusters")
    plt.xticks(range(0, max(nr_clusters)+1))
    plt.yticks(range(0, max(cnt_dict.values())+1))
    plt.xlabel('Nr. of clusters')
    plt.title(f'Distribution of clusters over {len(seeds)} training runs')
    plt.savefig(f'replication_results/cluster_distribution.png')
    plt.show()

def task_sim_histogram(mode='RSA'):
    # read text file into pandas DataFrame
    df = pd.read_csv("files/nr_clusters.txt", sep="\t", index_col=False)
    seeds = np.array(df['seed'])

    #read task stimilarity into dict to avoid reloading:
    SEED2TASKSIM = {}
    for seed in seeds:
        fname = f'files/seed={seed}_task_similarity.pkl'
        with open(fname, 'rb') as f:
            tasksim = pickle.load(f)
            # return upper triangular matrix values, excluding diagonal
            upper = tasksim[np.triu_indices(np.shape(tasksim)[0], k=1)]
        SEED2TASKSIM[seed] = upper

    sim_scores = []
    if mode == 'RSA':
        for seed1 in seeds[:-1]:
            for seed2 in seeds[1:]:
                score = 1 - distance.cosine(SEED2TASKSIM[seed1], SEED2TASKSIM[seed2])
                sim_scores.append(score)

    elif mode == 'clustering':
        raise NotImplementedError

    else:
        raise NotImplementedError

    df = pd.DataFrame(sim_scores, columns=['sim_scores'])
    sns.histplot(data=df, x="sim_scores")
    plt.xlabel('Similarity scores')
    plt.title(f'Pairwise similarity of task similarity for {len(seeds)} runs')
    plt.savefig(f'replication_results/task_similarity_distribution.png')
    plt.show()


def main():
    # create save directory
    path = Path('.') / 'replication_results'
    os.makedirs(path, exist_ok=True)

    nr_of_cluster_distribution()
    task_sim_histogram(mode='RSA')

if __name__ == '__main__':
    main()

