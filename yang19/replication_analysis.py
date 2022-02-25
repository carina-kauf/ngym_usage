import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist as pairwise_distance

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

def task_sim_histogram(mode=None):
    # read text file into pandas DataFrame
    df = pd.read_csv("files/nr_clusters.txt", sep="\t", index_col=False)
    seeds = np.array(df['seed'])

    # Step 1: read task stimilarity into dict to avoid reloading:
    SEED2TASKSIM = {}
    SEED2TASKSIM_upper = {}
    for seed in seeds:
        fname = f'files/seed={seed}_normalizedTV.pkl'
        with open(fname, 'rb') as f:
            norm_task_variance = pickle.load(f)
            tasksim = pairwise_distance(norm_task_variance, metric='correlation')
            #When metric='cosine', this is equivalent to below
            #tasksim = cosine_similarity(norm_task_variance)
            # return upper triangular matrix values, excluding diagonal. outputs vector created by concatenating numbers row-wise
            upper = tasksim[np.triu_indices(np.shape(tasksim)[0], k=1)]
        SEED2TASKSIM[seed] = tasksim
        SEED2TASKSIM_upper[seed] = upper

    sim_scores = []
    if 'RSA' in mode:
        # Step 2: Get pairwise cosine similarity scores for task similarities derived via different seeds
        for seed1 in seeds:
            for seed2 in seeds:
                if seed1 != seed2:
                    if mode == 'RSA_corr':
                        score = pearsonr(SEED2TASKSIM_upper[seed1], SEED2TASKSIM_upper[seed2])[0]
                    elif mode == 'RSA_cosine':
                        score = 1 - distance.cosine(SEED2TASKSIM_upper[seed1], SEED2TASKSIM_upper[seed2])
                    else:
                        raise NotImplementedError
                    sim_scores.append(score)

        # figure settings
        if mode == 'RSA_cosine':
            xlabel = 'Cosine similarity'
            title = f'Distribution of pairwise cosine similarity \n for task-similarity for {len(seeds)} runs'
        if mode == 'RSA_corr':
            xlabel = 'Pearson r'
            title = f'Distribution of pairwise correlation \n for task-similarity for {len(seeds)} runs'
        savename = f'replication_results/task_similarity_distribution_{mode}.png'

    elif mode == 'cluster':
        #Step 2: Compute clusterings on similarity matrix
        CLUSTERINGS = {}
        for seed in seeds:
            silhouette_scores = list()
            n_clusters = np.arange(2, 20)
            for n in n_clusters:
                cluster_model = AgglomerativeClustering(n_clusters=n)
                labels = cluster_model.fit_predict(SEED2TASKSIM[seed])
                silhouette_scores.append(silhouette_score(SEED2TASKSIM[seed], labels))
            n_cluster = n_clusters[np.argmax(silhouette_scores)]

            cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
            labels = cluster_model.fit_predict(SEED2TASKSIM[seed])
            CLUSTERINGS[seed] = labels

        #Step 3: Compare clusterings
        for seed1 in seeds:
            for seed2 in seeds:
                if seed1 != seed2:
                    score = metrics.adjusted_mutual_info_score(CLUSTERINGS[seed1], CLUSTERINGS[seed2])
                    sim_scores.append(score)

        #figure settings
        xlabel = 'Adjusted Mutual Information (AMI)'
        title = f'Distribution of pairwise AMI scores \n for task-similarity clusterings for {len(seeds)} runs'
        savename = f'replication_results/task_similarity_distribution_{mode}.png'

    else:
        raise NotImplementedError

    df = pd.DataFrame(sim_scores, columns=['sim_scores'])
    sns.histplot(data=df, x="sim_scores")

    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(savename,bbox_inches='tight',dpi=280)
    plt.show()


def main():
    # create save directory
    path = Path('.') / 'replication_results'
    os.makedirs(path, exist_ok=True)

    nr_of_cluster_distribution()
    task_sim_histogram(mode='RSA_corr')
    task_sim_histogram(mode='RSA_cosine')
    task_sim_histogram(mode='cluster')

if __name__ == '__main__':
    main()

