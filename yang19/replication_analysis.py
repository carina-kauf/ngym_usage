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
    print(len(seeds))
    cnt_dict = {}
    for i in range(1, mx_clusters+1):
        if i not in cnt_dict:
            cnt_dict[i] = np.count_nonzero(nr_clusters == i)
        else:
            cnt_dict[i] += 1

    print(cnt_dict)
    sns.histplot(data=df, x="nr_clusters", bins=np.arange(2,len(cnt_dict)+2)-0.5, edgecolor='white',
                 color=sns.color_palette()[0])
    plt.axvline(x=12, color='green', linestyle='dashed', label='Yang et al. 2019')
    plt.axvline(x=np.mean(nr_clusters), color='black', linestyle='dashed', label=f'Average={round(np.mean(nr_clusters),1)}')
    plt.xticks(range(2, max(nr_clusters)+1))
    plt.yticks(range(0, max(cnt_dict.values())+1))
    plt.xlabel('Nr. of clusters')
    plt.legend()
    plt.title(f'Distribution of number of unit clusters over {len(seeds)} runs \n '
              f'(training iterations = 40000 | bsz = 20)')
    plt.savefig(f'replication_results/cluster_distribution.png',bbox_inches='tight',dpi=200)
    plt.show()

def task_sim_histogram(mode=None):
    # read text file into pandas DataFrame
    df = pd.read_csv("files/nr_clusters.txt", sep="\t", index_col=False)
    seeds = np.array(df['seed'])

    # Step 1: read task similarity into dict to avoid reloading:
    SEED2TASKSIM = {}
    for seed in seeds:
        fname = f'files/seed={seed}_normalizedTV.pkl'
        with open(fname, 'rb') as f:
            norm_task_variance = pickle.load(f)

            if mode == "RSA_corr":
                tasksim = pairwise_distance(norm_task_variance, metric='correlation')
            elif mode == "RSA_cosine":
                tasksim = pairwise_distance(norm_task_variance, metric='cosine')
            elif mode == "cluster_cosine":
                tasksim = cosine_similarity(norm_task_variance)
            elif mode == 'cluster_corr':
                tasksim = pd.DataFrame(norm_task_variance.T)
                tasksim = tasksim.corr('pearson').values
            else:
                raise NotImplementedError
            #When metric='cosine', pairwise_distance(norm_task_variance, metric='cosine') is equivalent to below
            #tasksim = cosine_similarity(norm_task_variance)
            # return upper triangular matrix values, excluding diagonal. outputs vector created by concatenating numbers row-wise
            # upper = tasksim[np.triu_indices(np.shape(tasksim)[0], k=1)]
            # upper = np.ones_like(upper) - upper
        SEED2TASKSIM[seed] = tasksim

    sim_scores = []
    if 'RSA' in mode:
        # Step 2: Get pairwise cosine similarity scores for task similarities derived via different seeds
        for seed1 in seeds:
            for seed2 in seeds:
                if seed1 != seed2:
                    if mode == 'RSA_corr':
                        score = pearsonr(SEED2TASKSIM[seed1], SEED2TASKSIM[seed2])[0]
                    elif mode == 'RSA_cosine':
                        score = 1 - distance.cosine(SEED2TASKSIM[seed1], SEED2TASKSIM[seed2])
                    else:
                        raise NotImplementedError
                    sim_scores.append(score)

        # figure settings
        if mode == 'RSA_cosine':
            xlabel = 'Cosine similarity'
            title = f'Distribution of pairwise cosine similarity scores \n task-similarity ({len(seeds)} runs |' \
                    f' {sum([len(seeds) - i for i in range(1,len(seeds))])} comparisons)'
        if mode == 'RSA_corr':
            xlabel = 'Pearson r'
            title = f'Distribution of pairwise Pearson correlation scores \n task-similarity ({len(seeds)} runs |' \
                    f' {sum([len(seeds) - i for i in range(1,len(seeds))])} comparisons)'
        savename = f'replication_results/task_similarity_distribution_{mode}.png'

    elif 'cluster' in mode:
        task_sim_clusters = []
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
            task_sim_clusters.append(n_cluster)

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
        if 'cosine' in mode:
            toadd = 'cosine sim'
        else:
            toadd  = 'Pearson r'
        title = f'Distribution of pairwise AMI scores for task-similarity clusterings \n' \
                    f'{toadd} | {len(seeds)} runs | {sum([len(seeds) - i for i in range(1,len(seeds))])} comparisons'
        savename = f'replication_results/task_similarity_distribution_{mode}.png'

    else:
        raise NotImplementedError

    if "RSA" in mode:
        color=sns.color_palette()[2]
    else:
        color=sns.color_palette()[1]
    df = pd.DataFrame(sim_scores, columns=['sim_scores'])
    sns.histplot(data=df, x="sim_scores", edgecolor='white', bins=50, color=color)
    plt.axvline(x=np.mean(sim_scores), color='black', linestyle='dashed', label=f'Average={round(np.mean(sim_scores),3)}')
    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(savename,bbox_inches='tight',dpi=200)
    plt.show()

    if 'cluster' in mode:
        df = pd.DataFrame(task_sim_clusters, columns=['task_sim_clusters'])
        mx_clusters = np.max(task_sim_clusters)
        cnt_dict = {}
        for i in range(1, mx_clusters + 1):
            if i not in cnt_dict:
                cnt_dict[i] = sum([1 for x in task_sim_clusters if x == i])
            else:
                cnt_dict[i] += 1

        print(cnt_dict)
        sns.histplot(data=df, x="task_sim_clusters", bins=np.arange(2, len(cnt_dict) + 2) - 0.5, edgecolor='white',
                     color=sns.color_palette()[1])
        plt.axvline(x=np.mean(task_sim_clusters), color='black', linestyle='dashed',
                    label=f'Average={round(np.mean(task_sim_clusters), 1)}')
        plt.xticks(range(2, max(task_sim_clusters) + 1))
        plt.yticks(range(0, max(cnt_dict.values()) + 1))
        plt.xlabel('Nr. of clusters')
        plt.legend()
        plt.title(f'Distribution of number of task_similarity clusters over {len(seeds)} runs \n '
                  f'({toadd} | training iterations = 40000 | bsz = 20)')
        plt.savefig(f'replication_results/cluster_distribution_task_similarity_{mode}.png',bbox_inches='tight',dpi=200)
        plt.show()


def main():
    # create save directory
    path = Path('.') / 'replication_results'
    os.makedirs(path, exist_ok=True)

    nr_of_cluster_distribution()
    modes = ['RSA_corr', 'RSA_cosine', 'cluster_cosine', 'cluster_corr']
    for mode in modes:
        task_sim_histogram(mode=mode)

if __name__ == '__main__':
    main()

