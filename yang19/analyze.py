"""Analyze."""

import os
import matplotlib
import matplotlib.pyplot as plt

from models import get_performance
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from make_environments import set_seed

from os.path import exists

def print_performance(model, env, tasks, device):
    # Get performance
    for i in range(20):
        env.set_i(i)
        perf = get_performance(model, env, device, num_trial=200)
        print('Average performance {:0.2f} for task {:s}'.format(perf, tasks[i]))


def get_activity(model, env, device, num_trial=1000):
    """Get activity of equal-length trials"""

    trial_list = list()
    activity_list = list()
    for i in range(num_trial):
        env.new_trial()
        ob = env.ob
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, activity = model(inputs)
        activity = activity.detach().numpy()  # (seq_len, 1, hidden_size)
        trial_list.append(env.trial)
        activity_list.append(activity)  # list of size 500

    activity = np.concatenate(activity_list, axis=1)  # (seq_len, 500, hidden_size)
    return activity, trial_list


def get_normalized_tv(args, env, tasks, model, device):
    task_variance_list = list()
    for i in range(20):
        env.set_i(i)
        # print(env.spec.id)
        assert env.spec.id == tasks[i]
        activity, trial_list = get_activity(model, env, device, num_trial=500)
        # Compute task variance
        task_variance = np.var(activity, axis=1).mean(axis=0)  # (256,)
        task_variance_list.append(task_variance)  # task variance list is list of 20 arrays of size (256,)
    task_variance = np.array(task_variance_list)  # (n_task, n_units)  (20, 256)

    # First only get active units. Total variance across tasks larger than 1e-6
    thres = 1e-6
    task_variance = task_variance[:,
                    task_variance.sum(axis=0) > thres]  # e.g. size (20, 255) > one unit not above threshold

    # Normalize by the total variance across tasks
    norm_task_variance = task_variance / np.max(task_variance, axis=0)

    fname = f'files/seed={args.seed}_normalizedTV.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(norm_task_variance, fout)
    return norm_task_variance

def figure_settings():
    figsize = (3.5,2.5)
    rect = [0.25, 0.2, 0.6, 0.7]
    rect_color = [0.25, 0.15, 0.6, 0.05]
    rect_cb = [0.87, 0.2, 0.03, 0.7]
    fs = 6
    labelpad = 13
    return figsize, rect, rect_color, rect_cb, fs, labelpad

def get_cluster_plot(args, tasks, norm_task_variance):
    X = norm_task_variance.T
    silhouette_scores = list()
    n_clusters = np.arange(2, 20)
    for n in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        labels = cluster_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'o-')
    plt.xticks(range(2, 20))
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.savefig(os.path.join('figures', f'seed={args.seed}_silhouette_score.png'), bbox_inches='tight', dpi=280)
    plt.show()

    n_cluster = n_clusters[np.argmax(silhouette_scores)]
    ####Write number of clusters to file
    filename = 'files/nr_clusters.txt'
    if exists(filename):
        with open(filename, 'a') as file:
            file.write(f'{args.seed}\t{n_cluster}\n')
    else:
        with open(filename, 'w') as file:
            file.write('seed\tnr_clusters\n')
            file.write(f'{args.seed}\t{n_cluster}\n')


    cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
    labels = cluster_model.fit_predict(X) #0 to n_clusters-1

    # For each cluster set, get index of task with the highest cumulative task variance for the units that make up the cluster
    label_prefs = [np.argmax(norm_task_variance[:, labels==l].sum(axis=1)) for l in set(labels)]

    #sort clusters from lowest to highest cumulative task variance (for preferred task)
    ind_label_sort = np.argsort(label_prefs)
    label_prefs = np.array(label_prefs)[ind_label_sort]
    # Relabel
    labels2 = np.zeros_like(labels)
    for i, ind in enumerate(ind_label_sort):
        labels2[labels==ind] = i
    labels = labels2

    # Sort neurons by labels
    ind_sort = np.argsort(labels)
    labels = labels[ind_sort]
    norm_task_variance = norm_task_variance[:, ind_sort]

    # Plot Normalized Variance
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    tick_names = [task[len('yang19.'):-len('-v0')] for task in tasks]

    vmin, vmax = 0, 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    im = ax.imshow(norm_task_variance, cmap='magma',
                   aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks([])
    plt.title('Units', fontsize=7, y=0.97)
    plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
    ax.tick_params('both', length=0)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
    cb.outline.set_linewidth(0.5)
    clabel = 'Normalized Task Variance'

    cb.set_label(clabel, fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)

    # Plot color bars indicating clustering
    cmap = matplotlib.cm.get_cmap('tab10')
    ax = fig.add_axes(rect_color)
    for il, l in enumerate(np.unique(labels)):
        color = cmap(il % 10)
        ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
        ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                color=color)
        ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                ha='center', va='top', color=color)
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([-1, 1])
    ax.axis('off')

    plt.savefig(os.path.join('figures',f'seed={args.seed}_clusterplot.png'),bbox_inches='tight',dpi=280)
    fig.show()

def plot_task_similarity(args, norm_task_variance, tasks):
    similarity = cosine_similarity(norm_task_variance)  # TODO: check

    fname = f'files/seed={args.seed}_task_similarity.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(similarity, fout)

    print(np.shape(norm_task_variance), np.shape(similarity))

    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)

    tick_names = [task[len('yang19.'):-len('-v0')] for task in tasks]
    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, va='top', fontsize=fs)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plt.savefig(f'figures/seed={args.seed}_task_similarity.png')
    plt.show()

def plot_feature_similarity(args, norm_task_variance):
    X = norm_task_variance.T
    similarity = cosine_similarity(X)  # TODO: check
    print(np.shape(X), np.shape(similarity))
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig(f'figures/seed={args.seed}_feature_similarity.png')
    plt.show()

def main(args, model, env, tasks, device):
    set_seed(args.seed, args.cuda)
    print('Performance')
    print_performance(model, env, tasks, device)
    print("Computing task variance")
    norm_task_variance = get_normalized_tv(args, env, tasks, model, device)
    print("Plotting task variance")
    get_cluster_plot(args, tasks, norm_task_variance)
    plot_task_similarity(args, norm_task_variance, tasks)
    plot_feature_similarity(args, norm_task_variance)

if __name__ == '__main__':
    main(args, model, env, tasks, device)


