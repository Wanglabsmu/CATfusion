import os
import sys
import numpy as np
import pandas as pd
import argparse

# Visualization
import matplotlib.pyplot as plt  # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits  # for MNIST data
# for splitting data into train and test samples
from sklearn.model_selection import train_test_split

# UMAP dimensionality reduction
from umap import UMAP

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump, load


def plot_umap(features, labels, output_dirpath):
    # Configure UMAP hyperparameters
    reducer = UMAP(n_neighbors=100,  # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                   # default 2, The dimension of the space to embed into.
                   n_components=2,
                   # default 'euclidean', The metric to use to compute distances in high dimensional space.
                   metric='euclidean',
                   # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                   n_epochs=1000,
                   # default 1.0, The initial learning rate for the embedding optimization.
                   learning_rate=1.0,
                   # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                   init='spectral',
                   # default 0.1, The effective minimum distance between embedded points.
                   min_dist=0.1,
                   # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                   spread=1.0,
                   low_memory=False,  # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                   # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                   set_op_mix_ratio=1.0,
                   # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                   local_connectivity=1,
                   # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                   repulsion_strength=1.0,
                   # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                   negative_sample_rate=5,
                   # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                   transform_queue_size=4.0,
                   a=None,  # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   b=None,  # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   # default: None, If int, random_state is the seed used by the random number generator;
                   random_state=42,
                   # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                   metric_kwds=None,
                   # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                   angular_rp_forest=False,
                   # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                   target_n_neighbors=-1,
                   # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
                   # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                   # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                   # default 42, Random seed used for the stochastic aspects of the transform operation.
                   transform_seed=42,
                   # default False, Controls verbosity of logging.
                   verbose=False,
                   # default False, Controls if the rows of your data should be uniqued before being embedded.
                   unique=False,
                   )
    features_umap = reducer.fit_transform(features)

    # Check the shape of the new data
    print('Shape of umaps: ', features_umap.shape)

    np.save(os.path.join(output_dirpath, "features_umaps.npy"), features_umap)
    unique_labels = list(np.unique(labels))
    print("unique_labels", unique_labels)
    colors = ['#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87', '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658', '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA',
              '#58A4C3', '#E4C755', '#F7F398', '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B', '#712820', '#DCC1DD', '#CCE0F5', '#CCC9E6', '#625D9E', '#68A180', '#3A6963', '#968175']
    fig = plt.figure(figsize=(12, 10))
    for idx in range(len(unique_labels)):
        plt.scatter(features_umap[labels == unique_labels[idx], 0], features_umap[labels ==
                    unique_labels[idx], 1], color=colors[idx], label=unique_labels[idx], s=2)
        plt.legend()
        plt.savefig(os.path.join(output_dirpath, "umap_plot.pdf"))


def fit_classifier(features, labels, output_dirpath):

    clf = LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=1.0, fit_intercept=True, intercept_scaling=1,
                             class_weight="balanced", random_state=42, solver='sag', max_iter=500, multi_class='multinomial', n_jobs=20)
    clf.fit(features, labels)
    predict_y = clf.predict(features)
    output_str = classification_report(y_true=labels, y_pred=predict_y)
    with open(os.path.join(output_dirpath, "classifier_prediction_ouput.txt"), 'w') as f:
        f.write(output_str+"\n")

    dump(clf, os.path.join(output_dirpath,
         'LogisticRegression_classifier_model.joblib'))
