# -*- coding: utf-8 -*-
#
# version: 240209

import numpy as np
import math
import pandas as pd
from sklearn.utils import Bunch
from sklearn.datasets import load_digits, load_wine, load_breast_cancer, fetch_olivetti_faces
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
cmmax = 13

# Load a data set `Apple Quality` available in Kaggle.
# Some preprocessing is involved.
#
# URL: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

def load_apple_quality_csv(n_samples):
    raw_data = pd.read_csv('./data/apple_quality.csv', sep=',', header=0)
    records = raw_data.to_numpy()
    data_array = []
    for r0 in records:
        _d = []
        for r1 in r0[1:-1]:
            val = 0
            if isinstance(r1, str):
                try:
                    val = float(r1)
                except Exception as err:
                    pass
            elif np.isnan(r1): val = 0
            else: val = r1
            _d.append(val)
        data_array.append(_d)

    data = np.array(data_array)
    # there is no pre-defined classification of apples, hence we only assign the indices of the data for targets (classifier label)
    target = [1 if lbl=='good' else 0 for lbl in records[:,-1]]

    return Bunch(data = data[:n_samples] if n_samples is not None else data,
               target = target[:n_samples] if n_samples is not None else target,
                feature_names = raw_data.keys().to_list()[1:-1]
                )

# Generate a Bunch (dataset type) of 3d-coordinates, thought of NLDR methods as 3d-to-2d space projection
def load_3ds(n_samples = 10, supervised = None):
    np.random.seed(0)
    data = np.random.rand(n_samples, 3)
    if (supervised is not None) and (isinstance(supervised, int) and supervised < n_samples):
        target = []
        for i in range(n_samples):
            target.append(math.floor(i % supervised))
        target = np.array(target)
    else:
        # there is no pre-defined classification of 3d-coordinate, hence we only assign the indices of the data for targets (classifier label)
        target = np.array(range(len(data)))

    return Bunch(data = data,
                 target = target,
                 feature_names = ['x','y','z']
                )

def load_faces():
    samples = fetch_olivetti_faces()
    return samples

# Get an DR method implemented on the scikit-learn tool kits.
def get_algorithm(_class, data):
    algorithm_params = {
        "n_components": 2,
        "random_state": 42
    }

    if str(_class).find('LocallyLinearEmbedding')>=0:
        algorithm_params["method"] = "standard"
        algorithm_params["n_neighbors"] = 5
    elif str(_class).find('TSNE')>=0:
        algorithm_params["perplexity"] = min(49, len(data)-1)
    else:
        pass

    return _class(**algorithm_params)

def draw_summary(samples, algorithm):
    colLabels = samples.feature_names if hasattr(samples, 'feature_names') else []

    data = preprocessing.StandardScaler().fit_transform(samples.data)

    _algorithm = get_algorithm(algorithm, data)
    reduced_samples = _algorithm.fit_transform(data)

    plt.figure(figsize = (10,10))
    
    axes = []
    axshape = (math.ceil(2+len(colLabels)/2), 2) if len(colLabels) <= cmmax else (2,2)
    axes.append(plt.subplot2grid(axshape, (0,0), colspan=2, rowspan=2))

    axes[0].set_xlim(reduced_samples[:, 0].min(), reduced_samples[:, 0].max() + 1)
    axes[0].set_ylim(reduced_samples[:, 1].min(), reduced_samples[:, 1].max() + 1)

    for i in range(len(data)):
        color = plt.cm.tab20(samples.target[i])
        axes[0].text(
            reduced_samples[i, 0],
            reduced_samples[i, 1],
            str(samples.target[i]),
            color=color,
            fontweight = 'bold',
            fontsize = 9
        )
    
    axes[0].set_xlabel("reduced feature 0")
    axes[0].set_ylabel("reduced feature 1")
    
    if len(colLabels) <= cmmax:
        for i in range(len(colLabels)):
            _, bins = np.histogram(data[:,i], bins=50)
            ax = plt.subplot2grid(axshape, (math.floor(2+i/2),i%2))
            ax.hist(data[:,i], bins=bins, alpha=.5, color=plt.cm.tab20b(i))
            if len(colLabels)>0: ax.set_title(colLabels[i])
            ax.set_yticks(())
            axes.append(ax)

    plt.tight_layout()

    return (samples, reduced_samples)

def draw_decision_boundary(samples, reduced_samples):
    label_len = len(set(samples.target))

    if label_len <= 40:
        clf = Pipeline(
            steps=[("knn", KNeighborsClassifier(n_neighbors=11))]
        )
        clf.set_params(knn__weights="uniform").fit(reduced_samples, samples.target)
        
        _, ax = plt.subplots(ncols=1, figsize=(10, 10))
        dbd = DecisionBoundaryDisplay.from_estimator(
            clf,
            reduced_samples,
            response_method="predict",
            plot_method="pcolormesh",
            xlabel='feature 0',
            ylabel='feature 1',
            shading="auto",
            alpha=0.8,
            ax=ax,
        )
        for i in range(len(reduced_samples)):
            ax.scatter(
                *reduced_samples[i],
                marker=f"${samples.target[i]}$",
                color=plt.cm.nipy_spectral(samples.target[i] / 10),
                alpha=0.5,
            )
    else:
        print("too many labels of %d" % label_len)