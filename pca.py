import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import get_kaggle_MNIST


def main():
    Xtrain, Ytrain, Xtest, Ytest = get_kaggle_MNIST()

    pca = PCA()
    reduced = pca.fit_transform(Xtrain)
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, c=Ytrain, alpha=.5)
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    # Cumulative variance
    cumulative = []
    last = 0
    for variance in pca.explained_variance_ratio_:
        cumulative.append(last + variance)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.show()

if __name__ == "__main__":
    main()