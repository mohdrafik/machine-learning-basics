import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent
print(root_dir)
fullfile_path = root_dir/ "data/unsupervised_learning/anomaly_det"

def load_data():

    X = np.load(fullfile_path/"X_part1.npy")
    X_val = np.load(fullfile_path/"X_val_part1.npy")
    y_val = np.load(fullfile_path/"y_val_part1.npy")
    return X, X_val, y_val

def load_data_multi():
    X = np.load(fullfile_path/"X_part2.npy")
    X_val = np.load(fullfile_path/"X_val_part2.npy")
    y_val = np.load(fullfile_path/"y_val_part2.npy")
    return X, X_val, y_val


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
        
def visualize_fit(X, mu, var):
    """
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')


if __name__ == "__main__":
    X_train, X_val, y_val = load_data()
    print(X_train[:5])
    print(X_val[:5])
    print(y_val[:5])

