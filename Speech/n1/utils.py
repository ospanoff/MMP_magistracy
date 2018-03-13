import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_projections(V, title=''):
    '''
    Plots projections of the tensor on XY, XZ and YZ axis
    :param V: 3D numpy array of magnitudes
    '''
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(V.mean(0))
    plt.colorbar()
    plt.xlabel('Z', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.subplot(1, 3, 2)
    plt.imshow(V.mean(1))
    plt.colorbar()
    plt.xlabel('Z', fontsize=15)
    plt.ylabel('X', fontsize=15)
    plt.subplot(1, 3, 3)
    plt.imshow(V.mean(2))
    plt.colorbar()
    plt.xlabel('Y', fontsize=15)
    plt.ylabel('X', fontsize=15)
    plt.suptitle(title, fontsize=20)


def scatter_trajectory(x, y, z, ax=None):
    '''
    Scatters the trajectory of the plane
    :param x,y,z: np.array with corresponding coordinates
    of the trajectory
    '''
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    return ax


def scatter_volume(V, threshold, ax=None):
    '''
    Scatters the trajectory of the plane
    :param V: filtered radar signal
    :param threshold: points with higher intensity
    than this will be considered trajectory points
    '''
    x, y, z = np.where(V > threshold)
    return scatter_trajectory(x, y, z, ax)
