#!/usr/bin/env python3
import numpy as np

def prune(distanceMatrix, neighbours):
    """
    Prunes a complete graph (input distance matrix), keeping at least a
    specified number of neighbours for each node.

    Parameters:
    -----------
    distanceMatrix = 2D numpy array
    neighbours = integer

    Return: Dij_pruned, a 2D numpy array, represents distance matrix of pruned
            graph
    """
    Dij_temp = distanceMatrix
    Adj = np.zeros(distanceMatrix.shape)
    for ii in range(distanceMatrix.shape[0]):
        idx = np.argsort(Dij_temp[ii,:])
        Adj[ii,idx[1]] = 1
        Adj[idx[1],ii] = 1
        for jj in range(neighbours):
            Adj[ii,idx[jj]] = 1
            Adj[idx[jj],ii] = 1
    Dij_pruned = Dij_temp * Adj
    return Dij_pruned
