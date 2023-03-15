import numpy as np
import pandas as pd
import networkx as nx

from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix

from tqdm.auto import tqdm

from .ballmapper import BallMapper


def MoBM(cover_BM, target_space, eps, min_samples=1, dbscan_metric='euclidean', sparse=False, dbg=False):

    # mapper on BM using DBscan as clustering algo
    # it uses scipy csr sparse matrix to speed up computations
    # inputs:
    #     cover_BM        ball mapper graph
    #     target_space    numpy array where to pull back elements in the BM
    #     eps             radius for the DBscan algo
    #     min_samples     min number of elements in a cluster that make it a cluster and not noise
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    
    new_graph = nx.Graph()

    # creates a sparse CSR matrix
    if sparse:
        target_space = csr_matrix(target_space)

    for node in tqdm(cover_BM.nodes, disable=not(dbg)):
        X = target_space[cover_BM.nodes[node]['points covered'], :]

        db = DBSCAN(eps=eps, min_samples=min_samples, metric=dbscan_metric).fit(X)
        # create a set of unique labels
        labels = set(db.labels_) - {-1} # outliers are not clusters

        if dbg: 
            print('\n **********')
            print(node, X.shape[0], labels)

        # for each cluster
        # add a new vertex to the new graph
        for cluster in labels:
            if dbg:
                # print the number of points in the cluster
                print('\t', cluster, (db.labels_ == cluster).sum())

            # retrives the indices of the points_covered by the cluster
            points_covered_by_cluster = cover_BM.nodes[node]['points covered'][np.where(db.labels_ == cluster)]
            # creates a node
            new_graph.add_node(str(node)+'_'+str(cluster))
            new_graph.nodes[str(node)+'_'+str(cluster)]['points covered'] = points_covered_by_cluster
            new_graph.nodes[str(node)+'_'+str(cluster)]['size'] = len(points_covered_by_cluster)

        for neigh in [v for v in nx.neighbors(cover_BM, node) if v > node]:
            neigh_X = target_space[cover_BM.nodes[neigh]['points covered'], :]

            neigh_db = DBSCAN(eps=eps, min_samples=min_samples, metric=dbscan_metric).fit(neigh_X)
            neigh_labels = set(neigh_db.labels_) - {-1} # outliers are not clusters

            # add edges between clusters that belongs to neigh in the original graph
            # if they share at least one element
            for cluster in labels:
                for neigh_cluster in neigh_labels:
                    points_covered_by_cluster = cover_BM.nodes[node]['points covered'][np.where(db.labels_== cluster)]
                    points_covered_by_neigh= cover_BM.nodes[neigh]['points covered'][np.where(neigh_db.labels_ == neigh_cluster)]
                    if len( set(points_covered_by_cluster)&set(points_covered_by_neigh) ) != 0:
                        new_graph.add_edge(str(node)+'_'+str(cluster), str(neigh)+'_'+str(neigh_cluster) )

    # TODO: find a better way of creating a BM object
    # here I am just creating a BM with one point and then replacing its Graph
    fake_bm = BallMapper([0], eps=0)
    fake_bm.Graph = new_graph

    return fake_bm