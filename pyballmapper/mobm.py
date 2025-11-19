import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from .ballmapper import BallMapper


class MapperonBallMapper(BallMapper):
    def __init__(
        self,
        cover_BM,
        target_space,
        eps,
        min_samples=1,
        dbscan_metric="euclidean",
        sparse=False,
        dbg=False,
    ):
        """Mapper on BallMapper using DBSCAN as clustering algorithm.

        It uses scipy csr sparse matrix to speed up computations.

        Parameters
        ----------
        cover_B : BallMapper
            Ball Mapper graph covering the image space.

        target_space : array-like of shape (n_samples, n_features)
            The high dimensional pointcloud where to pull back from nodes of \
            the BallMapper.

        eps : float
            eps parameter for DBSCAN

        min_samples : int
            min number of elements in a cluster that make it a cluster and not noise. \
            Another DBSCAN parameter.

        Attributes
        ----------
        Graph: NetworkX graph
            The resulting Mapper on BallMapper graph. With node attributes `size` and \
            `points covered`.


        Notes
        ----------

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


        """

        new_graph = nx.Graph()

        cover_BM = cover_BM.Graph

        # creates a sparse CSR matrix
        if sparse:
            target_space = csr_matrix(target_space)

        for node in tqdm(cover_BM.nodes, disable=not (dbg)):
            X = target_space[cover_BM.nodes[node]["points covered"], :]

            db = DBSCAN(eps=eps, min_samples=min_samples, metric=dbscan_metric).fit(X)
            # create a set of unique labels
            labels = set(db.labels_) - {-1}  # outliers are not clusters

            if dbg:
                print("\n **********")
                print(node, X.shape[0], labels)

            # for each cluster
            # add a new vertex to the new graph
            for cluster in labels:
                if dbg:
                    # print the number of points in the cluster
                    print("\t", cluster, (db.labels_ == cluster).sum())

                # retrives the indices of the points_covered by the cluster
                points_covered_by_cluster = cover_BM.nodes[node]["points covered"][
                    np.where(db.labels_ == cluster)
                ]
                # creates a node
                new_graph.add_node(str(node) + "_" + str(cluster))
                new_graph.nodes[str(node) + "_" + str(cluster)][
                    "points covered"
                ] = points_covered_by_cluster
                new_graph.nodes[str(node) + "_" + str(cluster)]["size"] = len(
                    points_covered_by_cluster
                )

            for neigh in [v for v in nx.neighbors(cover_BM, node) if v > node]:
                neigh_X = target_space[cover_BM.nodes[neigh]["points covered"], :]

                neigh_db = DBSCAN(
                    eps=eps, min_samples=min_samples, metric=dbscan_metric
                ).fit(neigh_X)
                neigh_labels = set(neigh_db.labels_) - {-1}  # outliers are not clusters

                # add edges between clusters that belongs to neigh in the original graph
                # if they share at least one element
                for cluster in labels:
                    for neigh_cluster in neigh_labels:
                        points_covered_by_cluster = cover_BM.nodes[node][
                            "points covered"
                        ][np.where(db.labels_ == cluster)]
                        points_covered_by_neigh = cover_BM.nodes[neigh][
                            "points covered"
                        ][np.where(neigh_db.labels_ == neigh_cluster)]
                        if (
                            len(
                                set(points_covered_by_cluster)
                                & set(points_covered_by_neigh)
                            )
                            != 0
                        ):
                            new_graph.add_edge(
                                str(node) + "_" + str(cluster),
                                str(neigh) + "_" + str(neigh_cluster),
                            )

        # re-label the nodes to integers, and move the 'node_cluster' info to labels
        # this is to ensure that bokeh behaves well
        # better to use integers as node keys
        for node in new_graph.nodes:
            new_graph.nodes[node]["label"] = str(node)

        # convert node labels to int
        nx.relabel_nodes(
            new_graph, {n: i for i, n in enumerate(new_graph.nodes)}, copy=False
        )

        self.Graph = new_graph
