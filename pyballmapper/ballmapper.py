import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

import warnings

from tqdm.auto import tqdm

from numba import njit

import copy


@njit
def _euclid_distance(x, y):
    return np.linalg.norm(x - y)


def _find_landmarks_greedy(X, eps, orbits=None, metric=None, order=None, verbose=False):
    """Finds the landmaks points via a greedy search procedure.
    
    Selects the first non-covered points in the cosidered order, adds it to the \
    list of landmarks and labels as covered all point inside its eps-ball. \
    Repeats the procedure till there are no more uncovered points.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) \
            or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features.
        For metric='precomputed', the expected shape of X is
        (n_samples, n_samples).

    eps : float
        The radius of the balls.

    orbits : list of lenght n_samples, default=None
        For each data points, contains a list of points in its orbit. 
        Use it to create an Equivariant BallMapper.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy \
        search for landmarks. Different ordering might lead to different \
        BallMapper graphs.
        By defaults uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

    Returns
    ----------
    landmarks: list
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball

    
    """

    if metric == "precomputed":
        n_points = X.shape[0]
    else:
        n_points = len(X)

    # set the distance function
    # f is used to access the points
    f = lambda i: X[i]
    if metric == "euclidean":
        distance = _euclid_distance

    elif metric == "precomputed":
        distance = lambda x, y: X[x, y]
        f = lambda i: i
        if verbose:
            print("using precomputed distance matrix")

    else:
        distance = metric
        if verbose:
            print("using custom distance {}".format(distance))

    # set the orbits
    if orbits is None:
        points_have_orbits = False
    elif (type(orbits) is np.ndarray) or (type(orbits) is list):
        if len(orbits) != n_points:
            points_have_orbits = False
            warnings.warn(
                "Warning........... orbits is not compatible with points, ignoring it"
            )
        else:
            points_have_orbits = True
    else:
        warnings.warn(
            "Warning........... orbits should be a list or a numpy array, ignoring it"
        )
        points_have_orbits = False

    # find landmark points
    landmarks = {}  # dict of points {idx_v: idx_p, ... }
    centers_counter = 0

    # check wheter order is a list of lenght = len(points)
    # otherwise use the defaut ordering
    if order:
        if len(np.unique(order)) != n_points:
            warnings.warn(
                "Warning........... order is not compatible with points, using default ordering"
            )
            order = range(n_points)
    else:
        order = range(n_points)

    if verbose:
        print("Finding vertices...")

    pbar = tqdm(order, disable=not (verbose == "tqdm"))

    for idx_p in pbar:
        # current point
        p = f(idx_p)

        pbar.set_description("{} vertices found".format(centers_counter))

        is_covered = False

        for idx_v in landmarks:
            if distance(p, f(landmarks[idx_v])) <= eps:
                is_covered = True
                break

        if not is_covered:
            landmarks[centers_counter] = idx_p
            centers_counter += 1
            # add points in the orbit
            if points_have_orbits:
                for idx_p_o in orbits[idx_p]:
                    if idx_p_o != idx_p:
                        landmarks[centers_counter] = idx_p_o
                        centers_counter += 1

    # compute points_covered_by_landmarks
    if verbose:
        print("{} vertices found.".format(centers_counter))
        print("Computing points_covered_by_landmarks...")
    points_covered_by_landmarks = dict()
    for idx_v in tqdm(landmarks, disable=not (verbose == "tqdm")):
        points_covered_by_landmarks[idx_v] = []
        for idx_p in order:
            if distance(f(idx_p), f(landmarks[idx_v])) <= eps:
                points_covered_by_landmarks[idx_v].append(idx_p)

    return landmarks, points_covered_by_landmarks


def _find_landmarks(X, eps=None, orbits=None, metric=None, order=None, verbose=False):
    """Finds the landmaks points. At the moment the only option is a greedy search

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) \
            or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features.
        For metric='precomputed', the expected shape of X is
        (n_samples, n_samples).

    eps : float
        The radius of the balls.

    orbits : list of lenght n_samples, default=None
        For each data points, contains a list of points in its orbit. 
        Use it to create an Equivariant BallMapper.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy \
        search for landmarks. Different ordering might lead to different \
        BallMapper graphs.
        By defaults uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

    Returns
    ----------
    landmarks: list
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball

    
    """
    landmarks, points_covered_by_landmarks = _find_landmarks_greedy(
        X, eps, orbits, metric, order, verbose
    )

    return landmarks, points_covered_by_landmarks


class BallMapper:
    def __init__(
        self,
        X: np.ndarray,
        eps,
        orbits=None,
        metric="euclidean",
        order=None,
        verbose=False,
    ):
        """Create a BallMapper graph from vector array or distance matrix.

        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Data vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For metric='precomputed', the expected shape of X is
            (n_samples, n_samples).

        eps : float
            The radius of the balls.

        orbits : list of lenght n_samples, default=None
            For each data points, contains a list of points in its orbit. 
            Use it to create an Equivariant BallMapper.

        metric : str, or callable, default='euclidean'
            The metric to use when calculating distance between instances in a
            feature array.
            If metric is 'precomputed', X is assumed to be a distance matrix and
            must be square.

        order: array-like of shape (n_samples, ), default=None
            The order in which to consider the data points in the greedy \
            search for landmarks. Different ordering might lead to different \
            BallMapper graphs.
            By defaults uses the order of X.

        verbose: bool or string, default=False
            Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

        Attributes
        ------------

        Graph: NetworkX Graph object
            The BallMapper graph. Each node correspond to a covering ball and has attributes: \
            'landmark' the id of the corresponding landmark point \
            'points covered' the ids of the points covered by the corresponding ball

        eps: float
            The input radius of the balls.

        points_covered_by_landmarks: dict
            keys: landmarks ids \
            values: list of ids of the points covered by the corresponding ball

        Notes
        ----------
        https://arxiv.org/abs/1901.07410

        """

        self.eps = eps

        if X.dtype != float:
            warnings.warn("Warning..........the dtype of the input data is {}, not float. Change it to float if you want to use the default numba euclidean distance.".format(X.dtype))


        # find ladmarks
        landmarks, self.points_covered_by_landmarks = _find_landmarks(
            X, eps, orbits, metric, order, verbose
        )

        # find edges
        if verbose:
            print("Running BallMapper ")
            print("Finding edges...")
        edges = []  # list of edges [[idx_v, idx_u], ...]
        for i, idx_v in tqdm(
            enumerate(list(landmarks.keys())[:-1]), disable=not (verbose == "tqdm")
        ):
            for idx_u in list(landmarks.keys())[i + 1 :]:
                if (
                    len(
                        set(self.points_covered_by_landmarks[idx_v]).intersection(
                            self.points_covered_by_landmarks[idx_u]
                        )
                    )
                    != 0
                ):
                    edges.append([idx_v, idx_u])

        # create Ball Mapper graph
        if verbose:
            print("Creating Ball Mapper graph...")
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(landmarks.keys())
        self.Graph.add_edges_from(edges)

        for node in self.Graph.nodes:
            self.Graph.nodes[node]["landmark"] = landmarks[node]
            self.Graph.nodes[node]["points covered"] = np.array(
                self.points_covered_by_landmarks[node]
            )
            self.Graph.nodes[node]["size"] = len(
                self.Graph.nodes[node]["points covered"]
            )

        if verbose:
            print("Done")

    def add_coloring(self, coloring_df, add_std=False):
        """ Takes pandas dataframe and compute the average and standard deviation \
        of each column for the subset of points colored by each ball.
        Add such values as attributes to each node in the BallMapper graph

        Parameters
        ----------
        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function)

        add_std: bool, default=False
            Wheter to compute also the standard deviation on each ball.
        
        """
        # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
        for node in self.Graph.nodes:
            for name, avg in (
                coloring_df.loc[self.Graph.nodes[node]["points covered"]].mean().items()
            ):
                self.Graph.nodes[node][name] = avg
            # option to add the standar deviation on each node
            if add_std:
                for name, std in (
                    coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                    .std()
                    .items()
                ):
                    self.Graph.nodes[node]["{}_std".format(name)] = std

    def color_by_variable(
        self, my_variable, my_palette, MIN_VALUE=np.inf, MAX_VALUE=-np.inf
    ):
        """ Takes pandas dataframe and compute the average and standard deviation \
        of each column for the subset of points colored by each ball.
        Add such values as attributes to each node in the BallMapper graph

        Parameters
        ----------
        my_variable: string
            

        add_std: bool, default=False
            Wheter to compute also the standard deviation on each ball.
        
        """


        # get the coloring variables
        for k in self.Graph.nodes:
            node_keys = self.Graph.nodes[k].keys()
            break

        if my_variable is None:
            for node in self.Graph.nodes:
                self.Graph.nodes[node]["color"] = cm.get_cmap("tab10")(0)

        elif (
            my_variable not in node_keys
        ):  # TODO find a better way to check
            warnings.warn(
                "Warning........... {} is not a valid coloring, add it using the `add_coloring` method".format(
                    my_variable
                )
            )

        else:
            for node in self.Graph.nodes:
                if self.Graph.nodes[node][my_variable] > MAX_VALUE:
                    MAX_VALUE = self.Graph.nodes[node][my_variable]
                if self.Graph.nodes[node][my_variable] < MIN_VALUE:
                    MIN_VALUE = self.Graph.nodes[node][my_variable]

            for node in self.Graph.nodes:
                if not pd.isna(self.Graph.nodes[node][my_variable]):
                    color_id = (self.Graph.nodes[node][my_variable] - MIN_VALUE) / (
                        MAX_VALUE - MIN_VALUE
                    )
                    self.Graph.nodes[node]["color"] = my_palette(color_id)
                else:
                    self.Graph.nodes[node]["color"] = "black"

        return MIN_VALUE, MAX_VALUE


    def filter_by(self, list_of_points):

        filtered_bm = copy.deepcopy(self)

        for node in filtered_bm.Graph.nodes:
            filtered_bm.points_covered_by_landmarks[node] = list(set(filtered_bm.points_covered_by_landmarks[node]).intersection(list_of_points))
            filtered_bm.Graph.nodes[node]["points covered"] = np.array(
                filtered_bm.points_covered_by_landmarks[node]
            )

            filtered_bm.Graph.nodes[node]["size"] = len(
                filtered_bm.Graph.nodes[node]["points covered"]
            )

        filtered_bm.Graph.remove_nodes_from([node for node in filtered_bm.Graph if filtered_bm.Graph.nodes[node]["size"] == 0])

        return filtered_bm

        

    def draw_networkx(
        self,
        coloring_variable=None,
        color_palette=cm.get_cmap("Reds"),
        colorbar=False,
        this_ax=None,
        MIN_VALUE=np.inf,
        MAX_VALUE=-np.inf,
        MIN_SCALE=100,  # default in nx.draw_networkx is 300
        MAX_SCALE=600,
        pos=None,
        **kwargs
    ):
        MAX_NODE_SIZE = max(
            [self.Graph.nodes[node]["size"] for node in self.Graph.nodes]
        )

        if this_ax == None:
            this_ax = plt.gca()

        MIN_VALUE, MAX_VALUE = self.color_by_variable(coloring_variable, color_palette, MIN_VALUE, MAX_VALUE)

        if pos is None:
            pos=nx.spring_layout(self.Graph, seed=24)

        nx.draw_networkx(
            self.Graph,
            pos=pos,
            node_color=[self.Graph.nodes[node]["color"] for node in self.Graph.nodes],
            node_size=[
                MAX_SCALE * self.Graph.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SCALE
                for node in self.Graph.nodes
            ],
            alpha=0.8,
            ax=this_ax,
            **kwargs
        )

        # plot a legend
        if colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=color_palette,
                norm=plt.Normalize(
                    vmin=MIN_VALUE,
                    vmax=MAX_VALUE,
                ),
            )
            plt.colorbar(sm, ax=this_ax)

        return this_ax
