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


def _find_landmarks_greedy(
    X, eps, orbits=None, metric=None, order=None, multiple_eps=False, verbose=False
):
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

    n_points = X.shape[0]

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

    # since the radius of every ball might be different, we need to store them
    eps_dict = dict()

    if verbose:
        print("Finding vertices...")

    pbar = tqdm(order, disable=not (verbose == "tqdm"))

    for idx_p in pbar:
        # current point
        p = f(idx_p)

        pbar.set_description("{} vertices found".format(centers_counter))

        is_covered = False

        # get the radius of the point p
        if multiple_eps:
            eps_p = eps[idx_p]
        else:
            eps_p = eps

        for idx_v in landmarks:
            if distance(p, f(landmarks[idx_v])) <= eps_p:
                is_covered = True
                break

        if not is_covered:
            landmarks[centers_counter] = idx_p
            eps_dict[centers_counter] = eps_p
            centers_counter += 1
            # add points in the orbit
            if points_have_orbits:
                for idx_p_o in orbits[idx_p]:
                    if idx_p_o != idx_p:
                        landmarks[centers_counter] = idx_p_o
                        eps_dict[centers_counter] = eps[idx_p_o]
                        centers_counter += 1

    # compute points_covered_by_landmarks
    if verbose:
        print("{} vertices found.".format(centers_counter))
        print("Computing points_covered_by_landmarks...")
    points_covered_by_landmarks = dict()
    for idx_v in tqdm(landmarks, disable=not (verbose == "tqdm")):
        points_covered_by_landmarks[idx_v] = []
        for idx_p in order:
            # get the radius of the point p
            if multiple_eps:
                eps_p = eps[idx_p]
            else:
                eps_p = eps
            if distance(f(idx_p), f(landmarks[idx_v])) <= eps_p:
                points_covered_by_landmarks[idx_v].append(idx_p)

    return landmarks, points_covered_by_landmarks, eps_dict


def _find_landmarks_adaptive(
    X, eps, max_size, eta=0.7, orbits=None, metric=None, order=None, verbose=False
):
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
    centers_counter = -1

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

    # since the radius of every ball might be different, we need to store them
    eps_dict = dict()
    points_covered_by_landmarks = dict()

    for idx_p in pbar:
        # current point
        p = f(idx_p)

        pbar.set_description("{} vertices found".format(centers_counter))

        is_covered = False

        for idx_v in landmarks:
            if distance(p, f(landmarks[idx_v])) <= eps_dict[idx_v]:
                is_covered = True
                break

        if not is_covered:
            # we use this as a new landmark
            centers_counter += 1
            landmarks[centers_counter] = idx_p

            # compute points_covered_by this new landmarks
            points_covered_by_landmarks[centers_counter] = []
            eps_dict[centers_counter] = eps
            for idx_p2 in order:
                if (
                    distance(f(idx_p2), f(landmarks[centers_counter]))
                    <= eps_dict[centers_counter]
                ):
                    points_covered_by_landmarks[centers_counter].append(idx_p2)

            while len(points_covered_by_landmarks[centers_counter]) > max_size:
                # decrease the radius and recompute
                eps_dict[centers_counter] *= eta
                if verbose:
                    print(
                        "ball {} - point {}  has size {}. decreasing eps to {}".format(
                            centers_counter,
                            idx_p,
                            len(points_covered_by_landmarks[centers_counter]),
                            eps_dict[centers_counter],
                        )
                    )

                points_covered_by_landmarks[centers_counter] = []
                for idx_p2 in order:
                    if (
                        distance(f(idx_p2), f(landmarks[centers_counter]))
                        <= eps_dict[centers_counter]
                    ):
                        points_covered_by_landmarks[centers_counter].append(idx_p2)

            # add points in the orbit
            if points_have_orbits:

                eps_o = eps_dict[centers_counter]

                for idx_p_o in orbits[idx_p]:
                    if idx_p_o != idx_p:
                        centers_counter += 1
                        landmarks[centers_counter] = idx_p_o

                        # if verbose:
                        #     print(
                        #         "adding orbit landmark {} with eps {}".format(
                        #             centers_counter, eps_o
                        #         )
                        #     )

                        # compute points_covered_by this new landmarks
                        # using the same radius as the original landmark
                        points_covered_by_landmarks[centers_counter] = []
                        eps_dict[centers_counter] = eps_o
                        for idx_p2 in order:
                            if (
                                distance(f(idx_p2), f(landmarks[centers_counter]))
                                <= eps_dict[centers_counter]
                            ):
                                points_covered_by_landmarks[centers_counter].append(
                                    idx_p2
                                )

    return landmarks, points_covered_by_landmarks, eps_dict


def _find_landmarks(
    X,
    eps=None,
    orbits=None,
    metric=None,
    order=None,
    method=None,
    multiple_eps=False,
    verbose=False,
    **kwargs
):
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

    if method == "adaptive":
        if multiple_eps:
            raise ValueError("Adaptive with multiple radii still not implemented! :()")
        landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_adaptive(
            X=X,
            eps=eps,
            max_size=kwargs["max_size"],
            eta=kwargs["eta"],
            orbits=orbits,
            metric=metric,
            order=order,
            verbose=verbose,
        )
    else:
        landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_greedy(
            X, eps, orbits, metric, order, multiple_eps, verbose
        )

    return landmarks, points_covered_by_landmarks, eps_dict


class BallMapper:
    def __init__(
        self,
        X: np.ndarray,
        eps,
        coloring_df=None,
        orbits=None,
        metric="euclidean",
        order=None,
        method=None,
        verbose=False,
        **kwargs
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

        eps : float, or iterable
            The radius of the balls. If a list of lenght n_samples is passed \
            each point has its own radius.

        orbits : list of lenght n_samples, default=None
            For each data points, contains a list of points in its orbit. 
            Use it to create an Equivariant BallMapper.

        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function), default=None
            If defined, uses the `add_coloring` method to compute the average value 
            of of each column for the points covered by each ball.

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

        if not isinstance(X, np.ndarray):
            try:
                X = np.asanyarray(X, dtype=float)
            except:
                warnings.warn(
                    "the input is {} - cannot convert it to numpy array".format(type(X))
                )

        n_points = X.shape[0]

        ## convert order to a list
        if order == None:
            order = range(n_points)

        elif isinstance(order, np.ndarray):
            order = order.tolist()

        elif not isinstance(order, list):
            warnings.warn(
                "Warning........... order is not a list or numpy array, using default ordering"
            )
            order = range(n_points)

        # check wheter order is a list of lenght = len(points)
        # otherwise use the defaut ordering
        if len(np.unique(order)) != n_points:
            warnings.warn(
                "Warning........... order is not compatible with points, using default ordering"
            )
            order = range(n_points)

        # check the radius

        self.multiple_eps = False
        if isinstance(eps, (float, np.floating)):
            print("The eps is a single float.")
        elif isinstance(eps, (int, np.integer)):
            print("The eps is a single integer.")
        elif isinstance(eps, (list, np.ndarray)):
            print("The eps is a list or NumPy array of lenght {}.".format(len(eps)))
            self.multiple_eps = True
            if len(eps) != n_points:
                raise ValueError(
                    "The list of radii has the wrong lenght {}. Does not match the number of points: {}".format(
                        len(eps), n_points
                    )
                )
        else:
            raise ValueError("The variable is neither a scalar nor a list/array.")

        # find ladmarks
        landmarks, self.points_covered_by_landmarks, self.eps_dict = _find_landmarks(
            X, eps, orbits, metric, order, method, self.multiple_eps, verbose, **kwargs
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

        if isinstance(coloring_df, pd.DataFrame):
            if verbose:
                print("Computing coloring")
            self.add_coloring(coloring_df)

        if verbose:
            print("Done")

    def add_coloring(
        self, coloring_df, custom_function=np.mean, custom_name=None, add_std=False
    ):
        """Takes pandas dataframe and compute the average and standard deviation \
        of each column for the subset of points colored by each ball.
        Add such values as attributes to each node in the BallMapper graph

        Parameters
        ----------
        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function)
        custom_function : callable, optional
            a function to compute on the `coloring_df` columns, by default numpy.mean
        custom_name : string, optional
            sets the attributes naming scheme, by default None, the attribute names will be the column names
        add_std: bool, default=False
            Wheter to compute also the standard deviation on each ball
        """
        # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
        for node in self.Graph.nodes:
            for col_name, avg in (
                coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                .apply(custom_function, axis=0)
                .items()
            ):
                if custom_name:
                    name = "{}_{}".format(col_name, custom_name)
                else:
                    name = col_name
                self.Graph.nodes[node][name] = avg
            # option to add the standar deviation on each node
            if add_std:
                for col_name, std in (
                    coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                    .std()
                    .items()
                ):
                    self.Graph.nodes[node]["{}_std".format(col_name)] = std

    def color_by_variable(
        self, my_variable, my_palette, MIN_VALUE=np.inf, MAX_VALUE=-np.inf
    ):
        """Colors the BallMapper graph using a specified variable. The `add_coloring` method needs to be called first. Automatically computes the min and max value for the colormap.

        Parameters
        ----------
        my_variable : string
            the variable to color by
        my_palette : matplotlib.colors.Colormap
            a valid colormap
        MIN_VALUE : float, optional
            the value to be assigned to the lowest color in the cmap, by default np.inf
        MAX_VALUE : float, optional
            the value to be assigned to the highest color in the cmap, by default -np.inf

        Returns
        -------
        MIN_VALUE, MAX_VALUE
            the computed min and max values of `my_variable` on the BM nodes, useful to set the limits for a colorbar
        """

        # get the coloring variables
        for k in self.Graph.nodes:
            node_keys = self.Graph.nodes[k].keys()
            break

        if my_variable is None:
            for node in self.Graph.nodes:
                self.Graph.nodes[node]["color"] = cm.get_cmap("tab10")(0)

        elif my_variable not in node_keys:  # TODO find a better way to check
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
        """return a copy of the BallMapper object with only the nodes covering a subset of points

        Parameters
        ----------
        list_of_points : list
            list of the subset of points to keep

        Returns
        -------
        BallMapper
            the filtered BallMapper graph
        """

        filtered_bm = copy.deepcopy(self)

        for node in filtered_bm.Graph.nodes:
            filtered_bm.points_covered_by_landmarks[node] = list(
                set(filtered_bm.points_covered_by_landmarks[node]).intersection(
                    list_of_points
                )
            )
            filtered_bm.Graph.nodes[node]["points covered"] = np.array(
                filtered_bm.points_covered_by_landmarks[node]
            )

            filtered_bm.Graph.nodes[node]["size"] = len(
                filtered_bm.Graph.nodes[node]["points covered"]
            )

        filtered_bm.Graph.remove_nodes_from(
            [
                node
                for node in filtered_bm.Graph
                if filtered_bm.Graph.nodes[node]["size"] == 0
            ]
        )

        return filtered_bm

    def points_and_balls(self):
        """returns a DataFrame with the `points_covered_by_landmarks` information

        Returns
        -------
        pandas.DataFrame

        """
        to_df = []
        for ball, points in self.points_covered_by_landmarks.items():
            for p in points:
                to_df.append([p, ball])

        return pd.DataFrame(to_df, columns=["point", "ball"])

    def draw_networkx(
        self,
        coloring_variable=None,
        color_palette=cm.get_cmap("Reds"),
        colorbar=False,
        colorbar_label=None,
        ax=None,
        MIN_VALUE=np.inf,
        MAX_VALUE=-np.inf,
        MIN_SCALE=100,  # default in nx.draw_networkx is 300
        MAX_SCALE=600,
        pos=None,
        **kwargs
    ):
        """Wrapper around the `networkx.draw_networkx` method with colorbar support.

        Parameters
        ----------
        coloring_variable : string, optional
            the variable to use for coloring the BM graph, by default None
        color_palette : matplotlib.colors.Colormap, optional
            the coloring palette to use, by default cm.get_cmap("Reds")
        colorbar : bool, optional
            the label on the colorbar's long axis.
        colorbar_label : str, optional
            whether to add a colorbar to the plot, by default False
        ax : matplotlib.axes.Axes, optional
            the matplotlib ax where to plot the graph. If None, the current ax is used. By default None
        MIN_VALUE : float, optional
            the value to be assigned to the lowest color in the cmap, by default np.inf
        MAX_VALUE : float, optional
            the value to be assigned to the highest color in the cmap, by default -np.inf
        MIN_SCALE : int, optional
            the minimum radius for the nodes, by default 100
        MIN_SCALE : int, optional
            the maximum radius for the nodes, by default 100
        pos : dictionary, optional
            A dictionary with nodes as keys and positions as values. If not specified a spring layout positioning will be computed. See `networkx.drawing.layout` for functions that compute node positions. By default None

        Returns
        -------
        ax
            the matplotlib ax
        """
        MAX_NODE_SIZE = max(
            [self.Graph.nodes[node]["size"] for node in self.Graph.nodes]
        )

        if ax == None:
            ax = plt.gca()

        MIN_VALUE, MAX_VALUE = self.color_by_variable(
            coloring_variable, color_palette, MIN_VALUE, MAX_VALUE
        )

        if pos is None:
            pos = nx.spring_layout(self.Graph, seed=24)

        nx.draw_networkx(
            self.Graph,
            pos=pos,
            node_color=[self.Graph.nodes[node]["color"] for node in self.Graph.nodes],
            node_size=[
                MAX_SCALE * self.Graph.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SCALE
                for node in self.Graph.nodes
            ],
            alpha=0.8,
            ax=ax,
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
            plt.colorbar(sm, label=colorbar_label, ax=ax)

        return ax
