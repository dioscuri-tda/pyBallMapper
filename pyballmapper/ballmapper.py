import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import cm

import warnings

from tqdm.notebook import tqdm

class BallMapper():
    def __init__(self, X, epsilon, orbits=None, distance=None, order=None, distance_matrix = False, dbg=False):
        
        self.epsilon = epsilon
        if not distance_matrix:
            n_points = len(X)
        else:
            n_points = X.shape[0]

        # set the distance function
        
        f = lambda i : X[i]
        if not distance and not distance_matrix:
            distance = lambda x, y : np.linalg.norm(x - y)
            if dbg: print('using euclidean distance')
        elif not distance and distance_matrix:
            distance = lambda x, y : X[x,y]
            f = lambda i : i
        else:
            if dbg: print('using custom distance {}'.format(distance))

        # set the orbits
        if orbits is None:
            points_have_orbits = False
        elif type(orbits) is np.ndarray  or (type(orbits) is list):
            if len(orbits) != n_points:
                points_have_orbits = False
                warnings.warn("Warning........... orbits is not compatible with points, ignoring it")
            else:
                points_have_orbits = True
        else:
            warnings.warn("Warning........... orbits should be a list or a numpy array, ignoring it")
            points_have_orbits = False
        
        # find landmark points
        landmarks = {} # dict of points {idx_v: idx_p, ... }
        centers_counter = 1 # TODO change it to 0
        
        # check wheter order is a list of lenght = len(points)
        # otherwise use the defaut ordering
        
        if order:
            if len(np.unique(order)) != n_points:
                warnings.warn("Warning........... order is not compatible with points, using default ordering")
                order = range(n_points)
        else:
            order = range(n_points)
        
        if dbg:
            print('Finding vertices...')
        
        pbar = tqdm(order, disable=not(dbg))
        
        for idx_p in pbar:
            
            # current point
            p = f(idx_p)
            
            pbar.set_description("{} vertices found".format(centers_counter))
            
            is_covered = False

            for idx_v in landmarks:
                if distance(p, f(landmarks[idx_v])) <= epsilon:
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
        if dbg:
            print('Computing points_covered_by_landmarks...')
        self.points_covered_by_landmarks = dict()
        for idx_v in tqdm(landmarks, disable=not(dbg)):
            self.points_covered_by_landmarks[idx_v] = []
            for idx_p in order:
                if distance(f(idx_p), f(landmarks[idx_v])) <= epsilon:
                    self.points_covered_by_landmarks[idx_v].append(idx_p)
                
        # find edges
        if dbg:
            print('Finding edges...')
        edges = [] # list of edges [[idx_v, idx_u], ...]
        for i, idx_v in tqdm(enumerate(list(landmarks.keys())[:-1]), disable=not(dbg)):
            for idx_u in list(landmarks.keys())[i+1:]:
                if len(set(self.points_covered_by_landmarks[idx_v]).intersection(self.points_covered_by_landmarks[idx_u])) != 0:
                    edges.append([idx_v, idx_u])
                

                    
        # create Ball Mapper graph
        if dbg:
            print('Creating Ball Mapper graph...')
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(landmarks.keys())
        self.Graph.add_edges_from(edges)
        
        # TODO move the display sizing to the plotting utils
        # MIN_SCALE = 100 
        # MAX_SCALE = 500
    
        # MAX_NODE_SIZE = max([len(self.points_covered_by_landmarks[key]) 
        #                          for key in self.points_covered_by_landmarks])
       
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['landmark'] = landmarks[node]
            self.Graph.nodes[node]['points covered'] = self.points_covered_by_landmarks[node]
            self.Graph.nodes[node]['size'] = len(self.Graph.nodes[node]['points covered'])
            # TODO move the display sizing to the plotting utils
            # rescale the size for display
            # self.Graph.nodes[node]['size rescaled'] = MAX_SCALE*self.Graph.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE 
            self.Graph.nodes[node]['color'] = 'r' # TODO change the defaul color
                
        if dbg:
            print('Done')


    def add_coloring(self, coloring_df, add_std=False):
        # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
        for node in self.Graph.nodes:
            for name, avg in coloring_df.loc[self.Graph.nodes[node]['points covered']].mean(numeric_only=True).iteritems():
                self.Graph.nodes[node][name] = avg
            # option to add the standar deviation on each node
            if add_std:
                for name, std in coloring_df.loc[self.Graph.nodes[node]['points covered']].std(numeric_only=True).iteritems():
                    self.Graph.nodes[node]['{}_std'.format(name)] = std

            
    def color_by_variable(self, my_variable, my_palette, MIN_VALUE = np.inf, MAX_VALUE = -np.inf):

        if my_variable is None:
            for node in self.Graph.nodes:
                self.Graph.nodes[node]['color'] = cm.get_cmap('tab10')(0)

        elif my_variable not in self.Graph.nodes[1].keys(): # TODO find a better way to check
            warnings.warn("Warning........... {} is not a valid coloring, add it using the 'add_coloring' method".format(my_variable))

        else:
            for node in self.Graph.nodes:
                if self.Graph.nodes[node][my_variable] > MAX_VALUE:
                    MAX_VALUE = self.Graph.nodes[node][my_variable]
                if self.Graph.nodes[node][my_variable] < MIN_VALUE:
                    MIN_VALUE = self.Graph.nodes[node][my_variable]

            for node in self.Graph.nodes:
                if not pd.isna(self.Graph.nodes[node][my_variable]):
                    color_id = (self.Graph.nodes[node][my_variable] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
                    self.Graph.nodes[node]['color'] = my_palette(color_id)
                else:
                    self.Graph.nodes[node]['color'] = 'black'


    def draw_networx(self, coloring_variable=None, color_palette=cm.get_cmap(name='Reds'), 
                     colorbar = False,
                     this_ax = None, 
                     MIN_SCALE = 100, # default in nx.draw_networkx is 300
                     MAX_SCALE = 600,
                     **kwargs):
    
        MAX_NODE_SIZE = max([self.Graph.nodes[node]['size'] for node in self.Graph.nodes])

        if this_ax == None:
            this_ax = plt.gca()

        self.color_by_variable(coloring_variable, color_palette)

        nx.draw_networkx(self.Graph, 
                         pos = nx.spring_layout(self.Graph, seed=24),
                         node_color = [self.Graph.nodes[node]['color'] for node in self.Graph.nodes],
                         node_size =  [MAX_SCALE*self.Graph.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE for node in self.Graph.nodes],
                         alpha = 0.8,
                         ax = this_ax,
                         **kwargs)

        # plot a legend
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap = color_palette,
                                       norm = plt.Normalize(vmin=min([self.Graph.nodes[node][coloring_variable] for node in self.Graph.nodes]), 
                                                            vmax=max([self.Graph.nodes[node][coloring_variable] for node in self.Graph.nodes])))
            plt.colorbar(sm, ax=this_ax)

        return this_ax






# mapper on BM using DBscan as clustering algo
# it uses scipy csr sparse matrix to speed up computations
# inputs:
#     cover_BM        ball mapper graph
#     target_space    numpy array where to pull back elements in the BM
#     eps             radius for the DBscan algo
#     min_samples     min number of elements in a cluster that make it a cluster and not noise
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix


def mapper_on_BM(cover_BM, target_space, eps, min_samples=1, dbscan_metric='euclidean', sparse=False, dbg=False):
    new_graph = nx.Graph()

    # creates a sparse CSR matrix
    if sparse:
        target_space = csr_matrix(target_space)

    for node in tqdm(cover_BM.nodes):
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
            new_graph.add_node(str(node)+'_'+str(cluster),
                               points_covered=points_covered_by_cluster)

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


    return new_graph