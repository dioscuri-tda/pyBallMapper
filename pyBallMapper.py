import numpy as np
import pandas as pd
import networkx as nx

class BallMapper():
    def __init__(self, points, coloring_df, epsilon):
        
        # find vertices
        self.vertices = {} # dict of points {idx_v: idx_p, ... }
        centers_counter = 1
        for idx_p, p in enumerate(points):
            is_covered = False

            for idx_v in self.vertices:
                distance = np.linalg.norm(p - points[self.vertices[idx_v]])
                if distance <= epsilon:
                    is_covered = True
                    break

            if not is_covered:
                self.vertices[centers_counter] = idx_p
                centers_counter += 1
                    
        # compute points_covered_by_landmarks
        self.points_covered_by_landmarks = dict()
        for idx_v in self.vertices:
            self.points_covered_by_landmarks[idx_v] = []
            for idx_p, p in enumerate(points):
                distance = np.linalg.norm(p - points[self.vertices[idx_v]])
                if distance <= epsilon:
                    self.points_covered_by_landmarks[idx_v].append(idx_p)
                
        # find edges
        self.edges = [] # list of edges [[idx_v, idx_u], ...]
        for i, idx_v in enumerate(list(self.vertices.keys())[:-1]):
            for idx_u in list(self.vertices.keys())[i+1:]:
                if len(set(self.points_covered_by_landmarks[idx_v]).intersection(self.points_covered_by_landmarks[idx_u])) != 0:
                    self.edges.append([idx_v, idx_u])
                

                    
        # create Ball Mapper graph
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(self.vertices)
        self.Graph.add_edges_from(self.edges)
        
        MIN_SCALE = 100
        MAX_SCALE = 500
    
        MAX_NODE_SIZE = max([len(self.points_covered_by_landmarks[key]) 
                                 for key in self.points_covered_by_landmarks])
       
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['points covered'] = self.points_covered_by_landmarks[node]
            self.Graph.nodes[node]['size'] = len(self.Graph.nodes[node]['points covered'])
            # rescale the size for display
            self.Graph.nodes[node]['size rescaled'] = MAX_SCALE*self.Graph.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE 
            self.Graph.nodes[node]['color'] = 'r'
            
            for name, avg in coloring_df.loc[self.Graph.nodes[node]['points covered']].mean().iteritems():
                self.Graph.nodes[node][name] = avg
            

        # initialize min and max color values
        self.min_color_value = 0
        self.max_color_value = 0
            
            
    def color_by_variable(self, my_variable, my_palette, MIN_VALUE = 10000, MAX_VALUE = -10000):

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

        self.min_color_value = MIN_VALUE
        self.max_color_value = MAX_VALUE 