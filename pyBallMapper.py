import numpy as np
import networkx as nx

class BallMapper():
    def __init__(self, points, epsilon):
        
        # find vertices
        self.vertices = [] # list of points [idx_v, ... ]
        for idx_p, p in enumerate(points):
            is_covered = False

            for idx_v in self.vertices:
                distance = np.linalg.norm(p - points[idx_v])
                if distance <= epsilon:
                    is_covered = True
                    break

            if not is_covered:
                self.vertices.append(idx_p)
                    
        # compute points_covered_by_landmarks
        self.points_covered_by_landmarks = dict()
        for idx_v in self.vertices:
            self.points_covered_by_landmarks[idx_v] = []
            for idx_p, p in enumerate(points):
                distance = np.linalg.norm(p - points[idx_v])
                if distance <= epsilon:
                    self.points_covered_by_landmarks[idx_v].append(idx_p)
                
        # find edges
        self.edges = [] # list of edges [[idx_v, idx_u], ...]
        for i, idx_v in enumerate(self.vertices[:-1]):
            for idx_u in self.vertices[i+1:]:
                if len(set(self.points_covered_by_landmarks[idx_v]).intersection(self.points_covered_by_landmarks[idx_u])) != 0:
                    self.edges.append([idx_v, idx_u])
                

                    
        # create Ball Mapper graph
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(self.vertices)
        self.Graph.add_edges_from(self.edges)
        
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['points covered'] = self.points_covered_by_landmarks[node]