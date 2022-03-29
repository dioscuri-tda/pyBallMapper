import numpy as np
import pandas as pd
import networkx as nx

from matplotlib.colors import to_hex

from bokeh.plotting import figure, show

from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          Plot, ResetTool,
                          ColumnDataSource,
                          TapTool, WheelZoomTool, PanTool,
                          SaveTool, Range1d, MultiLine
                          )

#from bokeh.palettes import linear_palette, Reds256, Turbo256
from bokeh.plotting import from_networkx, figure, curdoc




# creates a nx graph that bokeh can plot
def create_bokeh_graph(G,
                       values_df,
                       my_palette):

    MIN_SCALE = 7
    MAX_SCALE = 20
    
    MAX_NODE_SIZE = max([len(G.nodes[node]['points covered']) 
                         for node in G.nodes])

    for node in G.nodes:
        G.nodes[node]['size'] = len(G.nodes[node]['points covered'])
        # rescale the size for display
        G.nodes[node]['size rescaled'] = MAX_SCALE*G.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE

        G.nodes[node]['color'] = to_hex(my_palette(0))

        for name, avg in values_df.loc[G.nodes[node]['points covered']].mean().iteritems():
            G.nodes[node][name] = avg

    return G


def color_nodes(G, my_variable, my_palette, MIN_VALUE = 10000, MAX_VALUE = -10000):

    for node in G.nodes:
        if G.nodes[node][my_variable] > MAX_VALUE:
            MAX_VALUE = G.nodes[node][my_variable]
        if G.nodes[node][my_variable] < MIN_VALUE:
            MIN_VALUE = G.nodes[node][my_variable]

    for node in G.nodes:
        if not pd.isna(G.nodes[node][my_variable]):
            color_id = (G.nodes[node][my_variable] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
            G.nodes[node]['color'] = to_hex(my_palette(color_id))
        else:
            G.nodes[node]['color'] = 'black'
        
        
    print('color by variable {} \nMIN_VALUE: {:.3f}, MAX_VALUE: {:.3f}'.format(my_variable, MIN_VALUE, MAX_VALUE))
    
    return G, MIN_VALUE, MAX_VALUE
    
    
    
    
class graph_GUI():
    def __init__(self, graph, coloring_df, my_palette):
        
        self.my_palette = my_palette
    
        self.bokeh_graph = create_bokeh_graph(graph, coloring_df, my_palette)
        
        self.plot = Plot(plot_width=700, plot_height=600,
                    x_range=Range1d(-1, 1), y_range=Range1d(-1, 1),
                    sizing_mode="stretch_both")

        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("size", "@size")]
                                             +[(name, '@{}'.format(name))
                                                for name in coloring_df.columns] )
        zoom_tool = WheelZoomTool()
        self.plot.add_tools(PanTool(), node_hover_tool, zoom_tool,
                            ResetTool(), SaveTool())
        self.plot.toolbar.active_scroll = zoom_tool

        self.graph_renderer = from_networkx(self.bokeh_graph, nx.spring_layout,
                                         seed=24, scale=1, center=(0, 0),
                                         k= 10/np.sqrt(len(self.bokeh_graph.nodes)),
                                         iterations=2000)

        # nodes
        self.graph_renderer.node_renderer.glyph = Circle(size='size rescaled',
                                                      fill_color='color',
                                                      fill_alpha=0.8)

        # edges
        self.graph_renderer.edge_renderer.glyph = MultiLine(line_color='black',
                                                         line_alpha=0.8, line_width=1)

        self.plot.renderers.append(self.graph_renderer)
        
        
        
    # this function changes the coloring of the nodes
    def color_by_variable(self, variable):

        if variable == 'None':
            for node in self.bokeh_graph.nodes:
                self.bokeh_graph.nodes[node]['color'] = to_hex(self.my_palette(0))
        else:
            color_nodes(self.bokeh_graph, variable, self.my_palette)

        self.graph_renderer.node_renderer.data_source.data['color'] = [self.bokeh_graph.nodes[n]['color'] 
                                                                       for n in self.bokeh_graph.nodes]    
        
        
    