import numpy as np
import pandas as pd
import networkx as nx

import csv
from tqdm.notebook import tqdm

from matplotlib.colors import to_hex

from bokeh.plotting import figure, show

from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          Plot, ResetTool,
                          ColumnDataSource,
                          TapTool, WheelZoomTool, PanTool,
                          SaveTool, Range1d, MultiLine,
                          ColorBar, LinearColorMapper
                          )

from bokeh.plotting import from_networkx, figure, curdoc




# creates a nx graph that bokeh can plot
def create_bokeh_graph(G,
                       my_palette,
                       MIN_SCALE = 7,
                       MAX_SCALE = 20
                       ):

    

    MAX_NODE_SIZE = max([G.nodes[node]['size'] for node in G.nodes])

    for node in G.nodes:
        # rescale the size for display
        G.nodes[node]['size rescaled'] = MAX_SCALE*G.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE

        G.nodes[node]['color'] = to_hex(my_palette(0))

    return G


def color_nodes(G, my_variable, my_palette, MIN_VALUE = 10000, MAX_VALUE = -10000, logscale=False):

    for node in G.nodes:
        if G.nodes[node][my_variable] > MAX_VALUE:
            MAX_VALUE = G.nodes[node][my_variable]
        if G.nodes[node][my_variable] < MIN_VALUE:
            MIN_VALUE = G.nodes[node][my_variable]

    for node in G.nodes:
        if not pd.isna(G.nodes[node][my_variable]):
            color_id = (G.nodes[node][my_variable] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
            if logscale:
                color_id = (np.log10(G.nodes[node][my_variable]) - np.log10(MIN_VALUE)) \
                            / (np.log10(MAX_VALUE) - np.log10(MIN_VALUE))
            G.nodes[node]['color'] = to_hex(my_palette(color_id))
        else:
            G.nodes[node]['color'] = 'black'


    print('color by variable {} \nMIN_VALUE: {:.3f}, MAX_VALUE: {:.3f}'.format(my_variable, MIN_VALUE, MAX_VALUE))

    return G, MIN_VALUE, MAX_VALUE




class graph_GUI():
    def __init__(self, graph, my_palette, 
                 tooltips_variables=[],
                 figsize = (800, 600),
                 output_format='svg',
                 render_seed=42,
                 render_iterations=100,
                 MIN_SCALE = 7,
                 MAX_SCALE = 20):

        self.my_palette = my_palette

        self.bokeh_graph = create_bokeh_graph(graph, my_palette, MIN_SCALE, MAX_SCALE)

        self.plot = Plot(plot_width=figsize[0], plot_height=figsize[1],
                    x_range=Range1d(-1, 1), y_range=Range1d(-1, 1),
                    output_backend=output_format)
                    #sizing_mode="stretch_both")

        tooltips=[("index", "@index"), ("size", "@size")]

        tooltips += [(name, '@{}'.format(name)) for name in tooltips_variables]

        node_hover_tool = HoverTool(tooltips=tooltips)
        zoom_tool = WheelZoomTool()
        self.plot.add_tools(PanTool(), node_hover_tool, zoom_tool,
                            ResetTool(), SaveTool())
        self.plot.toolbar.active_scroll = zoom_tool

        self.graph_renderer = from_networkx(self.bokeh_graph, nx.spring_layout,
                                            seed=render_seed, scale=1, center=(0, 0),
                                            iterations=render_iterations)
                                            #k= 10/np.sqrt(len(self.bokeh_graph.nodes)),
                                            

        # nodes
        self.graph_renderer.node_renderer.glyph = Circle(size='size rescaled',
                                                      fill_color='color',
                                                      fill_alpha=0.8)

        # edges
        self.graph_renderer.edge_renderer.glyph = MultiLine(line_color='black',
                                                         line_alpha=0.8, line_width=1)

        self.plot.renderers.append(self.graph_renderer)



    # this function changes the coloring of the nodes
    def color_by_variable(self, variable, MIN_VALUE = 0, MAX_VALUE = 0, logscale=False):

        if variable == 'None':
            for node in self.bokeh_graph.nodes:
                self.bokeh_graph.nodes[node]['color'] = to_hex(self.my_palette(0))
        else:
            _, MIN_VALUE, MAX_VALUE = color_nodes(self.bokeh_graph, variable, self.my_palette,
                                                  MIN_VALUE, MAX_VALUE, logscale)

        self.graph_renderer.node_renderer.data_source.data['color'] = [self.bokeh_graph.nodes[n]['color']
                                                                       for n in self.bokeh_graph.nodes]

        return MIN_VALUE, MAX_VALUE
    
    
    def add_colorbar(self, num_ticks, low=None, high=None):
        color_mapper = LinearColorMapper(palette=[to_hex(self.my_palette(color_id)) 
                                                  for color_id in np.linspace(0, 1, num_ticks)], 
                                         low=low, high=high)
        color_bar = ColorBar(color_mapper=color_mapper, major_label_text_font_size='14pt',
                             label_standoff=12)

        self.plot.add_layout(color_bar, 'right')
