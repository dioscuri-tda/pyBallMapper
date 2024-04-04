import numpy as np
import pandas as pd
import networkx as nx

import csv
from tqdm.notebook import tqdm

from matplotlib.colors import to_hex, to_rgb

from bokeh.plotting import figure, show

from bokeh.models import (
    BoxZoomTool,
    Circle,
    HoverTool,
    Plot,
    ResetTool,
    ColumnDataSource,
    TapTool,
    WheelZoomTool,
    PanTool,
    SaveTool,
    Range1d,
    MultiLine,
    ColorBar,
    LinearColorMapper,
)

from bokeh.plotting import from_networkx, figure, curdoc


# creates a nx graph that bokeh can plot
def _create_bokeh_graph(G, my_palette, MIN_SIZE=7, MAX_SIZE=20):
    MAX_NODE_SIZE = max([G.nodes[node]["size"] for node in G.nodes])

    for node in G.nodes:
        # rescale the size for display
        G.nodes[node]["size rescaled"] = (
            MAX_SIZE * G.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SIZE
        )

        G.nodes[node]["color"] = to_hex(my_palette(0))

    for edge in G.edges:
        G.edges[edge]["color"] = to_hex((0, 0, 0))

    return G


def _color_nodes(
    G, my_variable, my_palette, MIN_VALUE=10000, MAX_VALUE=-10000, logscale=False
):
    for node in G.nodes:
        if G.nodes[node][my_variable] > MAX_VALUE:
            MAX_VALUE = G.nodes[node][my_variable]
        if G.nodes[node][my_variable] < MIN_VALUE:
            MIN_VALUE = G.nodes[node][my_variable]

    for node in G.nodes:
        if not pd.isna(G.nodes[node][my_variable]):
            color_id = (G.nodes[node][my_variable] - MIN_VALUE) / (
                MAX_VALUE - MIN_VALUE
            )
            if logscale:
                color_id = (
                    np.log10(G.nodes[node][my_variable]) - np.log10(MIN_VALUE)
                ) / (np.log10(MAX_VALUE) - np.log10(MIN_VALUE))
            G.nodes[node]["color"] = to_hex(my_palette(color_id))
        else:
            G.nodes[node]["color"] = "black"

    print(
        "color by variable {} \nMIN_VALUE: {:.3f}, MAX_VALUE: {:.3f}".format(
            my_variable, MIN_VALUE, MAX_VALUE
        )
    )

    return G, MIN_VALUE, MAX_VALUE


class graph_GUI:
    def __init__(
        self,
        graph,
        my_palette,
        tooltips_variables=[],
        figsize=(800, 600),
        output_format="svg",
        render_seed=42,
        render_iterations=100,
        MIN_SIZE=7,
        MAX_SIZE=20,
    ):
        """Create a Bokeh plot rapresenting the BallMapper graph.

        Parameters
        -----------
        graph : NetworkX graph 
            The Graph attribute of a BallMapper object.

        my_palette : matplotlib palette
            The color palette used to color the nodes.

        tooltips_variables : list of strings, default=None
            Which variables to show in the mouse hovertool.

        figsize : tuple, default=(800, 600)
            The figure size.

        output_format: string, default='svg'
            The format in which the save button saves the plot.

        render_seed: int, default=42
            Seed for the networkx spring_layout.

        render_iterations: int, default=100
            Number of iterations for the networkx spring_layout.

        MIN_SIZE: int, default=7
            Minimal size of the graph nodes.

        MAX_SIZE; int, default=20
            Maximal size of the graph nodes.

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

        """
        self.my_palette = my_palette

        self.bokeh_graph = _create_bokeh_graph(graph, my_palette, MIN_SIZE, MAX_SIZE)

        self.plot = figure(
            width=figsize[0],
            height=figsize[1],
            x_range=Range1d(-1, 1),
            y_range=Range1d(-1, 1),
            output_backend=output_format,
        )

        self.plot.xaxis.visible = False
        self.plot.yaxis.visible = False

        self.plot.xgrid.visible = False
        self.plot.ygrid.visible = False

        tooltips = [("index", "@index"), ("size", "@size")]

        tooltips += [(name, "@{}".format(name)) for name in tooltips_variables]

        node_hover_tool = HoverTool(tooltips=tooltips)
        zoom_tool = WheelZoomTool()
        self.plot.add_tools(
            PanTool(), node_hover_tool, zoom_tool, ResetTool(), SaveTool()
        )
        self.plot.toolbar.active_scroll = zoom_tool

        self.graph_renderer = from_networkx(
            graph=self.bokeh_graph,
            layout_function=nx.spring_layout,
            seed=render_seed,
            scale=1,
            center=(0, 0),
            iterations=render_iterations,
        )

        self.graph_renderer.node_renderer.glyph.update(
            size="size rescaled",
            fill_color="color",
            fill_alpha=0.8,
        )

        self.graph_renderer.edge_renderer.glyph.update(
            line_color="color", line_alpha=0.8, line_width=1
        )

        self.plot.renderers.append(self.graph_renderer)

    # this function changes the coloring of the nodes
    def color_by_variable(
        self, variable, MIN_VALUE=np.inf, MAX_VALUE=-np.inf, logscale=False
    ):
        """Color the BallMapper Bokeh plot nodes using a specific variable.

        Coloring data can be added using the `BallMapper.add_coloring()` method.
        `MIN_VALUE` and `MAX_VALUE` are computed automatically, but lower (resp. higher) \
        values can be manually specified. 
    
        Parameters
        ----------
        variable : string
            Which coloring variable to use.

        MIN_VALUE : float, default=np.inf
            Minimum value for the coloring palette.

        MAX_VALUE : float, default=-np.inf
            Maximum value for the coloring palette.

        logscale : bool, default=False
            Wheter to use a logscale coloring. If False, linear is used.

        """
        if variable == "None":
            for node in self.bokeh_graph.nodes:
                self.bokeh_graph.nodes[node]["color"] = to_hex(self.my_palette(0))
        else:
            _, MIN_VALUE, MAX_VALUE = _color_nodes(
                self.bokeh_graph,
                variable,
                self.my_palette,
                MIN_VALUE,
                MAX_VALUE,
                logscale,
            )

        self.graph_renderer.node_renderer.data_source.data["color"] = [
            self.bokeh_graph.nodes[n]["color"] for n in self.bokeh_graph.nodes
        ]

        return MIN_VALUE, MAX_VALUE

    def color_edges(self):
        """color edges by interpolating between the nodes' colors"""

        for edge in self.bokeh_graph.edges:
            c0 = np.array(to_rgb(self.bokeh_graph.nodes[edge[0]]["color"]))
            c1 = np.array(to_rgb(self.bokeh_graph.nodes[edge[1]]["color"]))

            self.bokeh_graph.edges[edge]["color"] = to_hex((c0 + c1) / 2)

        self.graph_renderer.edge_renderer.data_source.data["color"] = [
            self.bokeh_graph.edges[e]["color"] for e in self.bokeh_graph.edges
        ]

    def add_colorbar(self, num_ticks, low, high):
        """Add a colorbar to the right side of the Bokeh plot.

        Parameters
        ----------
        num_ticks : int
            Number of ticks, use a high number (100) for a continuos \
            looking colorbar.

        low : float
            Value at the bottom of the colorbar.

        high : float
            Value at the top of the colorbar.
        """
        color_mapper = LinearColorMapper(
            palette=[
                to_hex(self.my_palette(color_id))
                for color_id in np.linspace(0, 1, num_ticks)
            ],
            low=low,
            high=high,
        )
        color_bar = ColorBar(
            color_mapper=color_mapper,
            major_label_text_font_size="14pt",
            label_standoff=12,
        )

        self.plot.add_layout(color_bar, "right")


def kmapper_visualize(bm, coloring_df, path_html="output.html", title=None, **kwargs):
    """leverages kepler-mapper visualization tool to produce an interactive html.
    https://kepler-mapper.scikit-tda.org

    Parameters
    ----------
    bm : BallMapper
        the BallMapper graph to plot
    coloring_df : pandas.DataFrame
        pandas dataframe of shape (n_samples, n_coloring_function)
    path_html : str, optional
        the output file, by default 'output.html'
    title : string, optional
        title to be displayed in the top bar, by default None
    """
    import kmapper as km
    from collections import defaultdict

    mapper = km.KeplerMapper(verbose=0)

    graph = {}
    graph["nodes"] = defaultdict(list)
    for n in bm.points_covered_by_landmarks:
        graph["nodes"][n] = bm.points_covered_by_landmarks[n]

    graph["links"] = defaultdict(list)
    for u, v in bm.Graph.edges:
        graph["links"][u].append(v)

    graph["meta_data"] = {}

    mapper.visualize(
        graph,
        path_html=path_html,
        color_values=coloring_df.to_numpy(),
        color_function_name=coloring_df.columns.tolist(),
        title=title,
        **kwargs
    )
