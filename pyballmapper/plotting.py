import numpy as np
import pandas as pd
import networkx as nx

from matplotlib.colors import to_hex, to_rgb

import networkx as nx

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


## stole from https://github.com/IBM/matilda/blob/main/matilda/mapper.py
def generate_partitions(node_contents, original_element_values):
    """
    Given collections X_i of sets Y_ij and a function U Y_ij -> Z, computes for each i the
    count of elements in X_i that map to a given value in Z.


    node_contents:
        dict of lists such that `node_contents[node]` is a list of all elements contained in `node`.
    original_element_values:
        dict of values associated to each of the original set of elements. Not necessarily numeric.
    """
    import collections

    partitions = {}
    for k, v in node_contents.items():
        partitions[k] = dict(
            collections.Counter([original_element_values[x] for x in v])
        )
    return partitions


def pie_graph_plot(
    partitions,
    g=None,
    nodes=None,
    edges=None,
    graph_layout=None,
    radius=0.01,
    node_labels=None,
    node_scaling="old",
    palette=None,
    background_fill_color="white",
    title=None,
    match_aspect=True,
    edge_width=1,
    edge_color="black",
    plot_height=600,
    plot_width=600,
    sizing_mode="fixed",
    show_node_labels=False,
    outline_line_width=1,
    outline_line_color="black",
):
    """
    Produces a bokeh plot of a graph where each node is represented by a pie chart.

    Parameters
    ----------

    g:
        networkx graph

    partitions:
        dict of dicts such that partitions[node][class] == portion of class at node.

    graph_layout:
        dict of size 2 arrays such that graph_layout[node] == position of node in plane.

    nodes:
        list of nodes. If used, parameter `g` is ignored.

    edges:
        list of edges. Used in conjunction with `nodes`. If used, parameter `g` is ignored.

    radius:
        Radius of nodes. Upper bound on the radii of nodes if `node_scaling` is set to True.

    node_labels:
        list of text labels for nodes. If provided, must be in same order as iteration
        over nodes in graph.

    node_scaling:
        If set to True, nodes are scaled from
            `((smallest partition size)/(largest partition size))*radius` to `radius`, with
        a minimum size of nodes of radius/10 for enhanced viewability.

    palette:
        dict with keys nodes and values bokeh colors, or list with colors with same size as
        number of classes present in `partitions`.

    title:
        Plot's title.

    edge_width:
        Graph's edge thickness.

    edge_color:
        Graph's edge color. Used only if `sizing_mode` is "fixed".

    show_node_labels:
        If set to `True`, adds node labels to the plot.

    match_aspect:
        Preserves aspect ratio when changing size. Passed as-is to bokeh's figure initializer.

    plot_height:
        Plot's height in pixels. Used only if `sizing_mode` is "fixed". Passed as-is to bokeh's figure initializer.

    plot_width:
        Plot's width in pixels. Passed as-is to bokeh's figure initializer.

    sizing_mode:
        One of "fixed", "stretch_both", "stretch_width", "stretch_height", "scale_width", "scale_height".
        Passed as-is to bokeh's figure initializer.

    outline_line_width:
        Thickness of the frame of the plot. Passed as-is to bokeh's figure initializer.

    outline_line_color:
        Color of the frame of the plot. Passed as-is to bokeh's figure initializer.

    background_fill_color:
        bokeh color to use for plot background. Passed as-is to bokeh's figure initializer.


    """
    import networkx, numpy, pathlib, os, warnings
    import bokeh.plotting
    from bokeh.models import HoverTool, CustomAction
    from bokeh.models import ColumnDataSource, CustomJS, LabelSet

    if g is None and nodes is None and edges is None:
        raise ValueError("At least one of g, nodes, or edges must be provided.")
    if nodes is not None or edges is not None:
        g = networkx.Graph()
        if nodes is not None:
            g.add_nodes_from(nodes)
        if edges is not None:
            g.add_edges_from(edges)
    nodes_list = list(g.nodes())
    if node_labels is None:
        node_labels = list(map(str, g.nodes()))
    if not all("pos" in v.keys() for v in g.nodes.values()):
        if graph_layout is None:
            raise ValueError(
                "Graph needs to have node position information stored in `pos` attribute "
                " (you can specify position information with graph_layout)."
            )
        else:
            for x in nodes_list:
                g.nodes[x]["pos"] = graph_layout[x]
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    p, node_x, node_y = init_bokeh_figure(
        g,
        background_fill_color,
        title,
        match_aspect,
        edge_width,
        edge_color,
        plot_height,
        plot_width,
        sizing_mode,
        outline_line_width,
        outline_line_color,
    )
    nodes_enum = {n: i for i, n in enumerate(nodes_list)}
    node_sizes = numpy.array(
        [sum([vv for kk, vv in partitions[n].items()]) for n in nodes_list]
    )

    if node_scaling == "old":
        node_sizes_min = numpy.amin(node_sizes)
        node_sizes_max = numpy.amax(node_sizes)
        radii = numpy.interp(
            node_sizes,
            (node_sizes_min, node_sizes_max),
            (max(node_sizes_min * radius / node_sizes_max, radius / 10), radius),
        )
    elif node_scaling == "new":
        MAX_NODE_SIZE = numpy.amax(node_sizes)
        MAX_SIZE = 20
        MIN_SIZE = 7
        # rescale the size for display
        radii = [radius * (MAX_SIZE * s / MAX_NODE_SIZE + MIN_SIZE) for s in node_sizes]
    else:
        radii = [radius] * len(node_sizes)
    all_node_sources = []
    factors = sorted(list(set.union(*[set(v.keys()) for v in partitions.values()])))
    factors_enum = {f: i for i, f in enumerate(factors)}
    if palette is None:
        palette = bokeh.palettes.d3["Category10"][max(3, len(factors))]
    factor_sizes = numpy.zeros((len(factors), len(nodes_list)))
    for n in nodes_list:
        for kk, vv in partitions[n].items():
            factor_sizes[factors_enum[kk], nodes_enum[n]] = vv
    end_angles = 2 * numpy.pi * numpy.cumsum(factor_sizes, axis=0) / node_sizes
    start_angles = numpy.roll(end_angles, 1, axis=0)
    start_angles[0, :] = 0
    # NOW PLOT ALL NODES AS MINI PIE CHARTS
    legend_items = []
    for l in factors:
        sourced = {}
        sourced["x"] = node_x
        sourced["y"] = node_y
        sourced["radius"] = radii
        # DEALS WITH CASE OF ONE CLASS PER NODE AND BOKEH'S BEHAVIOR
        for j in range(start_angles[factors_enum[l]].shape[0]):
            if (
                start_angles[factors_enum[l], j] == 0
                and end_angles[factors_enum[l], j] >= 2 * numpy.pi - 1e-3
            ):
                end_angles[factors_enum[l], j] = 2 * numpy.pi - 1e-3
        sourced["start_angle"] = start_angles[factors_enum[l]]
        sourced["end_angle"] = end_angles[factors_enum[l]]
        sourced["factor_size"] = factor_sizes[factors_enum[l]]
        sourced["factor_label"] = [str(l) for _ in node_x]
        sourced["node_size"] = node_sizes
        sourced["node_label"] = node_labels
        if isinstance(palette, dict):
            sourced["color"] = [palette[l]] * len(node_x)
        else:
            sourced["color"] = [palette[factors_enum[l]]] * len(node_x)
        source = ColumnDataSource(sourced)
        all_node_sources.append(source)
        legend_items += [
            (
                l,
                p.wedge(
                    x="x",
                    y="y",
                    start_angle="start_angle",
                    end_angle="end_angle",
                    source=source,
                    radius="radius",
                    fill_color="color",
                    alpha=0.9,
                    line_width=0,
                    legend_label=str(l),
                ),
            )
        ]
        p.arc(
            x="x",
            y="y",
            start_angle="start_angle",
            end_angle="end_angle",
            source=source,
            radius="radius",
            line_color="color",
            alpha=1,
        )
        labels = LabelSet(
            x="x",
            y="y",
            text="node_label",
            x_offset=5,
            y_offset=5,
            source=source,
            # render_mode="canvas",
        )
    tooltips = (
        "<div>Class <b>@factor_label</b> in node: <b>@factor_size</b> <br>"
        "Total in node: <b>@node_size</b> <br>"
        "Node Label: <b>@node_label</b>"
    )
    hover_tool = HoverTool(tooltips=tooltips, renderers=[x for _, x in legend_items])
    p.add_tools(hover_tool)
    if show_node_labels:
        p.add_layout(labels)

    params_node_size_increase_tool = dict(
        # icon=pathlib.Path(os.path.join(dir_path, "node_size_inc_tool_icon.png")),
        description="Increase node size",
        callback=CustomJS(
            args=dict(sources=all_node_sources), code=node_increase_size_js_code
        ),
    )
    try:
        node_size_increase_tool = CustomAction(**params_node_size_increase_tool)
    except ValueError as e:
        warnings.warn(
            "\nWorkaround for bokeh<2.4 exception:\n" + str(e), RuntimeWarning
        )
        # params_node_size_increase_tool["icon"] = os.path.join(
        #     dir_path, "node_size_inc_tool_icon.png"
        # )
        node_size_increase_tool = CustomAction(**params_node_size_increase_tool)
    params_node_size_decrease_tool = dict(
        # icon=pathlib.Path(os.path.join(dir_path, "node_size_dec_tool_icon.png")),
        description="Decrease node size",
        callback=CustomJS(
            args=dict(sources=all_node_sources), code=node_decrease_size_js_code
        ),
    )
    try:
        node_size_decrease_tool = CustomAction(**params_node_size_decrease_tool)
    except ValueError as e:
        warnings.warn(
            "\nWorkaround for bokeh<2.4 exception:\n" + str(e), RuntimeWarning
        )
        # params_node_size_decrease_tool["icon"] = os.path.join(
        #     dir_path, "node_size_dec_tool_icon.png"
        # )
        node_size_decrease_tool = CustomAction(**params_node_size_decrease_tool)
    p.add_tools(node_size_decrease_tool)
    p.add_tools(node_size_increase_tool)
    return p


def init_bokeh_figure(
    g,
    background_fill_color,
    title,
    match_aspect,
    edge_width,
    edge_color,
    plot_height,
    plot_width,
    sizing_mode,
    outline_line_width,
    outline_line_color,
):
    from bokeh.models import BoxZoomTool, WheelZoomTool
    import bokeh.plotting, networkx

    box_zoom_tool = BoxZoomTool(match_aspect=False)
    wheel_zoom_tool = WheelZoomTool(speed=0.002, zoom_on_axis=False)
    p = bokeh.plotting.figure(
        title=title,
        toolbar_location="right",
        tools=[box_zoom_tool, wheel_zoom_tool, "pan", "reset", "save"],
        active_scroll=wheel_zoom_tool,
        match_aspect=match_aspect,
        height=plot_height,
        width=plot_width,
        sizing_mode=sizing_mode,
        outline_line_width=outline_line_width,
        outline_line_color=outline_line_color,
        background_fill_color=background_fill_color,
    )
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    pos = networkx.get_node_attributes(g, "pos")
    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
        # HACKY SOLUTION TO GET EDGES INSTEAD OF ONE VERY LONG LINE
        edge_x.append(float("nan"))
        edge_y.append(float("nan"))

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    p.line(edge_x, edge_y, line_width=edge_width, color=edge_color)
    return p, node_x, node_y


node_increase_size_js_code = """
                                var j = 0;
                                for(j=0;j<sources.length;j++)
                                {
                                    var radius = sources[j].data.radius;
                                    var i = 0;
                                    for(i=0; i<radius.length; i++)
                                    {
                                        radius[i] = radius[i] * 1.25
                                    }
                                    sources[j].change.emit()  
                                }
                            """

node_decrease_size_js_code = """
                                var j = 0;
                                for(j=0;j<sources.length;j++)
                                {
                                    var radius = sources[j].data.radius;
                                    var i = 0;
                                    for(i=0; i<radius.length; i++)
                                    {
                                        radius[i] = radius[i] / 1.25
                                    }
                                    sources[j].change.emit()  
                                }
                            """
