""" Module to test DrawTrees """
from networkx.drawing.nx_pydot import read_dot
from test.tools import TempTestDir
import os
from jet_tools import DrawTrees, FormShower
#from ipdb import set_trace as st


def test_DotGraph():
    # to avoid the internal workings, read the graph in as a network x graph
    def read_write(dot):
        with TempTestDir("tst") as dir_name:
            path = os.path.join(dir_name, "graph.dot")
            with open(path, 'w') as graph_file:
                graph_file.write(str(dot))
            graph = read_dot(path)
        return graph
    # empty graph
    dot = DrawTrees.DotGraph()
    graph = read_write(dot)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    # add a single node
    dot.addNode(0, 'cat')
    graph = read_write(dot)
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    assert graph.nodes['0']['label'] == '"cat"'
    # add a second coloured nodes
    dot.addNode(1, "dog", "red", "house")
    graph = read_write(dot)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    assert graph.nodes['1']['label'] == '"dog"'
    assert graph.nodes['1']['fillcolor'] == 'red'
    assert graph.nodes['1']['shape'] == 'house'
    # put an edge betwwen them
    dot.addEdge(0, 1)
    graph = read_write(dot)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert ('0', '1', 0) in graph.edges
    # now the legend
    def read_write_legend(dot):
        with TempTestDir("tst") as dir_name:
            path = os.path.join(dir_name, "graph.dot")
            with open(path, 'w') as graph_file:
                graph_file.write(dot.legend)
            graph = read_dot(path)
        return graph
    # empty
    ledg = read_write_legend(dot)
    assert len(ledg.nodes) == 0
    assert len(ledg.edges) == 0
    # add a coloured
    dot.addLegendNode(1, "dog", "red", "house")
    ledg = read_write_legend(dot)
    assert len(ledg.nodes) == 1
    assert len(ledg.edges) == 0
    assert ledg.nodes['1']['label'] == '"dog"'
    assert ledg.nodes['1']['fillcolor'] == 'red'
    assert ledg.nodes['1']['shape'] == 'house'


def test_from_shower():
    # will need an eventwise with Parents, Children, MCPID
    # layer     -1  0       1    1      -1   2   2          3   3   3   -1
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [0,2, 3], [5], [6, 5, 4], [], [],     [7, 8, 9], [], [], []]
    parents =  [[1], [],     [1], [1],    [3], [2, 3], [3],       [6],[6],[6]]
    mcpid =    [4,  5,      5,   3,      2,  1,      -5,        -1, 7,  11]
    n_nodes = len(children)
    shower = FormShower.Shower(list(range(n_nodes)), parents, children, mcpid)
    dot = DrawTrees.DotGraph(shower)
    with TempTestDir("tst") as dir_name:
        path = os.path.join(dir_name, "graph.dot")
        with open(path, 'w') as graph_file:
            graph_file.write(str(dot))
        graph = read_dot(path)
    assert len(graph.nodes) == n_nodes
    assert len(graph.edges) == 10


# TODO - test the parts of this that work to make the detector layer etc.
