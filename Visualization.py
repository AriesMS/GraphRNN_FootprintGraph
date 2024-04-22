import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON file
with open('generated_graphs.json', 'r') as f:
    graph_data = json.load(f)

# This is assuming that your JSON has a list of edges defined for each graph
# along with node features which are (x,y) coordinates
for graph_id, graph_info in graph_data.items():
    G = nx.Graph()
    
    # If you have edge information
    edges = graph_info.get('TopologyList', [])
    for edge in edges:
        G.add_edge(edge[0], edge[1])  # Assuming edge is a tuple (node1, node2)

    # Add nodes with the coordinates as attributes
    for node_index, (x, y) in enumerate(graph_info['Vertices']):
        G.add_node(node_index, pos=(x, y))

    # Get positions from node data
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    
    # Show the plot
    plt.show()