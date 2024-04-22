import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_undirected
import networkx as nx
import os

file_path = r'C:\Users\yangy\OneDrive\文档\RA\Models\GRAPHRNNTEST\BuildingFootprints_Normalised_24k.json'

# Read the JSON file
with open(file_path, 'r') as file:
    data_json = json.load(file)

# Build a mapping from unique labels to integers
unique_labels = set()
for value in data_json.values():
    unique_labels.update(value['LU_DESC_Typology'])
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Process the JSON data into PyTorch Geometric Data objects with numerical labels
graph_list = []
for key, value in data_json.items():
    g = nx.Graph()
    g.add_edges_from(value['TopologyList'])
    node_features = torch.tensor(value['Vertices'], dtype=torch.float)
    
    # Convert the typology labels to numerical form
    typology_indices = torch.tensor([label_to_index[typology] for typology in value['LU_DESC_Typology']], dtype=torch.long)
    
    edge_index = torch.tensor(value['TopologyList']).t().contiguous()
    edge_index = to_undirected(edge_index)  # Make sure every edge is bidirectional

    data = Data(x=node_features, edge_index=edge_index, y=typology_indices)
    graph_list.append(data)

# Save the label to index mapping for later use (e.g., in generation script)
with open('label_to_index.json', 'w') as f:
    json.dump(label_to_index, f, indent=4)

print("Labels encoded and graph data objects created.")