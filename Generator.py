import torch
import json
from Model import GraphRNN


node_feature_dim = 2  # Assuming 2D coordinates for nodes
rnn_hidden_dim = 128
model = GraphRNN(node_feature_dim, rnn_hidden_dim)  # Initialize the model with the same parameters
checkpoint = torch.load(r'D:\Docs\New\RA\Models\Models\GRAPHRNNTEST2\model_checkpoint.pth')

# Load the state dict specifically for the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

print("Model loaded successfully for generation!")
def generate_graph(model, initial_node_features, max_nodes, label_idx):
    model.eval()  # Set the model to evaluation mode
    generated_graph = [initial_node_features]
    h = None
    for _ in range(max_nodes - 1):
        input_node = generated_graph[-1]
        with torch.no_grad():
            output, h = model(input_node, label_idx, h)  # Incorporate label_idx in generation
        new_node = torch.round(output).detach()  # Assuming the node feature space is discrete
        generated_graph.append(new_node)
    return generated_graph

# Load label to index mapping
with open('label_to_index.json', 'r') as f:
    label_to_index = json.load(f)

# Define the label you want to condition on
label = "SPORTS & RECREATION_residential"  # This should be a label from your label_to_index mapping
label_idx = torch.tensor([label_to_index[label]], dtype=torch.long)  # Convert label to tensor

max_nodes = 10
# Generate 10 graphs conditioned on the label
generated_graphs = []
for _ in range(10):
    initial_node_features = torch.tensor([[0.0, 0.0]])  # Example starting node features
    new_graph = generate_graph(model, initial_node_features, max_nodes, label_idx)
    generated_graphs.append(new_graph)

# Convert generated graphs to a dataset-like dictionary
dataset_dict = {}
for i, graph in enumerate(generated_graphs):
    # Here we assume graph is a list of tensors representing node features
    # You would need to convert this to the same format as your input dataset
    graph_data = {
        "TopologyList": [],  # You would need to generate or define edges as well
        "Vertices": [node.tolist() for node in graph],
        "LU_DESC_Typology": ["SPORTS & RECREATION_residential"]  # This label is just an example
    }
    dataset_dict[str(i)] = graph_data

# Save the dictionary in JSON format
with open('generated_graphs.json', 'w') as f:
    json.dump(dataset_dict, f, indent=4)

print("Generated graphs saved to generated_graphs.json")

