import torch
import torch.nn as nn
from torch.nn import RNNCell
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch_geometric.data import Data
from Dataset import graph_list
import random

if torch.cuda.is_available():
    print ('Using GPU')
else:
    print ('Using CPU')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.shuffle(graph_list)  # Shuffle the dataset before splitting
split_index = int(0.8 * len(graph_list))
train_graph_list = graph_list[:split_index]
val_graph_list = graph_list[split_index:]

embedding_dim = 64
class GraphRNN(nn.Module):
    def __init__(self, node_feature_dim, rnn_hidden_dim, num_labels, embedding_dim):
        super(GraphRNN, self).__init__()
        self.label_embedding = nn.Embedding(num_labels, embedding_dim)
        self.node_rnn = nn.RNNCell(input_size=node_feature_dim + embedding_dim, hidden_size=rnn_hidden_dim)
        self.node_mlp = nn.Linear(rnn_hidden_dim, node_feature_dim)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * rnn_hidden_dim, rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label_idx, h):
        label_embedding = self.label_embedding(label_idx).view(1, -1)
        x_with_label = torch.cat([x, label_embedding], dim=1)
        h_next = self.node_rnn(x_with_label, h)
        out_node = self.node_mlp(h_next)
        return out_node, h_next

    def predict_edge(self, h_new, h_existing):
        combined_h = torch.cat([h_new.repeat(h_existing.size(0), 1), h_existing], dim=1)
        #print(f"Combined hidden states shape: {combined_h.shape}")
        edge_probs = self.edge_classifier(combined_h)
        #print(f"Edge probabilities shape: {edge_probs.shape}")
        return edge_probs


node_feature_dim = 2  # Assuming 2D coordinates for nodes
rnn_hidden_dim = 128
model = GraphRNN(node_feature_dim, rnn_hidden_dim,num_labels=159,embedding_dim=embedding_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
criterion_node = torch.nn.MSELoss()  # Example: Mean Squared Error Loss
criterion_edge = torch.nn.BCELoss()  # Example: Binary Cross-Entropy Loss for edge prediction

def train(model, graph, optimizer, criterion_node, criterion_edge, device):
    model.train()
    optimizer.zero_grad()
    h = None
    node_loss = 0
    edge_loss = 0

    nodes = graph.x.to(device)
    edges = graph.edge_index.to(device)  # Edge index tensor
    label_idx = graph.y.to(device)
    h_states = []

    for i in range(graph.num_nodes):
        input_node = nodes[i].view(1, -1)
        h = h if h is not None else torch.zeros(1, model.node_rnn.hidden_size, device=device)
        output_node, h = model(input_node, label_idx, h)
        h_states.append(h)

        # Start predicting edges from the second node
        if i > 0:
            h_new = h_states[i]
            h_existing = torch.stack(h_states[:i], dim=0).squeeze(1)  # Stack to get previous hidden states

            # Predict edges for the current node against all previous nodes
            edge_probs = model.predict_edge(h_new, h_existing).view(-1)

            # Get actual edges from the graph
            actual_edges = torch.zeros(i, dtype=torch.float, device=device)
            for j in range(i):
                # Check for both possible directions since the graph is undirected
                if edges[0].eq(j).logical_and(edges[1].eq(i)).any() or edges[0].eq(i).logical_and(edges[1].eq(j)).any():
                    actual_edges[j] = 1.0

            actual_edges = actual_edges.view(-1)

            # Check that the sizes match
            if edge_probs.numel() != actual_edges.numel():
                raise ValueError(f"Size mismatch between predicted and actual edges: predicted {edge_probs.numel()}, actual {actual_edges.numel()}")

            # Calculate the loss on the edges
            edge_loss += criterion_edge(edge_probs, actual_edges)

    total_loss = node_loss + edge_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

def validate(model, graph, criterion_node, criterion_edge, device):
    model.eval()
    h = None
    node_loss = 0
    edge_loss = 0

    nodes = graph.x.to(device)
    edges = graph.edge_index.to(device)
    label_idx = graph.y.to(device)
    h_states = [torch.zeros(1, model.node_rnn.hidden_size, device=device)]  # Initial hidden state

    with torch.no_grad():
        for i in range(1, graph.num_nodes):
            input_node = nodes[i].view(1, -1)
            h = h if h is not None else h_states[0]
            output_node, h = model(input_node, label_idx, h)
            h_states.append(h)

            if i > 0:
                h_new = h_states[-1]
                h_existing = torch.stack(h_states[:-1], dim=0).squeeze(1)

                # Predict edges for the current node against all previous nodes
                edge_probs = model.predict_edge(h_new, h_existing).view(-1)

                # Get actual edges for the node i
                actual_edges = torch.zeros(i, dtype=torch.float, device=device)
                for j in range(i):
                    if edges[0].eq(j).logical_and(edges[1].eq(i)).any() or edges[0].eq(i).logical_and(edges[1].eq(j)).any():
                        actual_edges[j] = 1.0

                actual_edges = actual_edges.view(-1)

                # Make sure edge_probs has the same number of elements as actual_edges
                edge_probs = edge_probs[:i]
                actual_edges = actual_edges[:edge_probs.size(0)]

                # Compute the edge loss
                edge_loss += criterion_edge(edge_probs, actual_edges)

    # Calculate average losses
    avg_node_loss = node_loss / max(1, graph.num_nodes - 1)
    avg_edge_loss = edge_loss / max(1, graph.num_nodes - 1)
    total_loss = avg_node_loss + avg_edge_loss

    return total_loss

def check_improvement(old_loss, new_loss, tolerance=0.01):
    improvement = (old_loss - new_loss) / old_loss
    return improvement > tolerance

# Assume optimizer and criterion are already defined
# Example training loop
if __name__ == "__main__":

    num_epochs = 50
    best_val_loss = float('inf')

    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for graph in train_graph_list:
            train_loss += train(model, graph, optimizer, criterion_node, criterion_edge, device)  # Use correct criteria
        avg_train_loss = train_loss / len(train_graph_list)  # Normalize train loss

        val_loss = 0
        for graph in val_graph_list:
            val_loss += validate(model, graph, criterion_node, criterion_edge, device)  # Use correct criteria
        avg_val_loss = val_loss / len(val_graph_list)  # Normalize validation loss
    
        if check_improvement(best_val_loss, avg_val_loss):
            best_val_loss = avg_val_loss
            print(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: No significant improvement in validation loss.")
    
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Training loss': avg_train_loss,
        'Validation loss': avg_val_loss
        }, r'D:\Docs\New\RA\Models\Models\GRAPHRNNTEST\model_checkpoint.pth')
    print("Model saved successfully!")
