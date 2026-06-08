import torch
import streamlit as st
from streamlit_agraph import Node, Edge

def load_pytorch_data(file_path):
    # weights_only=False is required when loading custom Python lists (strings)
    return torch.load(file_path, weights_only=False, map_location="cpu")

def create_ui_elements(data):
    ui_nodes = []
    ui_edges = []
    
    # Extract names
    node_names = data["nodes"].get("names", [f"ID {i}" for i in range(data["metadata"]["num_nodes"])])
    edge_names = data["edges"].get("names", [f"ID {i}" for i in range(data["edges"]["indices"].shape[0])])
    
    # Process Nodes
    node_selected_mask = data["nodes"]["selected"]
    for idx, is_selected in enumerate(node_selected_mask):
        if is_selected.item():  
            ui_nodes.append(
                Node(id=str(idx), label=node_names[idx], size=25, shape="dot", color="#4CAF50")
            )
            
    # Process Edges
    edge_selected_mask = data["edges"]["selected"]
    edge_indices = data["edges"]["indices"]
    for idx, is_selected in enumerate(edge_selected_mask):
        if is_selected.item():
            src = int(edge_indices[idx][0].item())
            dst = int(edge_indices[idx][1].item())
            
            if node_selected_mask[src] and node_selected_mask[dst]:
                ui_edges.append(
                    Edge(source=str(src), target=str(dst), id=str(idx), label=edge_names[idx], color="#888888")
                )
                
    return ui_nodes, ui_edges