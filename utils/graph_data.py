import torch
import streamlit as st
from streamlit_agraph import Node, Edge

def load_pytorch_data(file_path):
    # weights_only=False is required when loading custom Python lists (strings)
    return torch.load(file_path, weights_only=False, map_location="cpu")

def create_ui_elements(data, filter_mode="selected", node_thresh=0.0, edge_thresh=0.0):
    ui_nodes = []
    ui_edges = []
    
    # Extract names
    node_names = data["nodes"].get("names", [f"ID {i}" for i in range(data["metadata"]["num_nodes"])])
    edge_names = data["edges"].get("names", [f"ID {i}" for i in range(data["edges"]["indices"].shape[0])])
    
    # Determine which mask to use based on the UI settings
    if filter_mode == "selected":
        node_mask = data["nodes"]["selected"]
        edge_mask = data["edges"]["selected"]
    else: # filter_mode == "confidence"
        node_mask = data["nodes"]["confidences"] >= node_thresh
        edge_mask = data["edges"]["confidences"] >= edge_thresh
    
    # Process Nodes
    for idx, is_valid in enumerate(node_mask):
        if is_valid.item():  
            conf_val = data["nodes"]["confidences"][idx].item()
            ui_nodes.append(
                Node(
                    id=str(idx), 
                    label=node_names[idx], 
                    title=f"Conf: {conf_val:.2f}", # Shows on mouse hover
                    size=25, 
                    shape="dot", 
                    color="#4CAF50"
                )
            )
            
    # Process Edges
    edge_indices = data["edges"]["indices"]
    for idx, is_valid in enumerate(edge_mask):
        if is_valid.item():
            src = int(edge_indices[idx][0].item())
            dst = int(edge_indices[idx][1].item())
            
            # Draw edge only if both source and target nodes are also passing the filter
            if node_mask[src] and node_mask[dst]:
                conf_val = data["edges"]["confidences"][idx].item()
                ui_edges.append(
                    Edge(
                        source=str(src), 
                        target=str(dst), 
                        id=str(idx), 
                        label=edge_names[idx], 
                        title=f"Conf: {conf_val:.2f}", # Shows on mouse hover
                        color="#888888"
                    )
                )
                
    return ui_nodes, ui_edges