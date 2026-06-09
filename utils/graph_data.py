import torch
import streamlit as st
from streamlit_agraph import Node, Edge
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_pytorch_data(file_path):
    return torch.load(file_path, weights_only=False, map_location="cpu")

def create_ui_elements(data, filter_mode="selected", node_thresh=0.0, edge_thresh=0.0, active_node_idx=None, sim_norm_mode="relative"):
    ui_nodes = []
    ui_edges = []
    
    # Extract names
    node_names = data["nodes"].get("names", [f"ID {i}" for i in range(data["metadata"]["num_nodes"])])
    edge_names = data["edges"].get("names", [f"ID {i}" for i in range(data["edges"]["indices"].shape[0])])
    
    # Filter Masks
    if filter_mode == "selected":
        node_mask = data["nodes"]["selected"]
        edge_mask = data["edges"]["selected"]
    else: 
        node_mask = data["nodes"]["confidences"] >= node_thresh
        edge_mask = data["edges"]["confidences"] >= edge_thresh
    
    # Cosine Similarity
    sims = None
    sim_min, sim_max = 0.0, 1.0
    cmap = plt.get_cmap('coolwarm')
    
    if active_node_idx is not None:
        embeddings = data["nodes"]["embeddings"]
        target_embedding = embeddings[active_node_idx].unsqueeze(0)
        
        sims = torch.nn.functional.cosine_similarity(target_embedding, embeddings, dim=1)
        sim_min = sims.min().item()
        sim_max = sims.max().item()
    
    # Process Nodes
    for idx, is_valid in enumerate(node_mask):
        if is_valid.item():  
            conf_val = data["nodes"]["confidences"][idx].item()
            
            # Dynamic Color Logic
            if active_node_idx is not None and sims is not None:
                score = sims[idx].item()
                
                if sim_norm_mode == "absolute":
                    # Absolute Mapping from [-1, 1] to [0, 1]
                    norm_score = (score + 1.0) / 2.0
                else:
                    # Relative Min-Max normalize for contrast stretching
                    if sim_max - sim_min > 1e-5:
                        norm_score = (score - sim_min) / (sim_max - sim_min)
                    else:
                        norm_score = 1.0
                
                color = mcolors.to_hex(cmap(norm_score))
                title_text = f"Conf: {conf_val:.2f} | Cos Sim: {score:.2f}"
            else:
                color = "#4CAF50" # Default green
                title_text = f"Conf: {conf_val:.2f}"
                
            ui_nodes.append(
                Node(
                    id=str(idx), 
                    label=node_names[idx], 
                    title=title_text,
                    size=25, 
                    shape="dot", 
                    color=color
                )
            )
            
    # Process Edges
    edge_indices = data["edges"]["indices"]
    for idx, is_valid in enumerate(edge_mask):
        if is_valid.item():
            src = int(edge_indices[idx][0].item())
            dst = int(edge_indices[idx][1].item())
            
            if node_mask[src] and node_mask[dst]:
                conf_val = data["edges"]["confidences"][idx].item()
                ui_edges.append(
                    Edge(
                        source=str(src), 
                        target=str(dst), 
                        id=str(idx), 
                        label=edge_names[idx], 
                        title=f"Conf: {conf_val:.2f}", 
                        color="#888888"
                    )
                )
                
    return ui_nodes, ui_edges