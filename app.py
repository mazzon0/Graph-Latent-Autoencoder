import sys
import torch
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Config
from utils.graph_data import load_pytorch_data, create_ui_elements

st.set_page_config(layout="wide", page_title="Scene Graph Inspector")

if len(sys.argv) < 2:
    st.error("Missing file path! Run using: streamlit run app.py -- path/to/file.pt")
    st.stop()

file_path = sys.argv[1]

# Init
if "graph_data" not in st.session_state:
    st.session_state.graph_data = load_pytorch_data(file_path)

# Sidebar and Save
st.sidebar.title("Controls")
if st.sidebar.button("💾 Save Changes to File", type="primary"):
    torch.save(st.session_state.graph_data, file_path)
    st.sidebar.success(f"Overwrote {file_path} successfully!")

st.sidebar.divider()
view_mode = st.sidebar.radio("View Mode", ["Page 1 (Scene Graph)", "Page 2 (Attention Maps)", "Side-by-Side"])

def render_page_1():
    st.subheader("Scene Graph View")
    
    # Visualization Filtering Options
    st.markdown("#### Filter Options")
    filter_choice = st.radio(
        "Show nodes and edges based on:", 
        ["Selected Mask", "Confidence Threshold"],
        horizontal=True
    )
    
    node_thresh, edge_thresh = 0.0, 0.0
    if filter_choice == "Confidence Threshold":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            node_thresh = st.slider("Node Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        with col_t2:
            edge_thresh = st.slider("Edge Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            
    filter_mode_arg = "selected" if filter_choice == "Selected Mask" else "confidence"
    
    st.write("Click on a node to inspect and edit names.")
    
    nodes, edges = create_ui_elements(
        st.session_state.graph_data, 
        filter_mode=filter_mode_arg, 
        node_thresh=node_thresh, 
        edge_thresh=edge_thresh
    )
    
    graph_config = Config(width="100%", height=600, directed=True, physics=True, hierarchical=False)
    clicked_node_id = agraph(nodes=nodes, edges=edges, config=graph_config)
    
    if clicked_node_id is not None:
        node_idx = int(clicked_node_id)
        
        st.session_state.active_node_idx = node_idx
        
        col_node, col_edge = st.columns(2)
        
        # Edit Node
        with col_node:
            st.markdown("#### Edit Node")
            current_node_name = st.session_state.graph_data["nodes"]["names"][node_idx]
            new_node_name = st.text_input("Node Name:", value=current_node_name)
            
            if st.button("Update Node"):
                st.session_state.graph_data["nodes"]["names"][node_idx] = new_node_name
                st.rerun()
        
        # Edit Connected Edges
        with col_edge:
            st.markdown("#### Edit Connected Edges")
            edge_indices = st.session_state.graph_data["edges"]["indices"]
            edge_names = st.session_state.graph_data["edges"]["names"]
            
            connected_edges = [
                e_idx for e_idx, (src, dst) in enumerate(edge_indices) 
                if src.item() == node_idx or dst.item() == node_idx
            ]
            
            if connected_edges:
                edge_options = {f"{edge_names[e]} (Index {e})": e for e in connected_edges}
                selected_edge_label = st.selectbox("Select an edge:", list(edge_options.keys()))
                selected_edge_idx = edge_options[selected_edge_label]
                
                new_edge_name = st.text_input("Edge Name:", value=edge_names[selected_edge_idx])
                if st.button("Update Edge"):
                    st.session_state.graph_data["edges"]["names"][selected_edge_idx] = new_edge_name
                    st.rerun()
            else:
                st.info("No edges connected to this node.")

def render_page_2():
    st.subheader("Attention Map View")
    
    target_type = st.radio(
        "Select Attention Target:", 
        ["Selected Node", "Global Token", "Relation Token"], 
        horizontal=True
    )
    
    num_layers = st.session_state.graph_data["metadata"]["num_layers"]
    selected_layer = st.slider("Decoder Layer", min_value=1, max_value=num_layers, value=1)
    layer_idx = selected_layer - 1 
    
    attn_map_tensor = None
    
    if target_type == "Selected Node":
        if "active_node_idx" not in st.session_state:
            st.info("👈 Please click a node in the Scene Graph first.")
            return
            
        node_idx = st.session_state.active_node_idx
        node_name = st.session_state.graph_data["nodes"]["names"][node_idx]
        st.markdown(f"**Viewing Attention for:** `{node_name}`")
        attn_map_tensor = st.session_state.graph_data["nodes"]["attention_maps"][node_idx, layer_idx]
        
    elif target_type == "Global Token":
        st.markdown("**Viewing Attention for:** `Global Token`")
        attn_map_tensor = st.session_state.graph_data["global_tokens"]["global_attention"][layer_idx]
        
    elif target_type == "Relation Token":
        st.markdown("**Viewing Attention for:** `Relation Token`")
        attn_map_tensor = st.session_state.graph_data["global_tokens"]["relation_attention"][layer_idx]
    
    if attn_map_tensor is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(attn_map_tensor.numpy(), cmap='jet', interpolation='nearest')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)

if view_mode == "Page 1 (Scene Graph)":
    render_page_1()
elif view_mode == "Page 2 (Attention Maps)":
    render_page_2()
elif view_mode == "Side-by-Side":
    col1, col2 = st.columns(2)
    with col1:
        render_page_1()
    with col2:
        render_page_2()