import sys
import torch
import streamlit as st
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
    st.write("Click on a node to inspect and edit names.")
    
    # Generate UI elements from current memory state
    nodes, edges = create_ui_elements(st.session_state.graph_data)
    graph_config = Config(width="100%", height=600, directed=True, physics=True, hierarchical=False)
    
    clicked_node_id = agraph(nodes=nodes, edges=edges, config=graph_config)
    
    if clicked_node_id is not None:
        node_idx = int(clicked_node_id)
        
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
            
            # Find all edges attached to the clicked node
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
    st.success("attention map...")


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