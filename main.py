import streamlit as st
import plotly.graph_objects as go
import torch
import smplx

# Function to load the MANO model
@st.cache_resource
def load_mano_model(model_path, is_right_hand):
    return smplx.create(model_path, model_type='mano', use_pca=False, is_rhand=is_right_hand)

# Sidebar for hand selection
hand_type = st.sidebar.selectbox('Select hand', ['Right hand', 'Left hand'])

# Set session state for hand type
if 'hand_type' not in st.session_state:
    st.session_state.hand_type = hand_type

# Load the appropriate MANO model based on user selection
model_path = 'path/to/mano/models'  # Path to the MANO model files

if "left_hand_model" not in st.session_state:
    left_hand_model = smplx.create(model_path, model_type='mano', use_pca=False, is_rhand=False)
    st.session_state.left_hand_model = left_hand_model
if "right_hand_model" not in st.session_state:
    right_hand_model = smplx.create(model_path, model_type='mano', use_pca=False, is_rhand=True)
    st.session_state.right_hand_model = right_hand_model

if hand_type == "Right hand":
    mano_model = st.session_state.right_hand_model
else:
    mano_model = st.session_state.left_hand_model

# Set arbitrary parameters
num_betas = 10
num_pca_comps = 12
betas = torch.zeros([1, num_betas], dtype=torch.float32)
hand_pose = torch.zeros([1, num_pca_comps], dtype=torch.float32)

output = mano_model(betas=betas, hand_pose=hand_pose)
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = mano_model.faces

# Create 3D plot using Plotly
fig = go.Figure(data=[go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    opacity=0.5,
    color='blue'
)])

# Display 3D plot in Streamlit app
st.plotly_chart(fig)
