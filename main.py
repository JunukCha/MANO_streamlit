import streamlit as st
import plotly.graph_objects as go
import torch
import smplx

from constants import hand_pose_names

# Sidebar for hand selection
hand_type = st.sidebar.selectbox('Select hand', ['Right hand', 'Left hand'])

# Set session state for hand type
if 'hand_type' not in st.session_state:
    st.session_state.hand_type = hand_type

if "left_hand_model" not in st.session_state:
    model_path = 'mano_v1_2/models/MANO_LEFT.pkl'  # Path to the MANO model files
    left_hand_model = smplx.create(model_path, use_pca=False, is_rhand=False)
    st.session_state.left_hand_model = left_hand_model
if "right_hand_model" not in st.session_state:
    model_path = 'mano_v1_2/models/MANO_RIGHT.pkl'  # Path to the MANO model files
    right_hand_model = smplx.create(model_path, use_pca=False, is_rhand=True)
    st.session_state.right_hand_model = right_hand_model

if hand_type == "Right hand":
    mano_model = st.session_state.right_hand_model
else:
    mano_model = st.session_state.left_hand_model

# Set arbitrary parameters
num_betas = 10
num_pca_comps = 45
betas = []
hand_pose = []

for i in range(num_betas):
    betas.append(st.sidebar.slider(f'Beta {i}', -2.0, 2.0, 0.0))

for i, name in enumerate(hand_pose_names):
    hand_pose.append(st.sidebar.slider(f'Pose PCA {i} | {name}', -2.0, 2.0, 0.0))

betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
hand_pose = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)

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
