import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Metrology Surface Visualizer", layout="wide", page_icon="🌪️")

st.title("🌪️ Wind Velocity Surface Visualizer")
st.markdown("Interactive 3D interpolation of **Velocity** and **Data Rate**.")

# --- 2. Data Loading & Cleaning ---
@st.cache_data
def load_data(uploaded_file=None, local_path='Export_CalWheel0.csv'):
    """
    Loads data from an uploaded file or a local default file.
    """
    df = None
    
    # Prioritize uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
    # Fallback to local path if it exists
    elif os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
        except Exception as e:
            st.error(f"Error reading local file: {e}")
            return None
    else:
        st.warning("Please upload a CSV file to begin.")
        return None

    if df is not None:
        # CLEANING: Strip whitespace and remove quote characters from column names
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        # Verify columns exist
        required_columns = {
            'x': 'X Axis Position',
            'y': 'Y Axis Position',
            'z': 'Velocity Mean Ch. 1 (m/sec)',
            'c': 'Vel.Data Rate Ch. 1 (Hz)'
        }
        
        # Simple check to ensure columns exist
        missing = [col for key, col in required_columns.items() if col not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
            return None

        # Rename for easier handling internally
        df_clean = df.rename(columns={
            required_columns['x']: 'x',
            required_columns['y']: 'y',
            required_columns['z']: 'z',
            required_columns['c']: 'c'
        })
        
        # Return only clean numbers
        return df_clean[['x', 'y', 'z', 'c']].dropna()

# --- 3. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    # Load Data
    df = load_data(uploaded_file)

if df is not None:
    # --- Interactive Range Sliders ---
    with st.sidebar:
        st.subheader("📏 Set Data Ranges")
        
        # Helper to get min/max
        def get_range(col): return float(df[col].min()), float(df[col].max())
        
        x_min, x_max = get_range('x')
        y_min, y_max = get_range('y')
        z_min, z_max = get_range('z')

        # Sliders
        x_range = st.slider("X Axis Range", x_min, x_max, (x_min, x_max))
        y_range = st.slider("Y Axis Range", y_min, y_max, (y_min, y_max))
        z_range = st.slider("Velocity Filter (Z)", z_min, z_max, (z_min, z_max))

        st.subheader("🎨 Visuals")
        colorscale = st.selectbox("Color Theme", ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Turbo'], index=0)
        grid_density = st.slider("Grid Resolution", 50, 300, 100, help="Higher = Smoother but slower")

    # --- 4. Filtering Data ---
    mask = (
        (df['x'] >= x_range[0]) & (df['x'] <= x_range[1]) &
        (df['y'] >= y_range[0]) & (df['y'] <= y_range[1]) &
        (df['z'] >= z_range[0]) & (df['z'] <= z_range[1])
    )
    df_filtered = df.loc[mask]

    # Check if we filtered out all data
    if len(df_filtered) < 4:
        st.error("📉 Not enough data points in the selected range to generate a surface. Please widen your ranges.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Points Loaded", f"{len(df_filtered)}")
        col2.metric("Max Velocity", f"{df_filtered['z'].max():.2f} m/s")
        col3.metric("Avg Data Rate", f"{df_filtered['c'].mean():.2f} Hz")

        # --- 5. Interpolation ---
        # Create grid based on filtered limits
        xi = np.linspace(df_filtered['x'].min(), df_filtered['x'].max(), grid_density)
        yi = np.linspace(df_filtered['y'].min(), df_filtered['y'].max(), grid_density)
        X, Y = np.meshgrid(xi, yi)

        with st.spinner("Interpolating surface..."):
            try:
                # Interpolate Height (Velocity)
                Z_grid = griddata(
                    (df_filtered['x'], df_filtered['y']), 
                    df_filtered['z'], 
                    (X, Y), 
                    method='cubic'
                )

                # Interpolate Color (Data Rate)
                C_grid = griddata(
                    (df_filtered['x'], df_filtered['y']), 
                    df_filtered['c'], 
                    (X, Y), 
                    method='cubic'
                )
            except Exception as e:
                st.error(f"Interpolation error: {e}. Try reducing the grid resolution or widening ranges.")
                Z_grid = None

        # --- 6. Plotting ---
        if Z_grid is not None:
            fig = go.Figure(data=[go.Surface(
                x=X,
                y=Y,
                z=Z_grid,
                surfacecolor=C_grid,
                colorscale=colorscale,
                colorbar=dict(title='Data Rate (Hz)'),
                contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )])

            fig.update_layout(
                title='Wind Velocity Surface',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Velocity (m/s)',
                    aspectmode='cube' # Keeps axes proportional
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting data...")