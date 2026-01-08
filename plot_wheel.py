import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import os

# ==========================================
# 1: PAGE SETUP
# ==========================================
# This sets the browser tab title and layout. 
# "layout='wide'" uses the full width of the monitor, which is great for big 3D plots.
st.set_page_config(page_title="Metrology Surface Visualizer", layout="wide", page_icon="🌪️")

st.title("🌪️ Wind Velocity Surface Visualizer")
st.markdown("""
**Welcome!** This tool turns raw sensor data into a 3D surface.
Use the menu on the left to filter noise and adjust the view.
""")

# ==========================================
# 2: DATA LOADING FUNCTION
# ==========================================
# We use @st.cache_data to speed things up. It tells Streamlit:
# "If the file hasn't changed, don't waste time reloading it from scratch."
@st.cache_data
def load_data(uploaded_file=None, local_path='Export_CalWheel0.csv'):
    """
    Tries to load data from a user upload. If none exists, looks for a local default file.
    Returns: A clean DataFrame (spreadsheet) or None if something goes wrong.
    """
    df = None
    
    # CASE A: User uploaded a file in the sidebar
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
            return None

    # CASE B: No upload, check if the default file exists on the computer
    elif os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
        except Exception as e:
            st.error(f"❌ Error reading local file: {e}")
            return None
    
    # CASE C: No file found at all
    else:
        st.info("👋 Please upload a CSV file in the sidebar to begin.")
        return None

    if df is not None:
        # 1. Clean Column Names: Remove hidden spaces or weird quotes (e.g., " X Axis " -> "X Axis")
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        # 2. Rename Columns: We map the long CSV names to simple x, y, z, c (color) variables
        # This makes the rest of the code much easier to write and read.
        rename_map = {
            'X Axis Position': 'x',
            'Y Axis Position': 'y',
            'Velocity Mean Ch. 1 (m/sec)': 'z',  # This is our Height/Velocity
            'Vel.Data Rate Ch. 1 (Hz)': 'c'      # This is our Color/Heatmap
        }
        
        # Check if the CSV actually has these columns
        missing_cols = [key for key in rename_map.keys() if key not in df.columns]
        if missing_cols:
            st.error(f"⚠️ Your CSV is missing these columns: {missing_cols}")
            return None

        # Rename and drop any rows that have missing values (NaN)
        df_clean = df.rename(columns=rename_map)
        return df_clean[['x', 'y', 'z', 'c']].dropna()

# ==========================================
# 3: SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Step 1: File Uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    # Step 2: Actually load the data using our function from 2
    df = load_data(uploaded_file)

# Only run the rest of the app if data loaded successfully
if df is not None:
    with st.sidebar:
        # --- NOISE FILTER ---
        st.subheader("🧹 Noise Reduction")
        # Checkbox to remove 0.0 values (sensor dropouts)
        filter_zeros = st.checkbox("Remove Static Noise (Velocity ≈ 0)", value=True,
                                   help="Sensors sometimes read 0.0 when they miss a reading. Check this to ignore those points.")
        
        # Apply the noise filter immediately
        if filter_zeros:
            original_count = len(df)
            # We keep only rows where Z (Velocity) is greater than a tiny number (0.001)
            # abs() ensures we catch negative noise too, though unlikely in velocity.
            df = df[np.abs(df['z']) > 0.001]
            
            # Show the user how many bad points were removed
            removed = original_count - len(df)
            if removed > 0:
                st.caption(f"🚫 Ignored {removed} noise points")
        
        st.divider() # Adds a nice visual line

        # --- RANGE SLIDERS ---
        st.subheader("📏 Set Data Ranges")
        st.caption("Adjust sliders to crop the 3D model.")

        # Helper to safely get min/max even if data is empty
        def get_limits(col):
            if df.empty: return 0.0, 1.0
            return float(df[col].min()), float(df[col].max())

        x_min, x_max = get_limits('x')
        y_min, y_max = get_limits('y')
        z_min, z_max = get_limits('z')

        # The sliders return a "tuple" (min_value, max_value)
        x_range = st.slider("X Axis (Width)", x_min, x_max, (x_min, x_max))
        y_range = st.slider("Y Axis (Depth)", y_min, y_max, (y_min, y_max))
        
        # Only show Z slider if we actually have a range of data
        if z_min < z_max:
            z_range = st.slider("Z Axis (Velocity Filter)", z_min, z_max, (z_min, z_max))
        else:
            z_range = (z_min, z_max)

        st.divider()

        # --- VISUAL SETTINGS ---
        st.subheader("🎨 Visuals")
        colorscale = st.selectbox("Color Theme", ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Turbo'], index=4)
        
        # Grid density controls how "smooth" the surface looks. 
        # Higher number = smoother but might lag on slow computers.
        grid_density = st.slider("Smoothness (Resolution)", 50, 300, 100)

    # ==========================================
    # 4: DATA PROCESSING
    # ==========================================
    
    # 1. Apply the Slider Filters
    # We create a "mask" (a list of True/False) for rows that fit inside our sliders
    mask = (
        (df['x'] >= x_range[0]) & (df['x'] <= x_range[1]) &
        (df['y'] >= y_range[0]) & (df['y'] <= y_range[1]) &
        (df['z'] >= z_range[0]) & (df['z'] <= z_range[1])
    )
    # Apply the mask to create our final dataset
    df_filtered = df.loc[mask]

    # Safety Check: If user filtered everything out, stop here so we don't crash
    if len(df_filtered) < 4:
        st.warning("📉 Not enough data points selected. Please widen your sliders in the sidebar.")
    else:
        # 2. Show Quick Stats at the top of the page
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Data Points", f"{len(df_filtered):,}")
        col2.metric("Max Velocity", f"{df_filtered['z'].max():.2f} m/s")
        col3.metric("Avg Data Rate", f"{df_filtered['c'].mean():.2f} Hz")

        # 3. Interpolation (The Math Part)
        # We have scattered dots, but we need a smooth sheet. 
        # 'np.meshgrid' creates a perfect grid of squares.
        # 'griddata' guesses the height of the grid points based on the nearest real dots.
        
        xi = np.linspace(df_filtered['x'].min(), df_filtered['x'].max(), grid_density)
        yi = np.linspace(df_filtered['y'].min(), df_filtered['y'].max(), grid_density)
        X, Y = np.meshgrid(xi, yi)

        with st.spinner("Calculating 3D Surface..."):
            try:
                # Calculate Height (Z)
                Z_grid = griddata(
                    (df_filtered['x'], df_filtered['y']), 
                    df_filtered['z'], 
                    (X, Y), 
                    method='cubic' # 'cubic' makes it curvy/smooth
                )

                # Calculate Color (C)
                C_grid = griddata(
                    (df_filtered['x'], df_filtered['y']), 
                    df_filtered['c'], 
                    (X, Y), 
                    method='cubic'
                )
            except Exception as e:
                st.error(f"Could not calculate surface: {e}")
                Z_grid = None

        # ==========================================
        # 5: PLOTTING
        # ==========================================
        if Z_grid is not None:
            # Create the 3D Surface
            surface = go.Surface(
                x=X, y=Y, z=Z_grid,
                surfacecolor=C_grid,
                colorscale=colorscale,
                colorbar=dict(title='Data Rate (Hz)'),
                # Add contour lines (rings) to make height easier to see
                contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            )

            fig = go.Figure(data=[surface])

            # Make the chart look professional
            fig.update_layout(
                title='Wind Velocity 3D Model',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Velocity (m/s)',
                    aspectmode='cube' # Keeps the box square, so it doesn't look squashed
                ),
                margin=dict(l=0, r=0, b=0, t=30), # Tight margins
                height=700 # Pixel height of the plot
            )

            # Finally, draw the chart on the screen
            st.plotly_chart(fig, use_container_width=True)