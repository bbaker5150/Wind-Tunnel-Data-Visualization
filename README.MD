# 🌪️ Wind Velocity Surface Visualizer

An interactive 3D analysis tool built with **Streamlit** and **Plotly**. This application takes raw coordinate and velocity data, interpolates it onto a uniform grid, and renders an interactive 3D surface plot.

It is designed to visualize **Wind Velocity** (Height) and **Data Rate** (Color) across a spatial XY grid.

## 🚀 Features

* **Interactive 3D Plotting:** Rotate, zoom, and pan the surface model.
* **Dynamic Filtering:** Sidebar sliders to filter X, Y, and Z (Velocity) ranges in real-time.
* **Data Interpolation:** Uses Cubic interpolation to create smooth surfaces from scattered data points.
* **Customization:** Change color maps (Jet, Viridis, Turbo) and grid resolution.
* **Drag & Drop:** Upload new CSV files directly through the browser interface.

## 📦 Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies** using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the application using the Streamlit command line interface:

```bash
streamlit run plot_wheel.py