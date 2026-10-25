import numpy as np
import plotly.express as px
import pandas as pd

# 1. Load trajectory data
data = np.load("new_simulator/new_simulation_data/data_killing_0.npy")
N, T, _ = data.shape

# 2. Define human-readable label mapping
TYPE_MAPPING = {
    0.0: "Dead Cell",
    1.0: "Immune Scout",
    1.4: "Immune Messenger",
    1.8: "Immune Killer",
    3.0: "Normal (Sessile) Cancer",
    4.0: "Evasive Cancer"
}

# 3. Define custom color palette matching your presentation scheme
COLOR_DISCRETE_MAP = {
    "Dead Cell": "#4A4A4A",              # Dark Gray
    "Immune Scout": "#87CEFA",           # Light Blue
    "Immune Messenger": "#008080",       # Teal
    "Immune Killer": "#00008B",          # Dark Blue
    "Normal (Sessile) Cancer": "#FF8C00",# Orange
    "Evasive Cancer": "#FF0000"          # Red
}

def code_to_label(type_code):
    # Snap float encodings to the nearest mapping key
    key = min(TYPE_MAPPING.keys(), key=lambda k: abs(k - type_code))
    return TYPE_MAPPING[key]

# 4. Reshape into a DataFrame for Plotly
records = []
for t in range(T):
    for i in range(N):
        raw_type = data[i, t, 4]
        label = code_to_label(raw_type)
        records.append({
            'x': data[i, t, 0],
            'y': data[i, t, 1],
            'vx': data[i, t, 2],
            'vy': data[i, t, 3],
            'Cell Type': label,
            'frame': t,
            'cell_id': i
        })

df = pd.DataFrame(records)

# 5. Generate Plotly scatter animation with human-readable labels
fig = px.scatter(
    df, 
    x="x", 
    y="y", 
    animation_frame="frame", 
    color="Cell Type",
    color_discrete_map=COLOR_DISCRETE_MAP,
    hover_data={
        "vx": ":.2f", 
        "vy": ":.2f", 
        "cell_id": True,
        "x": ":.1f",
        "y": ":.1f"
    },
    range_x=[0, df['x'].max()], 
    range_y=[0, df['y'].max()],
    labels={
        "x": "Spatial Coordinate X (μm)",
        "y": "Spatial Coordinate Y (μm)",
        "Cell Type": "Phenotype Category"
    },
    title="Interactive Immune-Cancer Trajectory Browser"
)

# Visual Polish
fig.update_traces(marker=dict(size=7, opacity=0.85))
fig.update_layout(
    template="plotly_white",
    legend_title_text="Cell Subtype",
    font=dict(family="Arial", size=12)
)

fig.show()