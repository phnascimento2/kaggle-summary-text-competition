"""
Simple heatmap with streamlit

Data preprocessing in notebook heatmap_experiments.ipynb
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 780


features = ["bigram_overlap_ratio", "trigram_overlap_ratio", "word_overlap_ratio", 
            "bleurt_fs", 'cs_text', 'cs_question']

targets = ["content", "wording"]
target_range = {
    "content": {"min": -1.6, "max": 3.3}, 
    "wording": {"min": -2, "max": 4.4},
}

@st.cache_data
def load_data():
    basepath = Path(__file__).parent
    data = pd.read_parquet(basepath / "data.parquet")
    return data


def expand_cols(df, cols, fillvalue=None):
    """ expands the columns to have all columns in the given list and 
    with the list sorting
    """
    for c in cols:
        if c not in df.columns:
            df[c] = fillvalue
    df = df[cols]
    return df


#@st.cache_data
def get_pivot_table(length_ratio, xcol, ycol, target):
    dx = load_data()
    index_values = sorted(dx[ycol].unique())
    col_values = sorted(dx[xcol].unique())
    pt = pd.pivot_table(dx[dx.length_ratio.eq(length_ratio)], index=ycol, columns=xcol, 
                        values=target, aggfunc="mean")
    pt = pt.reindex(index_values)
    pt = expand_cols(pt, col_values)
    #st.write(pt)
    return pt


st.set_page_config(layout="wide")

st.sidebar.header("Evaluate Student Summaries")

y_col = st.sidebar.selectbox(
    label="Select Y variable",
    index=3,
    options=features,
)

x_col = st.sidebar.selectbox(
    label="Select X variable",
    index=5,
    options=features,
)

target_col = st.sidebar.selectbox(
    label="Select target",
    index=0,
    options=targets,
)

size_slider = st.sidebar.slider(
    label="Ratio of summary size", min_value=0.05, max_value=0.6, value=0.05, step=0.05
)

pt = get_pivot_table(size_slider, x_col, y_col, target_col)

fig = go.Figure(
    data=go.Heatmap(
        y=pt.index,
        x=pt.columns,
        z=pt.values,
        zmin=target_range[target_col]["min"], 
        zmax=target_range[target_col]["max"]
    )
)

fig.update_layout(
    #xaxis=dict(tick0=1, dtick=1),
    margin={"t": 0},
    width=HEATMAP_WIDTH,
    height=HEATMAP_HEIGHT,
)

st.plotly_chart(fig, use_container_width=True)
