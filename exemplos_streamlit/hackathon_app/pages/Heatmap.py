from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 780


scoring_options = {
    "Activity + Reputation": "score",
    "Reputation": "reputation",
    "Activity": "act",
    "Questions": "n_questions",
}

fill_value = {"score": -0.5, "reputation": -0.5, "act": 0, "n_questions": 0}


hovertemplate = """<b>user:%{y}</b>
<br>reputation: %{customdata[0]}
<br>join date: %{customdata[1]}
<br>metric score: %{z}
"""


def get_custom_data(pt, n_months, rep):
    """builds the data that will show in the hover pop-up"""
    usr = user_profile()
    df = pd.DataFrame(pt.index, columns=["user_id"])
    df = df.merge(usr, on="user_id", how="left")

    # FIXME - hack for the case where the dimension do not match
    try:
        res = np.dstack(
            (
                rep.values,
                np.repeat(df.first_date.values.reshape(-1, 1), n_months, axis=1),
            )
        )
    except:
        res = np.dstack(
            (
                rep.values,
                np.repeat(df.first_date.values.reshape(-1, 1), rep.shape[1], axis=1),
            )
        )
    return res


@st.cache
def load_data():
    basepath = Path(__file__).parent.parent.parent / "output"
    data = pd.read_csv(basepath / "forecasters_time.csv")
    data.user_id = data.user_id.apply(lambda s: str(s) + "-")
    return data


@st.cache
def user_profile():
    data = load_data()
    return data[["user_id", "first_date"]].groupby("user_id").head(1)


@st.cache
def get_pivot_table(n_months, score_col):
    data = load_data()
    sel = data[data.n_months == n_months].sort_values(
        ["n_active_months", "user_id", "year", "month"],
        ascending=[False, True, True, True],
    )
    pt = pd.pivot_table(sel, index="user_id", columns="mpos", values=score_col).fillna(
        fill_value[score_col]
    )
    rep = pd.pivot_table(
        sel, index="user_id", columns="mpos", values="rep_range", aggfunc="first"
    ).fillna("")
    return pt, rep


st.set_page_config(layout="wide")

st.sidebar.header("Forecasters Time Analysis 🧙‍♂️")

months_slider = st.sidebar.slider(
    label="Number of months in the platform", min_value=1, max_value=80, value=6
)

scoring_selector = st.sidebar.selectbox(
    label="Select scoring method",
    options=scoring_options.keys(),
)
score_col = scoring_options[scoring_selector]

pt, rep = get_pivot_table(months_slider, score_col)

fig = go.Figure(
    data=go.Heatmap(
        y=pt.index,
        x=pt.columns,
        z=pt.values,
        customdata=get_custom_data(pt, months_slider, rep),
        hovertemplate=hovertemplate,
    )
)

fig.update_layout(
    xaxis=dict(tick0=1, dtick=1),
    margin={"t": 0},
    width=HEATMAP_WIDTH,
    height=HEATMAP_HEIGHT,
)

st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show activity distribution for users"):
    hist_fig = px.histogram(
        load_data()[["user_id", "n_months"]].drop_duplicates(),
        x="n_months",
        title="Activity distribution for users",
        labels=dict(x="N months of data", y="Number of users"),
    )

    st.plotly_chart(hist_fig, use_container_width=True)
