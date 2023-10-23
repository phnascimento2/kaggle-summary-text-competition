from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import logging


PLOT_HEIGHT = 750
FCST_CNT_CAP = 3000
N_QUESTIONS_CAP = 1500
N_CATEGORIES_CAP = 50

st.set_page_config(layout="wide")

featmap = {
    "Deviation to MP": "dev_mp",
    "Deviation to CP": "dev_cp",
    "Repeated forecasts": "refcst",
    "Accuracy score": "norm_logscore",
    "Number of questions": "n_questions",
    "Number of forecasts": "fcst_cnt",
    "Number of categories": "n_categories",
    "Seniority": "age_level",
    "Reputation": "rep_level",
}

rep_intervals = {
    0: (-10, -0.49),
    1: (-0.49, -0.3),
    2: (-0.3, -0.05),
    3: (-0.05, 0.1),
    4: (0.1, 0.2),
    5: (0.2, 10),
}

age_intervals = {
    0: (0, 4),
    1: (4, 7),
    2: (7, 11),
    3: (11, 18),
    4: (18, 28),
    5: (28, 100),
}


@st.cache
def get_data():
    basepath = Path(__file__).parent.parent.parent / "output"
    logging.info("Loading and preprocessing data files")
    data = pd.read_csv(basepath / "question_time.csv")
    data.cat1 = data.cat1.fillna("")
    data.cat2 = data.cat2.fillna("")
    # convert the 20 buckets to percentage values
    data.time_bucket = data.time_bucket * 5
    return data


@st.cache
def all_categories():
    df = get_data()
    return list(sorted(list(set(df.cat1.values) | set(df.cat2.values))))


st.sidebar.header("Question Time Analysis 🔮")

categories = st.sidebar.multiselect("Categories", all_categories(), [])

question_type = st.sidebar.selectbox(
    label="Question type",
    options=["All", "Binary", "Continuous"],
)

resolution = st.sidebar.selectbox(
    label="Resolution",
    options=["All", "Resolved", "Not resolved"],
)

y_axis = st.sidebar.selectbox(
    label="Y axis",
    options=[
        "Repeated forecasts",
        "Accuracy score",
        "Deviation to MP",
        "Deviation to CP",
        "Number of categories",
        "Number of questions",
        "Number of forecasts",
    ],
)

x_axis = st.sidebar.selectbox(
    label="X axis",
    options=[
        "Deviation to MP",
        "Deviation to CP",
        "Repeated forecasts",
        "Accuracy score",
        "Number of categories",
        "Number of questions",
        "Number of forecasts",
    ],
)

bsize = st.sidebar.selectbox(
    label="Bubble size",
    options=["Number of forecasts", "Number of questions", "Number of categories"],
)

colorscale = st.sidebar.selectbox(
    label="Color levels",
    options=["Reputation", "Seniority"],
)

n_user_forecasts = st.sidebar.slider(
    label="Min number of user forecasts", min_value=1, max_value=200, value=50
)

min_reputation = st.sidebar.slider(
    label="Min reputation", min_value=-0.5, max_value=0.5, value=-0.5
)


def update_fig_layout(fig):
    duration_transition = 450
    current_layout = fig.layout
    for step in current_layout["sliders"][0]["steps"]:
        step["args"][1]["transition"]["duration"] = duration_transition
        step["args"][1]["frame"]["duration"] = duration_transition

    return current_layout


def selection_filter(ap):
    if categories:
        ap = ap[ap.cat1.isin(categories) | ap.cat2.isin(categories)]

    if min_reputation > -0.5:
        sel_users = ap[ap.reputation_at_t.ge(min_reputation)].user_id.unique()
        ap = ap[ap.user_id.isin(sel_users)].copy()

    if n_user_forecasts > 1:
        ap = ap[ap.usr_fcst_cnt >= n_user_forecasts].copy()

    if question_type != "All":
        ck = 1 if question_type == "Binary" else 0
        ap = ap[ap.binary == ck].copy()

    if x_axis == "Accuracy score" or y_axis == "Accuracy score":
        # only resolved question are relevant
        ap = ap[ap.resolution.notnull()].copy()

    if resolution == "Resolved":
        ap = ap[ap.resolution.notnull()].copy()

    elif resolution == "Not resolved":
        ap = ap[ap.resolution.isnull()].copy()
    return ap


def aggregate():
    ap = selection_filter(get_data())

    gbpx = ap.groupby(["user_id", "time_bucket"]).head(1)[
        ["user_id", "time_bucket", "months_active", "first_date"]
    ]

    # count forecasts
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .question_id.count()
        .to_frame("fcst_cnt")
        .reset_index()
    )
    df.loc[df.fcst_cnt > FCST_CNT_CAP, "fcst_cnt"] = FCST_CNT_CAP
    df.fcst_cnt = np.log(df.fcst_cnt)
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # count questions
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .question_id.nunique()
        .to_frame("n_questions")
        .reset_index()
    )
    df.loc[df.n_questions > N_QUESTIONS_CAP, "n_questions"] = N_QUESTIONS_CAP
    df.n_questions = np.log(df.n_questions)
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # count main categories
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .cat1.nunique()
        .to_frame("n_categories")
        .reset_index()
    )
    df.loc[df.n_categories > N_CATEGORIES_CAP, "n_categories"] = N_CATEGORIES_CAP
    df.n_categories = np.log(df.n_categories)
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # reforecast
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .question_seq.mean()
        .to_frame("refcst")
        .reset_index()
    )
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # deviation
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .mp_dev.mean()
        .to_frame("dev_mp")
        .reset_index()
    )
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .cp_dev.mean()
        .to_frame("dev_cp")
        .reset_index()
    )
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # reputation
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .reputation_at_t.quantile(0.9)
        .to_frame("reputation_at_t")
        .reset_index()
    )
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    # accuracy
    df = (
        ap.groupby(["user_id", "time_bucket"])
        .norm_logscore.mean()
        .to_frame("norm_logscore")
        .reset_index()
    )
    gbpx = gbpx.merge(df, on=["user_id", "time_bucket"], how="left")

    gbpx["rep_level"] = 0
    for lvl, (lb, ub) in rep_intervals.items():
        gbpx.loc[gbpx.reputation_at_t.between(lb, ub), "rep_level"] = lvl

    gbpx["age_level"] = 0
    for lvl, (lb, ub) in age_intervals.items():
        gbpx.loc[gbpx.months_active.between(lb, ub), "age_level"] = lvl

    # check for out of bounds
    oob = gbpx[gbpx.time_bucket.lt(1) | gbpx.time_bucket.gt(100)]
    if len(oob) > 0:
        logging.warning(f"There are {len(oob)} entries outside the 1-100 time bounds")
    gbpx = gbpx[gbpx.time_bucket.between(1, 100)].copy()
    gbpx = gbpx.sort_values(["time_bucket"])
    return gbpx


gbpx = aggregate()

st.sidebar.write(f"Users selected: {gbpx.user_id.nunique()}")


def axis_range(gdf, col):
    return [int(gdf[col].min()), np.ceil(gdf[col].max())]


def get_labels():
    mapper = {v: k for k, v in featmap.items()}
    mapper.update(
        {
            "rep_level": "Reputation",
            "time_bucket": "Time (%)",
            "fcst_cnt": "Forecast Count",
        }
    )
    return mapper


fig = px.scatter(
    gbpx,
    x=featmap.get(x_axis),
    y=featmap.get(y_axis),
    color=featmap.get(colorscale),
    size=featmap[bsize],
    size_max=12,
    hover_name="user_id",
    animation_frame="time_bucket",
    animation_group="user_id",
    range_x=axis_range(gbpx, featmap.get(x_axis)),
    range_y=axis_range(gbpx, featmap.get(y_axis)),
    labels=get_labels(),
    height=PLOT_HEIGHT,
)

new_layout = update_fig_layout(fig)
fig.update_layout(new_layout)

st.plotly_chart(fig, use_container_width=True)
