import pandas as pd
import numpy as np
import streamlit as st

from datetime import  date


@st.cache
def create_data(data_path):

    df = pd.read_csv(data_path)

    first = df.sort_values(["user_id", "year", "month"]).groupby("user_id").head(1)
    last = df.sort_values(["user_id", "year", "month"]).groupby("user_id").tail(1)

    first["first_date"] = first.apply(lambda s: date(s.year, s.month, 1), axis=1)
    last["last_date"] = last.apply(lambda s: date(s.year, s.month, 1), axis=1)

    first = first.merge(last[["user_id", "last_date"]], on="user_id", how="left")

    first["n_days"] = first.apply(lambda s: s.last_date - s.first_date, axis=1) 

    first["n_months"] = np.round(first.n_days.dt.days / 30.5) + 1
    first["n_months"] = first["n_months"].astype(int)   
    df = df.merge(first[["user_id", "first_date", "n_months"]], on="user_id", how="left")

    df["curr_date"] = df.apply(lambda s: date(s.year, s.month, 1), axis=1)
    df["mpos"] = np.round((df.curr_date - df.first_date).dt.days / 30.5) + 1

    df["score"] = 2*df.reputation + df.act

    actmon = df.groupby("user_id").year.count().to_frame("n_active_months").reset_index()
    df = df.merge(actmon, on="user_id", how="left")

    return df