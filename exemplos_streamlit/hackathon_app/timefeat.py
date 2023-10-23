import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from clusterfeat import load_data, logscore_accuracy


def get_time_bucket(ts, first_ts, duration, n_buckets):
    delta_h = duration / n_buckets
    res = 0
    for i in range(n_buckets + 1):
        if ts < first_ts + timedelta(hours=delta_h*i):
            return res
        res += 1
    return res


def _time_bucket(s):
    return get_time_bucket(s.t, s.publish_ts, s.duration, 20)


# deviation to MP
def convert_mid_points(lst):
    return [(lst[i] + lst[i+1]) / 2 for i in range(len(lst)-1)]


def dev_to_mp(s):
    return abs((np.array(convert_mid_points(s.cdf)) - np.array(s.metaculus_cdf)).mean())


def dev_to_cp(s):
    return abs((np.array(convert_mid_points(s.cdf)) - np.array(s.community_cdf)).mean())


def time_process(bq, bp, cq, cp):

    bp["binary"] = 1
    cp["binary"] = 0

    # deviation from user prediction to community/metaculus aggregated prediction
    bp["mp_dev"] = np.abs(bp.prediction - bp.mp)
    cp["mp_dev"] = cp.apply(dev_to_mp, axis=1)
    bp["cp_dev"] = np.abs(bp.prediction - bp.cp)
    cp["cp_dev"] = cp.apply(dev_to_cp, axis=1)

    common_cols = list(set(bq.columns) & set(cq.columns))
    aq = pd.concat([bq[common_cols], cq[common_cols]])
    common_cols = list(set(bp.columns) & set(cp.columns))
    ap = pd.concat([bp[common_cols], cp[common_cols]])
    last_forecast = ap.t.max()

    apfirst = ap.sort_values(["question_id", "t"]).groupby("question_id").head(1)[["question_id", "t"]].rename(
                    columns={"t": "first_t"})
    aplast = ap.sort_values(["question_id", "t"]).groupby("question_id").tail(1)[["question_id", "t"]].rename(
            columns={"t": "last_t"})
    aq = aq.merge(apfirst, on="question_id", how="left")
    aq = aq.merge(aplast, on="question_id", how="left")
    aq["d_first_publish"] = (aq.first_t - aq.publish_ts).dt.seconds / (60*60)
    aq["end_ts"] = np.where(aq.resolve_ts < last_forecast, aq.resolve_ts, last_forecast)
    aq["duration"] = (aq.end_ts - aq.publish_ts) / np.timedelta64(1, 'h') #in hours
    ap = ap.merge(aq[["question_id", "publish_ts", "duration", "cat1", "cat2"]], on="question_id", how="left")

    # FIXME
    logging.warning("FIXME check the reason for nulls")
    ap = ap[ap.duration.notnull()]

    ap["time_bucket"] = ap.apply(_time_bucket, axis=1)

    ap["rep_level"] = 0
    ap.loc[ap.reputation_at_t.between(-0.49, -0.3), "rep_level"] = 1
    ap.loc[ap.reputation_at_t.between(-0.3, -0.05), "rep_level"] = 2
    ap.loc[ap.reputation_at_t.between(-0.05, 0.1), "rep_level"] = 3
    ap.loc[ap.reputation_at_t.between(0.1, 0.2), "rep_level"] = 4
    ap.loc[ap.reputation_at_t.between(0.2, 10), "rep_level"] = 5

    return ap


def load_time_data(datapath, do_preprocess=True, filter_user_days=10):
    bq, bp, cq, cp = load_data(datapath, do_preprocess, filter_user_days)
    bp, cp = logscore_accuracy(bq, bp, cq, cp)
    ap = time_process(bq, bp, cq, cp)

    # count the total forecasts for the user
    uc = ap.groupby("user_id").question_id.count().to_frame("usr_fcst_cnt").reset_index()
    ap = ap.merge(uc, on="user_id", how="left")

    # number of months active
    ma = ap[[["user_id", "year", "month"]]].drop_duplicates().groupby(["user_id"]).month.count().to_frame("months_active").reset_index()
    ap = ap.merge(ma, on="user_id", how="left")

    # first forecast date
    first = ap.sort_values("t").groupby("user_id").head(1)[["user_id","t"]]
    first["first_date"] = first.t.dt.date
    ap = ap.merge(first[["user_id", "first_date"]], on="user_id", how="left")

    return ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data for gapminder plot")
    parser.add_argument("--filter-user-days", type=int, default=0, help="filter out users with number of days lower or equal to this")
    parser.add_argument(
        "datapath", type=str, help="path to the data files folder"
    )
    parser.add_argument("output", type=str, help="path to the output CSV file")
    args = parser.parse_args()
    df = load_time_data(args.datapath, filter_user_days=args.filter_user_days)
    df.to_csv(args.output, index=False)
