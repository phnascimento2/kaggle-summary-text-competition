import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

logging.basicConfig(level="DEBUG")


class Config:
    act_cap = 100
    rep_min_cap = -0.5
    rep_max_cap = 0.25
    bin_logscore_min_cap = -3
    bin_logscore_max_cap = 1
    con_logscore_min_cap = -10
    con_logscore_max_cap = 0


def load_data(datapath, do_preprocess=True, filter_user_days=0):
    datapath = Path(datapath)
    bq = pd.read_json(
        datapath / "questions-binary-hackathon.json",
        orient="records",
        convert_dates=False,  # necessary, otherwise Pandas messes up date conversion.
    )
    cq = pd.read_json(
        datapath / "questions-continuous-hackathon.json",
        orient="records",
        convert_dates=False,  # necessary, otherwise Pandas messes up date conversion.
    )
    bp = pd.read_json(
        datapath / "predictions-binary-hackathon.json",
        orient="records",
    )
    cp = pd.read_parquet(
        datapath / "predictions-continuous-hackathon-v2.parquet",
    )
    # remove the forecasts for question ids missing in the question data
    missing = set(bp.question_id.unique()) - set(bq.question_id.unique())
    bp = bp[~bp.question_id.isin(missing)].copy()
    logging.info(
        f"Removed forecasts for {len(missing)} questions missing in binary question table"
    )
    missing = set(cp.question_id.unique()) - set(cq.question_id.unique())
    cp = cp[~cp.question_id.isin(missing)].copy()
    logging.info(
        f"Removed forecasts for {len(missing)} questions missing in continuous question table"
    )

    if do_preprocess:
        bq, bp, cq, cp = preprocess_base_files(bq, bp, cq, cp)
        bp, cp = add_days_months(bp, cp)
        if filter_user_days > 0:
            size = len(bp) + len(cp)
            bp = bp[bp.n_days > filter_user_days].copy()
            cp = cp[cp.n_days > filter_user_days].copy()
            logging.info(
                f"filter_user_days: removed {size - len(bp) - len(cp)} forecasts with less than {filter_user_days} days"
            )
    return bq, bp, cq, cp


def preprocess_base_files(bq, bp, cq, cp):
    bp["t"] = bp["t"].apply(datetime.fromtimestamp)
    cp["t"] = cp["t"].apply(datetime.fromtimestamp)

    bq["created_ts"] = bq.created_time.apply(datetime.fromtimestamp)
    bq["publish_ts"] = bq.publish_time.apply(datetime.fromtimestamp)
    bq["close_ts"] = bq.close_time.apply(datetime.fromtimestamp)
    bq["resolve_ts"] = bq.resolve_time.apply(datetime.fromtimestamp)
    cq["created_ts"] = cq.created_time.apply(datetime.fromtimestamp)
    cq["publish_ts"] = cq.publish_time.apply(datetime.fromtimestamp)
    cq["close_ts"] = cq.close_time.apply(datetime.fromtimestamp)
    cq["resolve_ts"] = cq.resolve_time.apply(datetime.fromtimestamp)

    bp["year"] = bp.t.dt.year
    bp["month"] = bp.t.dt.month
    cp["year"] = cp.t.dt.year
    cp["month"] = cp.t.dt.month

    # sequential number of user forecast for the question
    bp["seq"] = range(len(bp))
    bp["question_seq"] = bp.groupby(["user_id", "question_id"]).seq.rank()
    bp = bp.drop("seq", axis=1)
    cp["seq"] = range(len(cp))
    cp["question_seq"] = cp.groupby(["user_id", "question_id"]).seq.rank()
    cp = cp.drop("seq", axis=1)

    # remove predictions done after resolve time or before publish time
    bp = remove_after_resolve(bq, bp)
    cp = remove_after_resolve(cq, cp)
    bp = remove_before_publish(bq, bp)
    cp = remove_before_publish(cq, cp)

    # extract top categories
    bq["cat1"] = bq.categories.apply(lambda v: extract_top_cat(v, 0))
    bq["cat2"] = bq.categories.apply(lambda v: extract_top_cat(v, 1))
    cq["cat1"] = cq.categories.apply(lambda v: extract_top_cat(v, 0))
    cq["cat2"] = cq.categories.apply(lambda v: extract_top_cat(v, 1))

    return bq, bp, cq, cp


def extract_top_cat(lst, pos):
    """extract the top category from the list of categories hierarchy"""
    if lst is None:
        return ""
    if len(lst) <= pos:
        return ""
    for marker in ["—", "--", "–", "–", "–", "––"]:
        if marker in lst[pos]:
            return lst[pos].split(marker)[0].strip()
    return lst[pos].strip()


def remove_after_resolve(qx, px):
    """remove forecasts with timestamp after the resolve date"""
    qx = qx.copy()
    px = px.copy()
    initial_size = len(px)
    px = px.merge(
        qx[["question_id", "resolve_ts", "resolution", "resolution_comment"]],
        on="question_id",
        how="left",
    )
    px["after_resolve"] = False
    exclude = [
        "resolution ambiguous",
        "resolution > upper bound",
        "resolution < lower bound",
    ]
    px.loc[
        (px.resolution.notnull() | px.resolution_comment.isin(exclude))
        & px.resolve_ts.lt(px.t),
        "after_resolve",
    ] = True
    px = (
        px[~px.after_resolve]
        .copy()
        .drop(["after_resolve", "resolution_comment"], axis=1)
    )
    logging.info(f"Removed forecasts in after-resolve filter: {initial_size - len(px)}")
    return px


def remove_before_publish(qx, px):
    """remove forecasts with timestamp before publish date"""
    qx = qx.copy()
    px = px.copy()
    initial_size = len(px)
    px = px.merge(qx[["question_id", "publish_ts"]], on="question_id", how="left")
    px = px[px.publish_ts.le(px.t)].copy().drop("publish_ts", axis=1)
    logging.info(f"Removed entries in before-publish filter: {initial_size - len(px)}")
    return px


def add_days_months(bp, cp):
    """add number of days and number of months user was active in the platform"""
    cols = ["user_id", "question_id", "t", "year", "month"]
    ap = pd.concat([bp[cols], cp[cols]])
    ap["date"] = ap.t.dt.date

    uda = ap[["user_id", "date"]].drop_duplicates()
    uda = uda.groupby("user_id").date.count().to_frame("n_days").reset_index()

    umt = ap[["user_id", "year", "month"]].drop_duplicates()
    umt = umt.groupby("user_id").year.count().to_frame("n_months").reset_index()

    uda = uda.merge(umt, on="user_id")

    bp = bp.merge(uda, on="user_id", how="left")
    cp = cp.merge(uda, on="user_id", how="left")

    return bp, cp


def midpoints(lst):
    """transform a grid of N points and return the N-1 midpoints in the grid."""
    return [(lst[i] + lst[i + 1]) / 2 for i in range(len(lst) - 1)]


def get_proba(x_grid, cdf, res):
    if np.isnan(res):
        return np.nan
    x_grid = np.array(x_grid)
    cdf = np.array(cdf)
    pdf = np.diff(cdf)
    x_midgrid = midpoints(x_grid)
    pos = np.searchsorted(x_midgrid, [res])[0]
    if pos == 0:
        p = pdf[0]
    elif pos == len(pdf):
        p = pdf[-1]
    else:
        f = interp1d(x_midgrid[pos - 1 : pos + 1], pdf[pos - 1 : pos + 1])
        p = f(res).item()
    return p


def logscore_accuracy(bq, bp, cq, cp, cfg):
    """Add logscore accuracy columns values to the prediction datasets"""
    # binary
    xbp = bp.copy()
    if "resolution" not in bp.columns:
        xbp = xbp.merge(
            bq[["question_id", "resolution"]], on=["question_id"], how="left"
        )
    xbp.loc[xbp.resolution.notnull(), "logscore"] = (
        xbp.resolution * (np.log2(xbp.prediction))
        + (1 - xbp.resolution) * (np.log2(1 - xbp.prediction))
        + 1
    )

    # continuous
    xcp = cp.copy()
    cols_to_add = [c for c in ("resolution", "x_grid") if c not in xcp.columns]
    if cols_to_add:
        xcp = xcp.merge(
            cq[["question_id"] + cols_to_add], on=["question_id"], how="left"
        )

    xcp["proba"] = xcp.apply(lambda s: get_proba(s.x_grid, s.cdf, s.resolution), axis=1)

    # changing zero values to very low values for log to work
    xcp.loc[xcp.proba.eq(0), "proba"] = 1e-300
    xcp.loc[xcp.proba.notnull(), "logscore"] = np.log2(xcp.proba)

    # normalization of logscores to be possible to mix binary and continuous
    xcp["norm_logscore"] = xcp.logscore
    xcp.loc[
        xcp.logscore.lt(cfg.con_logscore_min_cap), "norm_logscore"
    ] = cfg.con_logscore_min_cap
    xcp.loc[
        xcp.logscore.gt(cfg.con_logscore_max_cap), "norm_logscore"
    ] = cfg.con_logscore_max_cap
    xcp.norm_logscore = minmax_scale(xcp, "norm_logscore")

    xbp["norm_logscore"] = xbp.logscore
    xbp.loc[
        xbp.logscore.lt(cfg.bin_logscore_min_cap), "norm_logscore"
    ] = cfg.bin_logscore_min_cap
    xbp.loc[
        xbp.logscore.gt(cfg.bin_logscore_max_cap), "norm_logscore"
    ] = cfg.bin_logscore_max_cap
    xbp.norm_logscore = minmax_scale(xbp, "norm_logscore")

    return xbp, xcp


def minmax_scale(df, col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())


# -----------------------------------------------------------------------------
#   Question time analysis
# -----------------------------------------------------------------------------


def get_time_bucket(ts, first_ts, duration, n_buckets):
    """identify the time bucket for the forecast time"""
    delta_h = duration / n_buckets
    res = 0
    for i in range(n_buckets + 1):
        if ts < first_ts + timedelta(hours=delta_h * i):
            return res
        res += 1
    return res


def _time_bucket(s):
    return get_time_bucket(s.t, s.publish_ts, s.duration, 20)


def dev_to_mp(s):
    """mean absolute error for user CDF and MP CDF"""
    return abs((np.array(midpoints(s.cdf)) - np.array(s.metaculus_cdf)).mean())


def dev_to_cp(s):
    """mean absolute error for user CDF and Community CDF"""
    return abs((np.array(midpoints(s.cdf)) - np.array(s.community_cdf)).mean())


def question_time_process(bq, bp, cq, cp):

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

    apfirst = (
        ap.sort_values(["question_id", "t"])
        .groupby("question_id")
        .head(1)[["question_id", "t"]]
        .rename(columns={"t": "first_t"})
    )
    aplast = (
        ap.sort_values(["question_id", "t"])
        .groupby("question_id")
        .tail(1)[["question_id", "t"]]
        .rename(columns={"t": "last_t"})
    )
    aq = aq.merge(apfirst, on="question_id", how="left")
    aq = aq.merge(aplast, on="question_id", how="left")
    aq["d_first_publish"] = (aq.first_t - aq.publish_ts).dt.seconds / (60 * 60)
    aq["end_ts"] = np.where(
        aq.resolve_ts < last_forecast,
        aq.resolve_ts,
        last_forecast + timedelta(seconds=1),
    )
    aq["duration"] = (aq.end_ts - aq.publish_ts) / np.timedelta64(1, "h")  # in hours
    ap = ap.merge(
        aq[["question_id", "publish_ts", "duration", "cat1", "cat2", "end_ts"]],
        on="question_id",
        how="left",
    )

    if ap.duration.isnull().sum() > 0:
        logging.error(f"Forecasts with NaN duration: {ap.duration.isnull().sum()}")

    ap["time_bucket"] = ap.apply(_time_bucket, axis=1)

    return ap


def load_time_data(bq, bp, cq, cp, cfg):

    bp, cp = logscore_accuracy(bq, bp, cq, cp, cfg)
    ap = question_time_process(bq, bp, cq, cp)

    # count the total forecasts for the user
    uc = (
        ap.groupby("user_id").question_id.count().to_frame("usr_fcst_cnt").reset_index()
    )
    ap = ap.merge(uc, on="user_id", how="left")

    # number of months active
    ma = (
        ap[["user_id", "year", "month"]]
        .drop_duplicates()
        .groupby("user_id")
        .month.count()
        .to_frame("months_active")
        .reset_index()
    )
    ap = ap.merge(ma, on="user_id", how="left")

    # first forecast date
    first = ap.sort_values("t").groupby("user_id").head(1)[["user_id", "t"]]
    first["first_date"] = first.t.dt.date
    ap = ap.merge(first[["user_id", "first_date"]], on="user_id", how="left")

    return ap


# -----------------------------------------------------------------------------
#   Forecasters time analysis
# -----------------------------------------------------------------------------


def user_month_frequency(ap, cfg):
    """returns activity statistics for user-month"""
    mp = (
        ap.groupby(["user_id", "year", "month"])
        .question_id.count()
        .to_frame("pcnt")
        .reset_index()
    )
    # make a normalized activity feature with a cap at act_cap monthly predictions
    mp["act"] = np.log(mp.pcnt)
    mp.loc[mp.act > np.log(cfg.act_cap), "act"] = np.log(cfg.act_cap)
    mp.act = mp.act / np.log(cfg.act_cap)
    return mp


def user_month_reputation(ap, cfg):
    """returns reputation statistics for user-month"""
    rep = (
        ap.groupby(["user_id", "year", "month"])
        .reputation_at_t.max()
        .to_frame("repmax")
        .reset_index()
    )
    rep["reputation"] = rep.repmax
    rep.loc[rep.reputation.lt(cfg.rep_min_cap), "reputation"] = cfg.rep_min_cap
    rep.loc[rep.reputation.gt(cfg.rep_max_cap), "reputation"] = cfg.rep_max_cap
    rep.reputation += -1 * cfg.rep_min_cap
    rep.reputation = np.log1p(rep.reputation)
    rep.reputation = rep.reputation / rep.reputation.max()

    repmin = (
        ap.groupby(["user_id", "year", "month"])
        .reputation_at_t.min()
        .to_frame("repmin")
        .reset_index()
    )
    rep = rep.merge(repmin, on=["user_id", "year", "month"])
    # make a string with range of reputation in period
    rep["rep_range"] = (
        rep.repmin.apply(lambda x: f"{x:3.2f}")
        + " to "
        + rep.repmax.apply(lambda x: f"{x:3.2f}")
    )
    return rep


def load_forecasters_time_data(bq, bp, cq, cp, cfg):
    bq["binary"] = bp["binary"] = 1
    cq["binary"] = cp["binary"] = 0
    common_cols = list(set(bp.columns) & set(cp.columns))
    ap = pd.concat([bp[common_cols], cp[common_cols]])

    df = user_month_frequency(ap, cfg)
    rep = user_month_reputation(ap, cfg)
    df = df.merge(rep, on=["user_id", "year", "month"], how="left")

    # number of questions
    mp = (
        ap.groupby(["user_id", "year", "month"])
        .question_id.nunique()
        .to_frame("n_questions")
        .reset_index()
    )
    df = df.merge(mp, on=["user_id", "year", "month"], how="left")

    # number of months elapsed
    first = df.sort_values(["user_id", "year", "month"]).groupby("user_id").head(1)
    last = df.sort_values(["user_id", "year", "month"]).groupby("user_id").tail(1)

    first["first_date"] = first[["year", "month"]].apply(
        lambda s: date(s.year, s.month, 1), axis=1
    )
    last["last_date"] = last[["year", "month"]].apply(
        lambda s: date(s.year, s.month, 1), axis=1
    )

    first = first.merge(last[["user_id", "last_date"]], on="user_id", how="left")

    first["n_days"] = first.apply(lambda s: s.last_date - s.first_date, axis=1)

    first["n_months"] = np.round(first.n_days.dt.days / 30.5) + 1
    first["n_months"] = first["n_months"].astype(int)
    df = df.merge(
        first[["user_id", "first_date", "n_months"]], on="user_id", how="left"
    )

    df["curr_date"] = df.apply(lambda s: date(s.year, s.month, 1), axis=1)
    df["mpos"] = np.round((df.curr_date - df.first_date).dt.days / 30.5) + 1

    # simple combination of reputation and activity
    df["score"] = 1.5 * df.reputation + df.act

    actmon = (
        df.groupby("user_id").year.count().to_frame("n_active_months").reset_index()
    )
    df = df.merge(actmon, on="user_id", how="left")
    return df


def tests():
    assert list(midpoints([1, 3, 5])) == [2, 4]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess data for visualizations")
    parser.add_argument("datapath", type=str, help="path to the data files folder")
    parser.add_argument(
        "--filter-user-days",
        type=int,
        default=0,
        help="filter out users with number of days lower or equal to this",
    )
    args = parser.parse_args()
    cfg = Config()

    bq, bp, cq, cp = load_data(
        args.datapath, do_preprocess=True, filter_user_days=args.filter_user_days
    )
    # folder to store the preprocessed files
    outpath = Path(__file__).parent.parent / "output"
    os.makedirs(outpath, exist_ok=True)
    df = load_time_data(bq, bp, cq, cp, cfg)
    df.to_csv(outpath / "question_time.csv", index=False)
    df = load_forecasters_time_data(bq, bp, cq, cp, cfg)
    df.to_csv(outpath / "forecasters_time.csv", index=False)
