import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd


def load_data(datapath, do_preprocess=True, filter_user_days=0):
    datapath = Path(datapath)
    bq = pd.read_json(
            datapath / "questions-binary-hackathon.json",
            orient="records",
            convert_dates=False, # necessary, otherwise Pandas messes up date conversion.
        )
    cq = pd.read_json(
        datapath / "questions-continuous-hackathon.json",
        orient="records",
        convert_dates=False, # This is necessary, otherwise Pandas messes up date conversion.
        )
    bp = pd.read_json(
        datapath / "predictions-binary-hackathon.json",
        orient="records",
        )
    cp = pd.read_parquet( # We use parquet here for memory reasons. Locally, you can use JSON.
        datapath / "predictions-continuous-hackathon-v2.parquet",
    )
    if do_preprocess:
        bq, bp, cq, cp = preprocess(bq, bp, cq, cp)
        if filter_user_days > 0:
            bp, cp = filter_users(bp, cp, filter_user_days=filter_user_days)
    return bq, bp, cq, cp


def preprocess(bq, bp, cq, cp):
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

    bq["duration"] = (bq.resolve_time - bq.publish_time) / (60*60*24) # duration in days
    bq.loc[bq.resolution.isnull(), "duration"] = np.nan

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

    # remove predictions done after resolve time
    bp = remove_after_resolve(bq, bp)
    cp = remove_after_resolve(cq, cp)

    # extract top categories
    bq["cat1"] = bq.categories.apply(lambda v: extract_top_cat(v, 0))
    bq["cat2"] = bq.categories.apply(lambda v: extract_top_cat(v, 1))
    cq["cat1"] = cq.categories.apply(lambda v: extract_top_cat(v, 0))
    cq["cat2"] = cq.categories.apply(lambda v: extract_top_cat(v, 1))

    return bq, bp, cq, cp


def midpoints(grid):
    res = []
    for i in range(len(grid)-1):
        res.append((grid[i]+grid[i+1])/2.0)
    return np.array(res)

assert list(midpoints([1,3,5])) == [2,4]

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
        f = interp1d(x_midgrid[pos-1:pos+1], pdf[pos-1:pos+1])
        p = f(res).item()
    return p


def logscore_accuracy(bq, bp, cq, cp):
    """ Add logscore accuracy columns values to the prediction datasets """
    # binary
    xbp = bp.copy()
    if not "resolution" in bp.columns:
        xbp = xbp.merge(bq[["question_id","resolution"]], on=["question_id"], how="left")
    xbp.loc[xbp.resolution.notnull(), "logscore"] = xbp.resolution*(np.log2(xbp.prediction)) + (1-xbp.resolution)*(np.log2(1 - xbp.prediction)) + 1

    # continuous
    xcp = cp.copy()
    cols_to_add = [c for c in ("resolution", "x_grid") if c not in xcp.columns]
    if cols_to_add:
        xcp = xcp.merge(cq[["question_id"] + cols_to_add], on=["question_id"], how="left")

    xcp["proba"] = xcp.apply(lambda s: get_proba(s.x_grid, s.cdf, s.resolution), axis=1)

    # changing zero values to very low values for log to work
    xcp.loc[xcp.proba.eq(0), "proba"] = 1e-300
    xcp.loc[xcp.proba.notnull(), "logscore"] = np.log2(xcp.proba)

    # normalization of logscores to be possible to mix binary and continuous
    xcp["norm_logscore"] = xcp.logscore
    xcp.loc[xcp.logscore.lt(-10), "norm_logscore"] = -10
    xcp.loc[xcp.logscore.gt(0), "norm_logscore"] = 0
    xcp.norm_logscore = (xcp.norm_logscore + 10) / 10

    xbp["norm_logscore"] = xbp.logscore
    xbp.loc[xbp.logscore.lt(-3), "norm_logscore"] = -3
    xbp.loc[xbp.logscore.gt(1), "norm_logscore"] = 1
    xbp.norm_logscore = (xbp.norm_logscore + 3) / 4

    return xbp, xcp


def extract_top_cat(lst, pos):
    """ extract the top category from the list of categories hierarchy """
    if lst is None:
        return ""
    if len(lst) <= pos:
        return ""
    for marker in ["—", "--", "–", "–", "–", "––"]:
        if marker in lst[pos]:
            return lst[pos].split(marker)[0].strip()
    return lst[pos].strip()


def filter_users(bp, cp, filter_user_days):
    cols = ["user_id", "question_id", "t"]
    ap = pd.concat([bp[cols], cp[cols]])
    ap["date"] = ap.t.dt.date
    initial_size = ap.user_id.nunique()
    udp = ap.groupby(["user_id", "date"]).question_id.count().to_frame("n_days").reset_index()
    to_keep = udp[udp.n_days > filter_user_days].index

    bp = bp[bp.user_id.isin(to_keep)].copy()
    cp = cp[cp.user_id.isin(to_keep)].copy()
    print(f"Removed {len(to_keep)} users in days filter from a total of {initial_size} users")
    return bp, cp


def remove_after_resolve(qx, px):
    qx = qx.copy()
    px = px.copy()
    initial_size = len(px)
    px = px.merge(qx[["question_id", "resolve_ts", "resolution"]], on="question_id", how="left")
    px["after_resolve"] = False
    px.loc[px.resolution.notnull() & px.resolve_ts.lt(px.t), "after_resolve"] = True
    px = px[px.after_resolve == False].copy()
    print("Removed in after resolve filter:", initial_size - len(px))
    return px


def user_month_frequency(bp, cp, act_cap=100, refcst_cap=10):
    """ returns frequency statistics for user-month
    pcnt - count of predictions for user
    pcnt_avg - avg count of predictions for all users in period
    per_pX - ratio of predictions that are 2nd, 3-5th, etc predictions for question
    """
    cols = ["user_id", "year", "month", "question_id", "question_seq"]
    ap = pd.concat([bp[cols], cp[cols]])
    # user-month prediction count
    mp = ap.groupby(["user_id", "year", "month"]).question_id.count().to_frame("pcnt").reset_index()
    mp_p2 = ap[ap.question_seq.eq(2)].groupby(["user_id", "year", "month"]).question_id.count().to_frame("per_p2").reset_index()
    mp_p35 = ap[ap.question_seq.between(3,5)].groupby(["user_id", "year", "month"]).question_id.count().to_frame("per_p35").reset_index()
    mp_p610 = ap[ap.question_seq.between(6,10)].groupby(["user_id", "year", "month"]).question_id.count().to_frame("per_p610").reset_index()
    mp_p11p = ap[ap.question_seq.ge(11)].groupby(["user_id", "year", "month"]).question_id.count().to_frame("per_p11p").reset_index()

    period_avg = mp.groupby(["year", "month"]).pcnt.mean().to_frame("pcnt_avg").reset_index()

    mp = mp.merge(mp_p2, on=["user_id", "year", "month"], how="left")
    mp = mp.merge(mp_p35, on=["user_id", "year", "month"], how="left")
    mp = mp.merge(mp_p610, on=["user_id", "year", "month"], how="left")
    mp = mp.merge(mp_p11p, on=["user_id", "year", "month"], how="left")
    mp = mp.fillna(0)

    mp = mp.merge(period_avg, on=["year", "month"], how="left")

    # convert to percentages / ratios
    mp.per_p2 /= mp.pcnt
    mp.per_p35 /=  mp.pcnt
    mp.per_p610 /=  mp.pcnt
    mp.per_p11p /=  mp.pcnt

    # make a normalized activity feature with a cap at act_cap monthly predictions
    mp["act"] = np.log(mp.pcnt)
    mp.loc[mp.act > np.log(act_cap), "act"] = np.log(act_cap)
    mp.act = mp.act / np.log(act_cap)

    # reforcast metric capped at 10
    rfp = ap.groupby(["user_id", "year", "month"]).question_seq.mean().to_frame("qseq").reset_index()
    rfp["refcst"] = np.log(rfp.qseq)
    rfp.loc[rfp.refcst > np.log(refcst_cap), "refcst"] = np.log(refcst_cap)
    rfp.refcst = rfp.refcst / np.log(refcst_cap)

    mp["act_level"] = "act-low"
    mp.loc[mp.act.between(0.1, 0.44), "act_level"] = "act-med"
    mp.loc[mp.act.between(0.44, 0.99), "act_level"] = "act-high"
    mp.loc[mp.act.between(0.99, 1), "act_level"] = "act-veryhigh"

    rfp["rfc_level"] = "rfc-low"
    rfp.loc[rfp.refcst.between(0.1, 0.3), "rfc_level"] = "rfc-med"
    rfp.loc[rfp.refcst.between(0.3, 0.7), "rfc_level"] = "rfc-high"
    rfp.loc[rfp.refcst.between(0.7, 1), "rfc_level"] = "rfc-veryhigh"

    mp = mp.merge(rfp[["user_id", "year", "month", "refcst", "rfc_level"]], on=["user_id", "year", "month"], how="left")

    return mp


def user_month_accuracy(bq, bp, cq, cp):
    """ user-month features related to prediction accuracy
    logscore_avg - average of log score for user prediction on resolved questions
    logscore_cnt - count of user prediction on resolved questions
    logscore_std - std of log score for user prediction on resolved questions
    """
    # binary
    if not "resolution" in bp.columns:
        xbp = bp.merge(bq[["question_id","resolution"]], on=["question_id"], how="left")
    else:
        xbp = bp.copy()
    xbp = xbp[xbp.resolution.notnull()].copy()
    xbp["log_score"] = xbp.resolution*(np.log2(xbp.prediction)) + (1-xbp.resolution)*(np.log2(1 - xbp.prediction)) + 1

    # continuous
    if not "resolution" in cp.columns:
        xcp = cp.merge(cq[["question_id","resolution"]], on=["question_id"], how="left")
    else:
        xcp = cp.copy()
    xcp = xcp[xcp.resolution.notnull()].copy()
    # TODO - FIXME - find way to calculate log score for continuous
    xcp["log_score"] = 0

    xp = pd.concat([xbp, xcp])

    cols = ["user_id", "year", "month"]
    mp = xp.groupby(cols).log_score.mean().to_frame("logscore_avg").reset_index()
    mp_cnt = xp.groupby(cols).log_score.count().to_frame("logscore_cnt").reset_index()
    mp_std = xp.groupby(cols).log_score.std().to_frame("logscore_std").reset_index()

    mp = mp.merge(mp_cnt, on=cols, how="left")
    mp = mp.merge(mp_std, on=cols, how="left")

    nulls =  mp.isnull().sum().sum()
    if nulls > 0:
        logging.warning(f"Unexpected null values {nulls}")
    return mp


def user_month_recency(bp, cp, recency_cap=500):
    cols = ["user_id", "year", "month", "t"]
    ap = pd.concat([bp[cols], cp[cols]])
    ufirst = ap.sort_values("t").groupby("user_id").head(1).rename(columns={"t": "t_first"})
    ap = ap.merge(ufirst[["user_id", "t_first"]], on=["user_id"], how="left")
    # number of days since first user forecast
    ap["delta_first_t"] = (ap.t - ap.t_first).dt.days
    rp = ap.groupby(["user_id", "year", "month"]).delta_first_t.mean().to_frame("recency_raw").reset_index()
    # log of recency capped
    rp["recency"] = np.log1p(rp.recency_raw)
    rp.loc[rp.recency > np.log(recency_cap), "recency"] = np.log(recency_cap)
    rp.recency = rp.recency / np.log(recency_cap)

    rp["rec_level"] = "rec-low"
    rp.loc[rp.recency.between(0.05, 0.6), "rec_level"] = "rec-med"
    rp.loc[rp.recency.between(0.6, 0.98), "rec_level"] = "rec-high"
    rp.loc[rp.recency.between(0.98, 1), "rec_level"] = "rec-veryhigh"

    return rp


def user_month_reputation(bp, cp, rep_min_cap=-0.5, rep_max_cap=0.25):
    cols = ["user_id", "year", "month", "t", "reputation_at_t"]
    ap = pd.concat([bp[cols], cp[cols]])
    rep = ap.groupby(["user_id", "year", "month"]).reputation_at_t.max().to_frame("reputation").reset_index()
    rep.loc[rep.reputation.lt(rep_min_cap), "reputation"] = rep_min_cap
    rep.loc[rep.reputation.gt(rep_max_cap), "reputation"] = rep_max_cap
    rep.reputation += -1*rep_min_cap
    rep.reputation = np.log1p(rep.reputation)
    rep.reputation = rep.reputation / rep.reputation.max()


    rep["rep_level"] = "rep-low"
    rep.loc[rep.reputation.between(0.1, 0.4), "rep_level"] = "rep-med"
    rep.loc[rep.reputation.between(0.4, 1), "rep_level"] = "rep-high"

    return rep


def convert_mid_points(lst):
    return [(lst[i] + lst[i+1]) / 2 for i in range(len(lst)-1)]


assert convert_mid_points([1,3,5]) == [2,4]


def dev_to_mp(s):
    return abs((np.array(convert_mid_points(s.cdf)) - np.array(s.metaculus_cdf)).mean())


def user_month_deviation_mp(bp, cp):
    """ deviation from metaculus aggregated prediction """
    cpx = cp.copy()
    bpx = bp.copy()
    cpx["mp_dev"] = cpx.apply(dev_to_mp, axis=1)
    bpx["mp_dev"] = np.abs(bpx.prediction - bpx.mp)

    cols = ["user_id", "year", "month", "mp_dev"]
    ap = pd.concat([bpx[cols], cpx[cols]])
    devmp = ap.groupby(["user_id", "year", "month"]).mp_dev.mean().to_frame("devmp").reset_index()

    devmp["dmp_level"] = "dmp-low"
    devmp.loc[devmp.devmp.between(0.1, 0.2), "dmp_level"] = "dmp-med"
    devmp.loc[devmp.devmp.between(0.2, 0.4), "dmp_level"] = "dmp-high"
    devmp.loc[devmp.devmp.between(0.4, 1), "dmp_level"] = "dmp-veryhigh"

    return devmp


def forecaster_features_s1(bq, bp, cq, cp):
    """ scenario 1 - combines one feature for each dimension """
    cols = ["user_id", "year", "month"]
    # not ready yet
    #mp_acc = user_month_accuracy(bq, bp, cq, cp)
    act = user_month_frequency(bp, cp)
    act = act[cols +["act", "refcst", "act_level", "rfc_level"]]
    recency = user_month_recency(bp, cp)
    recency = recency[cols + ["recency", "rec_level"]]
    dev = user_month_deviation_mp(bp, cp)
    dev = dev[cols + ["devmp", "dmp_level"]]
    rep = user_month_reputation(bp, cp)
    rep = rep[cols + ["reputation", "rep_level"]]

    act = act.merge(recency, on=["user_id", "year", "month"], how="left")
    act = act.merge(dev, on=["user_id", "year", "month"], how="left")
    act = act.merge(rep, on=["user_id", "year", "month"], how="left")
    return act


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for forecasters clustering")
    parser.add_argument(
        "datapath", type=str, help="path to the data files folder"
    )
    parser.add_argument("--filter-user-days", type=int, default=0, help="filter out users with number of days lower or equal to this")
    parser.add_argument("output", type=str, help="path to the output CSV file")
    args = parser.parse_args()
    bq, bp, cq, cp = load_data(args.datapath, do_preprocess=True, filter_user_days=args.filter_user_days)
    output = forecaster_features_s1(bq, bp, cq, cp)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    output.to_csv(args.output, index=False)