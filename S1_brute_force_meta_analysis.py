#!/usr/bin/env python3
"""
Supplementary File S1 - data and analysis code

Machine Learning and Deep Learning for Brute-Force Attack Detection:
A Systematic Review and Three-Level Meta-Analysis of Reported Performance

This single file contains
  (1) the extracted data: 21 model-level accuracy estimates from 8 studies and
      14 recall estimates from 7 studies, and
  (2) the complete analysis that reproduces Table 6 and the values in the text:
      - logit transformation of accuracy, binomial sampling variance
      - three-level random-effects model (REML), primary analysis
      - model-level DerSimonian-Laird and REML models, HKSJ adjustment
      - study-level model, leave-one-study-out, exclusion of the LSTM outlier
      - subgroup analysis by model family (classical ML, DL, hybrid DL)
      - Egger's regression test for small-study effects
      - three-level meta-analysis of recall (n = number of brute-force records)

Requirements: Python >= 3.9, numpy, scipy   (pip install numpy scipy)

Usage:
  python S1_brute_force_meta_analysis.py                 # print the results report
  python S1_brute_force_meta_analysis.py --csv data.csv  # also write the extracted data (with logit, variance, SE)
  python S1_brute_force_meta_analysis.py --json res.json # also write all results as JSON

Included studies (key: reference):
  fakrullah : Fakrullah et al., Int. J. Innov. Comput. 13(1), 2023. doi:10.11113/ijic.v13n1.386
  alotibi   : Alotibi & Alshammari, IJACSA 14(6), 2023. doi:10.14569/IJACSA.2023.0140612
  akbar     : Akbar et al., Internet Things Artif. Intell. J. 3(3), 2023. doi:10.31763/iota.v3i3.630
  agghey    : Agghey et al., Symmetry 13(11), 2021. doi:10.3390/sym13112192
  odeh      : Odeh & Abu Taleb, Appl. Sci. 13(21), 2023. doi:10.3390/app132111985
  bello     : Bello Suleiman et al., IJISRT, 2024. doi:10.38124/ijisrt/IJISRT24JUL160
  wanjau    : Wanjau et al., IJCATR 10(01), 2021. doi:10.7753/IJCATR1001.1008
  gouda     : Gouda et al., Neural Comput. Appl. 36(6), 2024. doi:10.1007/s00521-023-09309-y
  (Zhang et al., IEEE Access 2020, was included in the review but reported no detection accuracy.)
"""
import argparse
import csv
import io
import json
import sys

import numpy as np
from scipy import optimize, stats

# =============================================================================
# 1. EXTRACTED DATA
# =============================================================================
# Accuracy: study, model, family, accuracy (proportion), n_records (records analysed)
ACCURACY_CSV = """\
study,model,family,accuracy,n_records
fakrullah,DT,ML,0.9995,2830743
fakrullah,k-NN,ML,0.9987,2830743
alotibi,ANN,DL,0.9990,16233002
akbar,RF (primary data),ML,0.9955,1796
akbar,RF (secondary data),ML,0.9977,85002
agghey,DT,ML,0.9993,36273
agghey,RF,ML,0.9994,36273
agghey,NB,ML,0.9985,36273
agghey,k-NN,ML,0.9995,36273
odeh,CNN-LSTM,Hybrid DL,0.9973,4000
odeh,CNN-GRU,Hybrid DL,0.9960,4000
bello,LSTM,DL,0.8520,100000
wanjau,CNN,DL,0.9430,16233002
gouda,SVM,ML,0.9826,100000
gouda,DT,ML,0.9878,100000
gouda,RF,ML,0.9907,100000
gouda,k-NN,ML,0.9816,100000
gouda,CatBoost,ML,0.9904,100000
gouda,GB,ML,0.9880,100000
gouda,LSTM,DL,0.9887,9500003
gouda,CNN,DL,0.9856,9500003
"""

# Recall (sensitivity for the brute-force class): study, model, recall, n_bruteforce
RECALL_CSV = """\
study,model,recall,n_bruteforce
fakrullah,DT,0.9999,13835
fakrullah,k-NN,0.9993,13835
alotibi,ANN,0.9800,10657
akbar,RF (primary data),0.9954,1750
akbar,RF (secondary data),0.9943,3492
odeh,CNN-LSTM,0.9990,484
odeh,CNN-GRU,0.9990,484
bello,LSTM,0.9220,50000
wanjau,CNN,0.9780,10657
gouda,SVM,0.9900,291955
gouda,DT,0.9900,291955
gouda,RF,0.9900,291955
gouda,LSTM,0.9880,291955
gouda,CNN,0.9880,291955
"""


def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


# =============================================================================
# 2. META-ANALYTIC FUNCTIONS
# =============================================================================
def ilogit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_and_variance(p, n):
    """Eq. (2)-(3): y = ln(p/(1-p)),  v = 1 / (n p (1-p))."""
    return np.log(p / (1 - p)), 1.0 / (n * p * (1 - p))


def fixed_effect(y, v):
    """Inverse-variance fixed-effect mean and Cochran's Q (Eq. 5)."""
    w = 1 / v
    m = np.sum(w * y) / w.sum()
    return m, np.sum(w * (y - m) ** 2), w


def tau2_dl(y, v):
    """DerSimonian-Laird between-estimate variance (Eq. 6)."""
    m, Q, w = fixed_effect(y, v)
    k = len(y)
    C = w.sum() - np.sum(w ** 2) / w.sum()
    return max(0.0, (Q - (k - 1)) / C), Q


def tau2_reml(y, v):
    """Restricted maximum-likelihood between-estimate variance."""
    def nll(lt):
        t2 = np.exp(lt)
        W = 1 / (v + t2)
        mu = np.sum(W * y) / W.sum()
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(W.sum()) + np.sum(W * (y - mu) ** 2))
    r = optimize.minimize_scalar(nll, bounds=(-20, 5), method="bounded")
    return float(np.exp(r.x))


def random_effects(y, v, t2, hksj=False):
    """Pooled random-effects estimate, 95% CI and 95% prediction interval (Eq. 9)."""
    W = 1 / (v + t2)
    mu = np.sum(W * y) / W.sum()
    k = len(y)
    if hksj:   # Hartung-Knapp-Sidik-Jonkman adjustment
        q = np.sum(W * (y - mu) ** 2) / (k - 1)
        se = np.sqrt(q / W.sum())
        c = stats.t.ppf(0.975, k - 1)
    else:
        se = np.sqrt(1 / W.sum())
        c = 1.96
    tp = stats.t.ppf(0.975, k - 2)
    pi = (mu - tp * np.sqrt(t2 + se ** 2), mu + tp * np.sqrt(t2 + se ** 2))
    return mu, se, (mu - c * se, mu + c * se), pi


def two_level_summary(y, v, label):
    k = len(y)
    t2d, Q = tau2_dl(y, v)
    t2r = tau2_reml(y, v)
    df = k - 1
    out = {"label": label, "k": k, "Q": Q, "df": df, "pQ": stats.chi2.sf(Q, df),
           "I2": max(0.0, (Q - df) / Q) * 100, "tau2_DL": t2d, "tau2_REML": t2r}
    for name, t2 in (("DL", t2d), ("REML", t2r)):
        mu, se, ci, pi = random_effects(y, v, t2)
        out[name] = dict(logit=mu, se=se, acc=ilogit(mu), ci=[ilogit(ci[0]), ilogit(ci[1])],
                         pi=[ilogit(pi[0]), ilogit(pi[1])])
    mu, se, ci, pi = random_effects(y, v, t2r, hksj=True)
    out["REML_HKSJ"] = dict(logit=mu, acc=ilogit(mu), ci=[ilogit(ci[0]), ilogit(ci[1])])
    out["FE"] = ilogit(fixed_effect(y, v)[0])
    return out


def three_level(y, v, studies):
    """Three-level random-effects model (Eq. 4) fitted by REML.
    Level 2: models within a study (sigma2_within); level 3: studies (sigma2_between)."""
    order = list(dict.fromkeys(studies))
    idx = [np.array([i for i, s in enumerate(studies) if s == st]) for st in order]

    def nll(par):
        s2, s3 = np.exp(par)
        A = B = logdet = 0.0
        blocks = []
        for ix in idx:
            m = len(ix)
            Vi = np.diag(v[ix]) + s2 * np.eye(m) + s3 * np.ones((m, m))
            Vinv = np.linalg.inv(Vi)
            one = np.ones(m)
            blocks.append((ix, Vinv))
            A += one @ Vinv @ one
            B += one @ Vinv @ y[ix]
            logdet += np.linalg.slogdet(Vi)[1]
        mu = B / A
        rss = sum((y[ix] - mu) @ Vinv @ (y[ix] - mu) for ix, Vinv in blocks)
        return 0.5 * (logdet + np.log(A) + rss), mu, 1 / A

    best = min((optimize.minimize(lambda q: nll(q)[0], x0=[a, b], method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
                for a in (-3, 0) for b in (-3, 0)), key=lambda r: r.fun)
    _, mu, var = nll(best.x)
    s2, s3 = np.exp(best.x)
    se = np.sqrt(var)
    k = len(order)
    c = stats.t.ppf(0.975, k - 1)
    pi = (mu - c * np.sqrt(s2 + s3 + var), mu + c * np.sqrt(s2 + s3 + var))
    tv = np.mean(v)   # typical sampling variance, for I2 partitioning
    _, Q = tau2_dl(y, v)
    return dict(logit=mu, se=se, acc=ilogit(mu), ci=[ilogit(mu - c * se), ilogit(mu + c * se)],
                pi=[ilogit(pi[0]), ilogit(pi[1])], sigma2_within=s2, sigma2_between=s3,
                I2_within=100 * s2 / (s2 + s3 + tv), I2_between=100 * s3 / (s2 + s3 + tv),
                Q=Q, df=len(y) - 1, I2_total=max(0.0, (Q - len(y) + 1) / Q) * 100,
                studies=k, estimates=len(y))


# =============================================================================
# 3. ANALYSIS
# =============================================================================
def run():
    A = read_csv(ACCURACY_CSV)
    S = [r["study"] for r in A]
    fam = np.array([r["family"] for r in A])
    p = np.array([float(r["accuracy"]) for r in A])
    n = np.array([float(r["n_records"]) for r in A])
    y, v = logit_and_variance(p, n)
    R = {}

    # Primary analysis
    R["three_level"] = three_level(y, v, S)

    # Model-level (two-level) models
    R["model_level"] = two_level_summary(y, v, "All model estimates")

    # Study-level composites (mean logit; squared mean SE as variance)
    studies = list(dict.fromkeys(S))
    idx = [np.array([i for i, s in enumerate(S) if s == st]) for st in studies]
    ys = np.array([y[ix].mean() for ix in idx])
    vs = np.array([np.mean(np.sqrt(v[ix])) ** 2 for ix in idx])
    R["study_level"] = two_level_summary(ys, vs, "Study-level composites")

    # Leave-one-study-out (study-level REML)
    R["leave_one_out"] = []
    for i, st in enumerate(studies):
        m = np.arange(len(ys)) != i
        mu, se, ci, pi = random_effects(ys[m], vs[m], tau2_reml(ys[m], vs[m]))
        R["leave_one_out"].append(dict(omitted=st, acc=ilogit(mu), ci=[ilogit(ci[0]), ilogit(ci[1])]))

    # Excluding the outlying LSTM study
    keep = np.array([s != "bello" for s in S])
    R["excluding_bello"] = two_level_summary(y[keep], v[keep], "Excluding bello")

    # Subgroups by model family and test for subgroup differences
    R["subgroups"] = {f: two_level_summary(y[fam == f], v[fam == f], f) for f in ("ML", "DL", "Hybrid DL")}
    mus = np.array([R["subgroups"][f]["REML"]["logit"] for f in ("ML", "DL", "Hybrid DL")])
    w = 1 / np.array([R["subgroups"][f]["REML"]["se"] for f in ("ML", "DL", "Hybrid DL")]) ** 2
    Qb = np.sum(w * (mus - np.sum(w * mus) / w.sum()) ** 2)
    R["Q_between"] = dict(Q=Qb, df=2, p=stats.chi2.sf(Qb, 2))

    # Egger's regression test (study level)
    se_s = np.sqrt(vs)
    z, prec = ys / se_s, 1 / se_s
    X = np.vstack([np.ones_like(prec), prec]).T
    b = np.linalg.lstsq(X, z, rcond=None)[0]
    res = z - X @ b
    cov = (res @ res / (len(z) - 2)) * np.linalg.inv(X.T @ X)
    t = b[0] / np.sqrt(cov[0, 0])
    R["egger"] = dict(intercept=b[0], se=np.sqrt(cov[0, 0]), t=t, p=2 * stats.t.sf(abs(t), len(z) - 2))

    # Recall: three-level model, n = number of brute-force records
    Rc = read_csv(RECALL_CSV)
    yr, vr = logit_and_variance(np.array([float(r["recall"]) for r in Rc]),
                                np.array([float(r["n_bruteforce"]) for r in Rc]))
    R["recall_three_level"] = three_level(yr, vr, [r["study"] for r in Rc])

    data = [dict(r, logit=float(y[i]), variance=float(v[i]), se=float(np.sqrt(v[i]))) for i, r in enumerate(A)]
    return R, data


# =============================================================================
# 4. REPORT
# =============================================================================
def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def report(R):
    T, M, SL, RC = R["three_level"], R["model_level"], R["study_level"], R["recall_three_level"]
    ci = lambda d: f"{pct(d['ci'][0])}-{pct(d['ci'][1])}"
    pi = lambda d: f"{pct(d['pi'][0])}-{pct(d['pi'][1], 3)}"
    L = ["=" * 78, "ACCURACY - PRIMARY ANALYSIS (three-level random effects, REML)", "=" * 78,
         f"Studies / estimates : {T['studies']} / {T['estimates']}",
         f"Pooled accuracy     : {pct(T['acc'])}%  (95% CI {ci(T)}; 95% PI {pi(T)})",
         f"Variance components : sigma2(3) = {T['sigma2_between']:.2f}, sigma2(2) = {T['sigma2_within']:.2f}",
         f"I2 partition        : between-study {T['I2_between']:.1f}%, within-study {T['I2_within']:.1f}%",
         f"Cochran's Q         : {T['Q']:,.0f} (df = {T['df']}, p < 0.001)", "",
         "SENSITIVITY ANALYSES", "-" * 78]
    rows = [("Model-level RE (REML)", M["REML"], M["tau2_REML"]),
            ("Model-level RE (DerSimonian-Laird)", M["DL"], M["tau2_DL"]),
            ("Model-level REML + HKSJ", M["REML_HKSJ"], None),
            ("Study-level RE (REML)", SL["REML"], SL["tau2_REML"]),
            ("Study-level REML + HKSJ", SL["REML_HKSJ"], None),
            ("Excluding bello (model-level REML)", R["excluding_bello"]["REML"], R["excluding_bello"]["tau2_REML"])]
    for name, d, t2 in rows:
        L.append(f"{name:<38} {pct(d['acc'])}%  (95% CI {ci(d)})" + (f"  tau2 = {t2:.2f}" if t2 is not None else ""))
    L.append(f"{'Fixed effect (comparison only)':<38} {pct(M['FE'])}%")
    loo = [d["acc"] for d in R["leave_one_out"]]
    L.append(f"{'Leave-one-study-out range':<38} {pct(min(loo))}-{pct(max(loo))}%")
    for d in R["leave_one_out"]:
        L.append(f"    omitting {d['omitted']:<10} {pct(d['acc'])}%  (95% CI {ci(d)})")
    L += ["", "SUBGROUPS BY MODEL FAMILY (model-level REML)", "-" * 78]
    for f, s in R["subgroups"].items():
        L.append(f"{f:<12} k = {s['k']:<3} {pct(s['REML']['acc'])}%  (95% CI {ci(s['REML'])})  I2 = {s['I2']:.1f}%")
    QB = R["Q_between"]
    L.append(f"Test for subgroup differences: Q = {QB['Q']:.2f}, df = 2, p = {QB['p']:.2f}")
    E = R["egger"]
    L += ["", f"Egger's test: intercept = {E['intercept']:.1f} (SE {E['se']:.1f}), p = {E['p']:.2f}", "",
          "=" * 78, "RECALL - SECONDARY ANALYSIS (three-level random effects, REML)", "=" * 78,
          f"Studies / estimates : {RC['studies']} / {RC['estimates']}",
          f"Pooled recall       : {pct(RC['acc'])}%  (95% CI {ci(RC)}; 95% PI {pct(RC['pi'][0], 1)}-{pct(RC['pi'][1])})",
          f"Cochran's Q         : {RC['Q']:,.0f} (df = {RC['df']}), I2 = {RC['I2_total']:.1f}%"]
    return "\n".join(L)


def to_builtin(o):
    if isinstance(o, dict):
        return {k: to_builtin(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", help="write the extracted accuracy data (with logit, variance, SE) to this CSV file")
    ap.add_argument("--json", help="write all results to this JSON file")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    R, data = run()
    print(report(R))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
        print(f"\nData written to {args.csv}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(to_builtin(dict(results=R, data=data)), f, indent=1)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
