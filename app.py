from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

DEFAULT_DATA_FILE = "data.csv"


# ----------------------------
# UI
# ----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #fafafa; color: #111827; }
          section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(15, 23, 42, 0.08);
          }
          .card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 16px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .card h3 { margin: 0 0 8px 0; }
          .muted { color: rgba(17, 24, 39, 0.70); font-size: 0.92rem; }
          .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
          }
          .kpi {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 14px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .kpi .label { color: rgba(17, 24, 39, 0.70); font-size: 0.85rem; margin-bottom: 6px; }
          .kpi .value { font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.1; }
          .kpi .sub { margin-top: 6px; color: rgba(17, 24, 39, 0.65); font-size: 0.85rem; }

          @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
          @media (max-width: 600px) { .kpi-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
      <div class="kpi">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
      </div>
    """


def card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <h3>{title}</h3>
          <div class="muted">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Robust CSV loading
# ----------------------------
@dataclass
class DataMeta:
    source_label: str
    rows: int
    cols: int


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode(errors="ignore")


def _detect_delimiter(sample_text: str) -> str:
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    header = lines[0]
    candidates = [",", ";", "\t", "|"]
    counts = {c: header.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    txt = _decode_bytes(file_bytes[:4096])
    sep = _detect_delimiter(txt)

    # Try detected delimiter first
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception:
        df = None

    # Fallbacks
    if df is None:
        for s in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=s)
                break
            except Exception:
                df = None

    if df is None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # If it loaded as a single column but looks delimited, retry
    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample_val = str(df.iloc[0, 0]) if len(df) else ""
        if ";" in col0 or ";" in sample_val:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
            except Exception:
                pass

    return df


@st.cache_data(show_spinner=False)
def load_dataset(use_uploaded: bool, uploaded_file) -> Tuple[pd.DataFrame, DataMeta]:
    if use_uploaded and uploaded_file is not None:
        raw = uploaded_file.read()
        df = safe_read_csv(raw)
        return df, DataMeta(
            source_label=f"Uploaded file: {uploaded_file.name}",
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
        )

    if not os.path.exists(DEFAULT_DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DATA_FILE} next to app.py. "
            "Place the file in the same folder as app.py, or use the upload option in the sidebar."
        )

    with open(DEFAULT_DATA_FILE, "rb") as f:
        raw = f.read()
    df = safe_read_csv(raw)
    return df, DataMeta(
        source_label=f"Default file: {DEFAULT_DATA_FILE}",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


# ----------------------------
# Target detection / cleaning
# ----------------------------
def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    # Prefer common churn labels
    preferred = [
        "churn",
        "churnstatus",
        "churn_status",
        "exited",
        "attrition",
        "is_churn",
        "target",
        "label",
        "y",
        "left",
        "customerchurn",
    ]
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    hits = []
    for p in preferred:
        for lc, orig in lower_map.items():
            if lc == p or p in lc:
                hits.append(orig)

    # Add any binary-ish column candidates (0/1, yes/no, true/false)
    for c in cols:
        s = df[c]
        if s.dtype == "object":
            vals = set(str(v).strip().lower() for v in s.dropna().unique()[:50])
            if vals.issubset({"0", "1", "yes", "no", "true", "false"}):
                hits.append(c)
        else:
            vals = pd.to_numeric(s, errors="coerce").dropna().unique()
            if 1 <= len(vals) <= 2 and set(vals).issubset({0, 1}):
                hits.append(c)

    # Unique while preserving order
    seen = set()
    out = []
    for c in hits:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def coerce_binary_target(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == "object":
        s2 = s.astype(str).str.strip().str.lower()
        mapped = s2.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.notna().mean() >= 0.6:
            return mapped.fillna(0).astype(int)
    # numeric fallback
    s_num = pd.to_numeric(s, errors="coerce")
    # if values not in {0,1}, try to binarize by >0
    uniq = set(s_num.dropna().unique().tolist())
    if uniq.issubset({0, 1}):
        return s_num.fillna(0).astype(int)
    return (s_num.fillna(0) > 0).astype(int)


def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError("Target column not found in the dataset.")

    y = coerce_binary_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    # Drop constant columns
    nun = X.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    return X, y


# ----------------------------
# Modeling
# ----------------------------
def build_pipeline(model_name: str, X: pd.DataFrame, seed: int) -> Pipeline:
    Xc = X.copy()

    cat_cols = [c for c in Xc.columns if Xc[c].dtype == "object" or str(Xc[c].dtype).startswith("category")]
    num_cols = [c for c in Xc.columns if c not in cat_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
            random_state=seed,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    return Pipeline([("pre", pre), ("clf", clf)])


@st.cache_resource(show_spinner=False)
def train_model_cached(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float,
    seed: int,
):
    X, y = split_X_y(df, target_col)

    # Drop obvious identifiers by heuristic (keeps robustness)
    id_like = [c for c in X.columns if c.lower() in {"id", "customerid", "userid"} or c.lower().endswith("id")]
    if id_like:
        X = X.drop(columns=id_like)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), stratify=y if y.nunique() > 1 else None
    )

    pipe = build_pipeline(model_name, X_train, seed=seed)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    auc = float(roc_auc_score(y_test, proba)) if proba is not None and y_test.nunique() > 1 else float("nan")

    return {
        "pipeline": pipe,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "proba_test": proba,
        "auc": auc,
    }


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict:
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred),
    }


def probability_bands(proba: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    bins = [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    labels = [f"{int(bins[i]*100)}–{int(bins[i+1]*100)}%" for i in range(len(bins) - 1)]
    band = pd.cut(proba, bins=bins, labels=labels, include_lowest=True)
    dfb = pd.DataFrame({"band": band.astype(str), "actual": y})
    out = dfb.groupby("band")["actual"].agg(["count", "mean"]).reset_index()
    out = out.rename(columns={"count": "Customers", "mean": "Observed churn rate"})
    return out


def safe_numeric_corr(x: pd.Series, y: pd.Series) -> float:
    xn = pd.to_numeric(x, errors="coerce")
    if xn.notna().sum() < 10:
        return float("nan")
    a = xn.fillna(xn.median()).to_numpy(dtype=float)
    b = y.to_numpy(dtype=float)
    if np.std(a) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------
# Pages
# ----------------------------
def page_summary(df: pd.DataFrame, meta: DataMeta, target_col: str) -> None:
    st.title("Customer Churn Prediction Dashboard")
    st.caption(meta.source_label)

    y = coerce_binary_target(df[target_col]) if target_col in df.columns else pd.Series([], dtype=int)
    churn_rate = float(y.mean()) if len(y) else float("nan")

    # KPI cards (ONLY on Summary)
    missing_total = int(df.isna().sum().sum())
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    kpis_html = f"""
    <div class="kpi-grid">
      {kpi_card("Rows", f"{n_rows:,}", "Records loaded")}
      {kpi_card("Columns", f"{n_cols:,}", "Fields available")}
      {kpi_card("Churn rate", f"{churn_rate:.1%}" if np.isfinite(churn_rate) else "Not available", f"Target: {target_col}")}
      {kpi_card("Missing values", f"{missing_total:,}", "Across all columns")}
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1.15, 1.0], gap="large")
    with c1:
        card(
            "Dataset overview",
            f"""
            • Target column: {target_col}<br>
            • Positive class interpretation: 1 indicates churn<br>
            • Data quality check: {missing_total:,} missing entries across the table
            """.strip(),
        )
    with c2:
        # quick segment churn if possible
        cat_cols = [c for c in df.columns if c != target_col and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
        if cat_cols:
            seg_col = cat_cols[0]
            tmp = df[[seg_col, target_col]].copy()
            tmp[target_col] = coerce_binary_target(tmp[target_col])
            seg = (
                tmp.groupby(seg_col)[target_col]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={target_col: "Churn rate"})
            )
            st.markdown('<div class="card"><h3>Churn rate by segment</h3>', unsafe_allow_html=True)
            st.dataframe(seg.head(15), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            card("Churn rate by segment", "No categorical columns were found for a grouped churn view.")

    st.write("")
    st.markdown('<div class="card"><h3>Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_exploration(df: pd.DataFrame, target_col: str) -> None:
    st.title("Exploration")
    st.caption("Inspect churn patterns across segments and numeric variables.")

    if target_col not in df.columns:
        card("Data check", "Target column is not available.")
        return

    y = coerce_binary_target(df[target_col])

    st.markdown('<div class="card"><h3>Segment churn</h3>', unsafe_allow_html=True)
    cat_cols = [c for c in df.columns if c != target_col and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
    if cat_cols:
        seg_col = st.selectbox("Group by", cat_cols, index=0)
        tmp = df[[seg_col]].copy()
        tmp["churn"] = y.values
        seg = tmp.groupby(seg_col)["churn"].mean().sort_values(ascending=False).reset_index()
        seg = seg.rename(columns={"churn": "Churn rate"})
        st.dataframe(seg, use_container_width=True, hide_index=True)
    else:
        st.write("No categorical columns were found for segment analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Numeric relationships</h3>', unsafe_allow_html=True)
    num_cols = [c for c in df.columns if c != target_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6]
    if num_cols:
        xcol = st.selectbox("Numeric variable", num_cols, index=0)
        tmp = pd.DataFrame({"x": pd.to_numeric(df[xcol], errors="coerce"), "churn": y})
        tmp = tmp.dropna()
        if len(tmp) >= 30:
            tmp["bin"] = pd.qcut(tmp["x"], q=min(10, max(2, tmp["x"].nunique() // 2)), duplicates="drop")
            view = tmp.groupby("bin")["churn"].mean().reset_index()
            view["bin"] = view["bin"].astype(str)
            view = view.rename(columns={"churn": "Churn rate", "bin": "Bin"})
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.write("Not enough valid values after cleaning to show binned churn rates.")
    else:
        st.write("No suitable numeric columns were found for relationship analysis.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_model(df: pd.DataFrame, target_col: str, model_name: str, test_size: float, seed: int) -> None:
    st.title("Model")
    st.caption("Train a churn model and evaluate performance with adjustable thresholds.")

    artifacts = train_model_cached(df, target_col, model_name, test_size, seed)
    proba = artifacts["proba_test"]
    y_test = artifacts["y_test"].to_numpy()

    st.markdown('<div class="card"><h3>Performance summary</h3>', unsafe_allow_html=True)
    if proba is not None and np.isfinite(artifacts["auc"]):
        st.write(f"ROC AUC: {artifacts['auc']:.3f}")
    else:
        st.write("ROC AUC is not available for the current configuration.")
    st.write(f"Model: {model_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    if proba is None or len(np.unique(y_test)) < 2:
        st.warning("Probability outputs or class diversity are insufficient to show threshold analysis.")
        return

    st.write("")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    m = metrics_at_threshold(y_test, proba, thr)

    st.markdown('<div class="card"><h3>Metrics at selected threshold</h3>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(
        [{
            "Threshold": m["threshold"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
        }]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    cm = m["confusion_matrix"]
    st.write("Confusion matrix (rows = actual, columns = predicted):")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Curves
    st.write("")
    st.markdown('<div class="card"><h3>Curves</h3>', unsafe_allow_html=True)
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig1 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    st.pyplot(fig1, clear_figure=True)

    pr, rc, _ = precision_recall_curve(y_test, proba)
    fig2 = plt.figure()
    plt.plot(rc, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    st.pyplot(fig2, clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Probability bands
    st.write("")
    st.markdown('<div class="card"><h3>Probability bands</h3>', unsafe_allow_html=True)
    bands = probability_bands(proba, y_test)
    st.dataframe(bands, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_insights(df: pd.DataFrame, target_col: str, model_name: str, test_size: float, seed: int) -> None:
    st.title("Insights")
    st.caption("Key findings based on observed data patterns and model behavior.")

    if target_col not in df.columns:
        card("Data check", "Target column is not available.")
        return

    y = coerce_binary_target(df[target_col])

    artifacts = train_model_cached(df, target_col, model_name, test_size, seed)
    pipe = artifacts["pipeline"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"].to_numpy()
    proba = artifacts["proba_test"]

    # 1) Data-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1. Data-driven insights")

    churn_rate = float(y.mean())
    st.write(f"Overall churn rate is {churn_rate:.1%} across {len(df):,} records.")

    # Top segment difference
    cat_cols = [c for c in df.columns if c != target_col and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
    if cat_cols:
        seg_col = cat_cols[0]
        tmp = df[[seg_col]].copy()
        tmp["churn"] = y.values
        seg = tmp.groupby(seg_col)["churn"].mean().sort_values(ascending=False).reset_index()
        if len(seg) >= 2:
            hi = seg.iloc[0]
            lo = seg.iloc[-1]
            st.write(
                f"Churn varies by '{seg_col}'. The highest-churn segment is '{hi[seg_col]}' at {float(hi['churn']):.1%}, "
                f"while the lowest-churn segment is '{lo[seg_col]}' at {float(lo['churn']):.1%}."
            )
        st.dataframe(seg.rename(columns={"churn": "Churn rate"}).head(20), use_container_width=True, hide_index=True)
    else:
        st.write("No categorical columns were found for segment-level churn insights.")

    # Numeric correlations snapshot
    num_cols = [c for c in df.columns if c != target_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6]
    if num_cols:
        corrs = []
        for c in num_cols[:40]:
            corrs.append((c, safe_numeric_corr(df[c], y)))
        corr_df = pd.DataFrame(corrs, columns=["Variable", "Correlation with churn"]).dropna()
        corr_df = corr_df.sort_values("Correlation with churn", ascending=False)
        if len(corr_df) > 0:
            st.write("Correlation snapshot (directional, not causal):")
            st.dataframe(corr_df.head(15), use_container_width=True, hide_index=True)
    else:
        st.write("No suitable numeric columns were found for correlation insights.")

    # Descriptive stats
    if num_cols:
        desc = df[num_cols[:20]].apply(pd.to_numeric, errors="coerce").describe().T.reset_index().rename(columns={"index": "Feature"})
        st.write("Descriptive statistics for numeric features:")
        st.dataframe(desc, use_container_width=True, hide_index=True)
    else:
        st.write("Descriptive statistics are not available because numeric features could not be detected.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 2) Model-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2. Model-driven insights")

    if proba is not None and len(np.unique(y_test)) > 1 and np.isfinite(artifacts["auc"]):
        st.write(f"The model achieves ROC AUC of {artifacts['auc']:.3f} on the test split.")
    else:
        st.write("Model probability-based evaluation is limited for the current configuration.")

    # Feature importance
    st.subheader("Feature importance")
    try:
        # Prefer direct coefficient importance for logistic regression (stable, fast)
        if model_name == "Logistic Regression":
            pre = pipe.named_steps["pre"]
            clf = pipe.named_steps["clf"]
            feature_names = []
            # Build feature names from transformers
            for name, transformer, cols in pre.transformers_:
                if name == "num":
                    feature_names.extend(list(cols))
                elif name == "cat":
                    ohe = transformer.named_steps["onehot"]
                    ohe_names = ohe.get_feature_names_out(cols)
                    feature_names.extend(list(ohe_names))
            coef = clf.coef_.ravel()
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coef)})
            imp_df = imp_df.sort_values("Importance", ascending=False)
            st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)
        else:
            # Permutation importance on raw columns (pipeline handles preprocessing)
            X_imp = X_test.copy()
            y_imp = y_test.copy()
            if len(X_imp) > 800:
                X_imp = X_imp.sample(800, random_state=seed)
                y_imp = y_imp[: len(X_imp)]
            perm = permutation_importance(pipe, X_imp, y_imp, n_repeats=5, random_state=seed, scoring="roc_auc")
            imp_df = pd.DataFrame({"Feature": X_imp.columns, "Importance (AUC decrease)": perm.importances_mean})
            imp_df = imp_df.sort_values("Importance (AUC decrease)", ascending=False)
            st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)
    except Exception as e:
        st.write("Feature importance could not be computed for the current setup.")
        st.write(f"Details: {e}")

    # Probability bands
    st.subheader("Probability bands")
    if proba is not None and len(proba) == len(y_test):
        bands = probability_bands(proba, y_test)
        st.dataframe(bands, use_container_width=True, hide_index=True)
    else:
        st.write("Probability bands are not available for the current model output.")

    # Threshold tradeoffs
    st.subheader("Threshold tradeoffs")
    if proba is not None and len(np.unique(y_test)) > 1:
        thr = st.slider("Operational threshold", 0.05, 0.95, 0.50, 0.01, key="ins_thr")
        mt = metrics_at_threshold(y_test, proba, thr)
        trade = pd.DataFrame([{
            "Threshold": thr,
            "Accuracy": mt["accuracy"],
            "Precision": mt["precision"],
            "Recall": mt["recall"],
            "F1": mt["f1"],
        }])
        st.dataframe(trade, use_container_width=True, hide_index=True)
        cm = mt["confusion_matrix"]
        st.write("Confusion matrix (rows = actual, columns = predicted):")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
        st.write(
            "Higher thresholds typically increase precision but reduce recall. "
            "Choose a threshold based on the relative cost of false positives versus false negatives."
        )
    else:
        st.write("Threshold tradeoffs are not available because probability outputs or class diversity are insufficient.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_data(df: pd.DataFrame, meta: DataMeta) -> None:
    st.title("Data")
    st.caption(meta.source_label)

    st.markdown('<div class="card"><h3>Column health</h3>', unsafe_allow_html=True)
    miss = df.isna().mean().sort_values(ascending=False).reset_index()
    miss.columns = ["Column", "Missing share"]
    st.dataframe(miss, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(300), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Summary", "Exploration", "Model", "Insights", "Data"], index=0)

    st.sidebar.write("")
    st.sidebar.subheader("Data loading")
    use_upload = st.sidebar.toggle("Use uploaded file instead of default", value=False)
    uploaded = st.sidebar.file_uploader(
        "Upload a dataset file",
        type=["csv"],
        help="CSV only. The loader auto-detects common delimiters.",
    )

    try:
        df, meta = load_dataset(use_upload, uploaded)
    except Exception as e:
        st.error("Data could not be loaded.")
        st.write(str(e))
        return

    if df.shape[0] == 0:
        st.error("The dataset is empty.")
        return

    # Target selection (auto-detect, with safe fallback)
    candidates = detect_target_candidates(df)
    default_target = candidates[0] if candidates else df.columns[-1]
    st.sidebar.subheader("Model settings")
    target_col = st.sidebar.selectbox("Target column", options=list(df.columns), index=list(df.columns).index(default_target))
    model_name = st.sidebar.selectbox("Model type", ["Logistic Regression", "Random Forest"], index=0)
    test_size = st.sidebar.slider("Test size", 0.15, 0.40, 0.25, 0.01)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    # Route
    if page == "Summary":
        page_summary(df, meta, target_col)
    elif page == "Exploration":
        page_exploration(df, target_col)
    elif page == "Model":
        page_model(df, target_col, model_name, test_size, seed)
    elif page == "Insights":
        page_insights(df, target_col, model_name, test_size, seed)
    elif page == "Data":
        page_data(df, meta)


if __name__ == "__main__":
    main()
