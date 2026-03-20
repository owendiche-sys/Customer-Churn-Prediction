
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_FILE = "data.csv"
APP_TITLE = "Customer Churn Intelligence Dashboard"


# ----------------------------
# Styling
# ----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .hero {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            padding: 28px 28px 22px 28px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: #2563eb;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2rem;
            line-height: 1.05;
        }
        .hero p {
            margin: 0.75rem 0 0 0;
            color: #475569;
            font-size: 1rem;
            max-width: 900px;
            line-height: 1.65;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 1rem 0 1.2rem 0;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .metric-label {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .metric-sub {
            color: #64748b;
            font-size: 0.84rem;
            margin-top: 0.35rem;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .section-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
            line-height: 1.55;
        }
        .insight-list {
            margin: 0;
            padding-left: 1.1rem;
            color: #334155;
            line-height: 1.7;
        }
        .callout {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 14px 16px;
            color: #1e3a8a;
            line-height: 1.6;
        }
        .small-note {
            color: #64748b;
            font-size: 0.85rem;
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        @media (max-width: 1100px) {
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .metric-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .hero h1 {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}'
        '</div>'
    )


def section_open(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def section_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def fmt_feature(name: str) -> str:
    clean = re.sub(r"[_\-]+", " ", str(name))
    clean = re.sub(r"(?<!^)(?=[A-Z])", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:1].upper() + clean[1:] if clean else str(name)


def format_pct(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "Not available"


def light_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.06)")
    return fig


# ----------------------------
# Data loading
# ----------------------------
@dataclass
class DataMeta:
    source_label: str
    rows: int
    cols: int


def _decode_bytes(blob: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return blob.decode(enc)
        except Exception:
            continue
    return blob.decode(errors="ignore")


def _detect_delimiter(sample_text: str) -> str:
    lines = [line for line in sample_text.splitlines() if line.strip()]
    if not lines:
        return ","
    header = lines[0]
    candidates = [",", ";", "\t", "|"]
    counts = {c: header.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    sample = _decode_bytes(file_bytes[:4096])
    sep = _detect_delimiter(sample)

    df = None
    for current_sep in [sep, ";", ",", "\t", "|"]:
        if df is not None:
            break
        try:
            candidate = pd.read_csv(io.BytesIO(file_bytes), sep=current_sep)
            df = candidate
        except Exception:
            df = None

    if df is None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    if df.shape[1] == 1 and len(df.columns) > 0:
        col_name = str(df.columns[0])
        cell_sample = str(df.iloc[0, 0]) if len(df) else ""
        for retry_sep in [";", ",", "\t", "|"]:
            if retry_sep in col_name or retry_sep in cell_sample:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=retry_sep)
                    break
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
            f"{DEFAULT_DATA_FILE} was not found next to app.py. "
            "Place the file beside the app or turn on upload mode in the sidebar."
        )

    with open(DEFAULT_DATA_FILE, "rb") as handle:
        raw = handle.read()

    df = safe_read_csv(raw)
    return df, DataMeta(
        source_label=f"Default file: {DEFAULT_DATA_FILE}",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


# ----------------------------
# Cleaning and target handling
# ----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]

    missing_tokens = {
        "",
        " ",
        "na",
        "n/a",
        "null",
        "none",
        "nan",
        "missing",
        "unknown",
        "?",
        "-",
    }

    for col in out.columns:
        if out[col].dtype == "object":
            series = out[col].astype(str).str.strip()
            series = series.mask(series.str.lower().isin(missing_tokens))
            # Common Telco pattern: TotalCharges comes in as object with spaces.
            numeric_candidate = pd.to_numeric(series.str.replace(",", "", regex=False), errors="coerce")
            non_missing_share = series.notna().mean()
            numeric_success = numeric_candidate.notna().mean()
            if non_missing_share > 0 and numeric_success >= 0.85:
                out[col] = numeric_candidate
            else:
                out[col] = series.replace("nan", np.nan)
        elif str(out[col].dtype).lower() == "bool":
            out[col] = out[col].astype(int)

    return out


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    preferred = [
        "churn",
        "churnstatus",
        "churn_status",
        "exited",
        "attrition",
        "is_churn",
        "left",
        "customerchurn",
        "target",
        "label",
        "y",
    ]
    cols = list(df.columns)
    lower_map = {str(col).lower(): col for col in cols}

    hits: List[str] = []
    for preferred_name in preferred:
        for lower_col, original_col in lower_map.items():
            if lower_col == preferred_name or preferred_name in lower_col:
                hits.append(original_col)

    for col in cols:
        series = df[col]
        if series.dtype == "object":
            values = set(str(v).strip().lower() for v in series.dropna().unique()[:50])
            if values and values.issubset({"0", "1", "yes", "no", "true", "false", "churned", "stayed"}):
                hits.append(col)
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna().unique()
            if 1 <= len(numeric) <= 2 and set(np.unique(numeric)).issubset({0, 1}):
                hits.append(col)

    seen = set()
    ordered = []
    for col in hits:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "churned": 1,
                    "stayed": 0,
                }
            )
        )
        if mapped.notna().mean() >= 0.6:
            return mapped.fillna(0).astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    unique_values = set(numeric.dropna().unique().tolist())
    if unique_values.issubset({0, 1}) and unique_values:
        return numeric.fillna(0).astype(int)
    return (numeric.fillna(0) > 0).astype(int)


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError("The selected target column is not present.")
    y = coerce_binary_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    drop_cols: List[str] = []
    for col in X.columns:
        lower = str(col).lower()
        if lower in {"id", "customerid", "customer_id", "userid", "user_id"} or lower.endswith("id"):
            drop_cols.append(col)
            continue
        if X[col].dtype == "object":
            n_unique = X[col].nunique(dropna=True)
            if len(X) > 0 and n_unique > 50 and (n_unique / max(len(X), 1)) > 0.4:
                drop_cols.append(col)

    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y


def get_column_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


# ----------------------------
# Modeling
# ----------------------------
def build_pipeline(model_name: str, X: pd.DataFrame, seed: int) -> Pipeline:
    numeric_cols, categorical_cols = get_column_groups(X)

    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=3000,
            random_state=seed,
            class_weight="balanced",
            solver="lbfgs",
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )

    return Pipeline(steps=[("pre", preprocessor), ("clf", estimator)])


def threshold_table(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_true, pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "predicted_positive_share": float(pred.mean()) if len(pred) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confusion_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])


def build_risk_bands(proba: np.ndarray, y_true: np.ndarray, bands: int = 5) -> pd.DataFrame:
    labels = ["Very low", "Low", "Moderate", "High", "Very high"][:bands]
    cut = pd.qcut(proba, q=bands, labels=labels, duplicates="drop")
    band_df = pd.DataFrame({"band": pd.Categorical(cut, categories=labels, ordered=True), "actual": y_true, "score": proba})
    summary = (
        band_df.groupby("band", observed=False, sort=False)
        .agg(
            customers=("actual", "size"),
            observed_churn_rate=("actual", "mean"),
            avg_model_score=("score", "mean"),
        )
        .reset_index()
    )
    summary["band"] = summary["band"].astype(str)
    order_map = {label: i for i, label in enumerate(labels)}
    summary["band_order"] = summary["band"].map(order_map)
    summary = summary.sort_values("band_order").drop(columns="band_order").reset_index(drop=True)
    return summary


def probability_label(probability: float) -> str:
    if probability >= 0.75:
        return "Very high risk"
    if probability >= 0.50:
        return "High risk"
    if probability >= 0.30:
        return "Moderate risk"
    if probability >= 0.15:
        return "Low risk"
    return "Very low risk"


@st.cache_resource(show_spinner=False)
def fit_analysis_bundle(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int,
) -> Dict[str, Any]:
    X, y = split_features_target(df, target_col)

    if y.nunique() < 2:
        raise ValueError("The selected target column must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    model_names = ["Logistic Regression", "Random Forest"]
    results: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        pipeline = build_pipeline(model_name, X_train, seed)
        pipeline.fit(X_train, y_train)
        proba_test = pipeline.predict_proba(X_test)[:, 1]
        pred_default = (proba_test >= 0.50).astype(int)

        results[model_name] = {
            "pipeline": pipeline,
            "proba_test": proba_test,
            "metrics": {
                "roc_auc": roc_auc_score(y_test, proba_test),
                "average_precision": average_precision_score(y_test, proba_test),
                "accuracy": accuracy_score(y_test, pred_default),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred_default),
                "precision": precision_score(y_test, pred_default, zero_division=0),
                "recall": recall_score(y_test, pred_default, zero_division=0),
                "f1": f1_score(y_test, pred_default, zero_division=0),
                "brier": brier_score_loss(y_test, proba_test),
            },
            "thresholds": threshold_table(y_test.to_numpy(), proba_test),
        }

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "ROC AUC": result["metrics"]["roc_auc"],
                "Average precision": result["metrics"]["average_precision"],
                "Accuracy @ 0.50": result["metrics"]["accuracy"],
                "Balanced accuracy @ 0.50": result["metrics"]["balanced_accuracy"],
                "Precision @ 0.50": result["metrics"]["precision"],
                "Recall @ 0.50": result["metrics"]["recall"],
                "F1 @ 0.50": result["metrics"]["f1"],
                "Brier loss": result["metrics"]["brier"],
            }
            for name, result in results.items()
        ]
    ).sort_values("ROC AUC", ascending=False)

    champion_name = comparison.iloc[0]["Model"]
    champion = results[champion_name]
    thresholds = champion["thresholds"].copy()
    recommended_row = thresholds.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    recommended_threshold = float(recommended_row["threshold"])

    X_imp = X_test.copy()
    y_imp = y_test.copy()
    if len(X_imp) > 1000:
        sampled_idx = X_imp.sample(1000, random_state=seed).index
        X_imp = X_imp.loc[sampled_idx]
        y_imp = y_imp.loc[sampled_idx]

    perm = permutation_importance(
        champion["pipeline"],
        X_imp,
        y_imp,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": X_imp.columns,
                "importance": perm.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["feature_label"] = importance_df["feature"].map(fmt_feature)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_results": results,
        "comparison": comparison,
        "champion_name": champion_name,
        "champion_pipeline": champion["pipeline"],
        "champion_proba": champion["proba_test"],
        "recommended_threshold": recommended_threshold,
        "importance": importance_df,
        "risk_bands": build_risk_bands(champion["proba_test"], y_test.to_numpy(), bands=5),
    }


# ----------------------------
# Data insight helpers
# ----------------------------
def compute_segment_table(df: pd.DataFrame, target_col: str, min_count: int = 25) -> pd.DataFrame:
    y = coerce_binary_target(df[target_col])
    overall_rate = float(y.mean())
    rows: List[Dict[str, Any]] = []

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        n_unique = series.nunique(dropna=True)
        if not (
            series.dtype == "object"
            or str(series.dtype).startswith("category")
            or n_unique <= 12
        ):
            continue

        working = pd.DataFrame(
            {
                "segment": series.fillna("Missing").astype(str),
                "target": y,
            }
        )
        grouped = (
            working.groupby("segment", observed=False)["target"]
            .agg(["size", "mean"])
            .reset_index()
            .rename(columns={"size": "customers", "mean": "churn_rate"})
        )
        grouped = grouped[grouped["customers"] >= min_count]

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": col,
                    "feature_label": fmt_feature(col),
                    "segment": row["segment"],
                    "customers": int(row["customers"]),
                    "churn_rate": float(row["churn_rate"]),
                    "lift_vs_overall": float(row["churn_rate"] / overall_rate) if overall_rate > 0 else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "feature_label", "segment", "customers", "churn_rate", "lift_vs_overall"]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["churn_rate", "customers"], ascending=[False, False]).reset_index(drop=True)
    return out


def compute_numeric_band_table(df: pd.DataFrame, target_col: str, max_features: int = 4) -> pd.DataFrame:
    y = coerce_binary_target(df[target_col])
    records: List[Dict[str, Any]] = []

    for col in df.columns:
        if col == target_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() < 0.75:
            continue
        valid = pd.DataFrame({"x": numeric, "target": y}).dropna()
        if len(valid) < 50 or valid["x"].nunique() < 5:
            continue
        try:
            valid["band"] = pd.qcut(valid["x"], q=4, duplicates="drop")
        except ValueError:
            continue
        grouped = (
            valid.groupby("band", observed=False)["target"]
            .agg(["size", "mean"])
            .reset_index()
            .rename(columns={"size": "customers", "mean": "churn_rate"})
        )
        spread = float(grouped["churn_rate"].max() - grouped["churn_rate"].min())
        for _, row in grouped.iterrows():
            records.append(
                {
                    "feature": col,
                    "feature_label": fmt_feature(col),
                    "band": str(row["band"]),
                    "customers": int(row["customers"]),
                    "churn_rate": float(row["churn_rate"]),
                    "spread": spread,
                }
            )

    if not records:
        return pd.DataFrame(columns=["feature", "feature_label", "band", "customers", "churn_rate", "spread"])

    out = pd.DataFrame(records)
    top_features = (
        out[["feature", "spread"]]
        .drop_duplicates()
        .sort_values("spread", ascending=False)
        .head(max_features)["feature"]
        .tolist()
    )
    return out[out["feature"].isin(top_features)].copy()


def key_numeric_snapshot(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    y = coerce_binary_target(df[target_col])
    candidate_cols = []
    preferred_names = ["tenure", "monthlycharges", "totalcharges", "monthly_charge", "revenue"]
    lower_map = {str(col).lower(): col for col in df.columns if col != target_col}
    for pref in preferred_names:
        for lower_col, original_col in lower_map.items():
            if pref in lower_col:
                candidate_cols.append(original_col)

    if not candidate_cols:
        numeric_candidates = [
            col for col in df.columns
            if col != target_col and pd.to_numeric(df[col], errors="coerce").notna().mean() >= 0.75
        ]
        candidate_cols = numeric_candidates[:3]

    rows: List[Dict[str, Any]] = []
    for col in list(dict.fromkeys(candidate_cols))[:3]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "Metric": fmt_feature(col),
                "Stayed": numeric[y == 0].mean(),
                "Churned": numeric[y == 1].mean(),
                "Difference": numeric[y == 1].mean() - numeric[y == 0].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_data_insight_sentences(df: pd.DataFrame, target_col: str) -> List[str]:
    y = coerce_binary_target(df[target_col])
    overall = float(y.mean())
    segments = compute_segment_table(df, target_col, min_count=max(20, int(len(df) * 0.02)))
    numeric_bands = compute_numeric_band_table(df, target_col, max_features=2)

    insights = [
        f"Observed churn rate is {overall:.1%} across {len(df):,} customer records."
    ]

    if not segments.empty:
        highest = segments.iloc[0]
        lowest = segments.sort_values("churn_rate", ascending=True).iloc[0]
        insights.append(
            f"The most exposed segment is {highest['feature_label']} = {highest['segment']}, with churn at {highest['churn_rate']:.1%} across {highest['customers']:,} customers."
        )
        insights.append(
            f"The most stable segment is {lowest['feature_label']} = {lowest['segment']}, where churn falls to {lowest['churn_rate']:.1%}."
        )

    if not numeric_bands.empty:
        top_feature = (
            numeric_bands[["feature_label", "spread"]]
            .drop_duplicates()
            .sort_values("spread", ascending=False)
            .iloc[0]
        )
        feature_rows = numeric_bands[numeric_bands["feature_label"] == top_feature["feature_label"]].sort_values("churn_rate")
        low_row = feature_rows.iloc[0]
        high_row = feature_rows.iloc[-1]
        insights.append(
            f"{top_feature['feature_label']} shows one of the clearest gradients: churn rises from {low_row['churn_rate']:.1%} in the lowest-risk band to {high_row['churn_rate']:.1%} in the highest-risk band."
        )

    return insights


def build_model_insight_sentences(bundle: Dict[str, Any]) -> List[str]:
    comparison = bundle["comparison"]
    champion_name = bundle["champion_name"]
    champion_row = comparison[comparison["Model"] == champion_name].iloc[0]
    threshold = bundle["recommended_threshold"]
    threshold_row = (
        bundle["model_results"][champion_name]["thresholds"]
        .loc[lambda x: np.isclose(x["threshold"], threshold)]
        .iloc[0]
    )

    statements = [
        f"{champion_name} is the strongest holdout model with ROC AUC of {champion_row['ROC AUC']:.3f} and average precision of {champion_row['Average precision']:.3f}."
    ]
    statements.append(
        f"At the recommended threshold of {threshold:.2f}, precision is {threshold_row['precision']:.2f}, recall is {threshold_row['recall']:.2f}, and F1 is {threshold_row['f1']:.2f}."
    )

    importance = bundle["importance"]
    if not importance.empty:
        top_features = importance.head(3)["feature_label"].tolist()
        statements.append(
            f"The model is most sensitive to shifts in {', '.join(top_features[:-1]) + ', and ' + top_features[-1] if len(top_features) >= 3 else ', '.join(top_features)}."
        )
    return statements


# ----------------------------
# Scenario tools
# ----------------------------
def scenario_feature_list(bundle: Dict[str, Any]) -> List[str]:
    importance = bundle["importance"]["feature"].tolist()
    X = bundle["X"]
    chosen: List[str] = []
    for feature in importance:
        if feature in X.columns and feature not in chosen:
            if X[feature].dtype == "object":
                if X[feature].nunique(dropna=True) <= 12:
                    chosen.append(feature)
            else:
                chosen.append(feature)
        if len(chosen) >= 5:
            break

    if len(chosen) < 5:
        for col in X.columns:
            if col not in chosen:
                chosen.append(col)
            if len(chosen) >= 5:
                break
    return chosen[:5]


def scenario_editor(bundle: Dict[str, Any], threshold: float) -> None:
    X_test = bundle["X_test"].copy()
    champion = bundle["champion_pipeline"]
    features = scenario_feature_list(bundle)

    if X_test.empty or not features:
        st.info("Scenario tools are not available for the current dataset.")
        return

    profile_index = st.selectbox(
        "Reference customer",
        options=list(range(len(X_test))),
        index=0,
        format_func=lambda idx: f"Customer profile {idx + 1}",
    )

    base_row = X_test.iloc[[profile_index]].copy()
    edited_row = base_row.copy()

    st.markdown('<div class="small-note">Edit a few of the most important profile fields to see how risk changes.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            label = fmt_feature(feature)
            series = bundle["X"][feature]

            if series.dtype == "object":
                options = [str(v) for v in series.dropna().astype(str).unique().tolist()]
                options = sorted(options)
                current = str(base_row.iloc[0][feature]) if pd.notna(base_row.iloc[0][feature]) else (options[0] if options else "")
                if current not in options and current != "nan":
                    options = [current] + options
                value = st.selectbox(label, options=options, index=max(options.index(current), 0) if options else 0)
                edited_row.at[edited_row.index[0], feature] = value
            else:
                numeric_series = pd.to_numeric(series, errors="coerce")
                clean = numeric_series.dropna()
                current_val = pd.to_numeric(base_row.iloc[0][feature], errors="coerce")
                if clean.empty:
                    st.write(f"{label}: no valid values available")
                    continue

                lower = float(clean.quantile(0.05))
                upper = float(clean.quantile(0.95))
                if lower == upper:
                    lower = float(clean.min())
                    upper = float(clean.max())

                if pd.isna(current_val):
                    current_val = float(clean.median())

                if clean.nunique() <= 12:
                    options = sorted(clean.unique().tolist())
                    current_val = min(options, key=lambda x: abs(float(x) - float(current_val)))
                    value = st.selectbox(label, options=options, index=options.index(current_val))
                else:
                    step = max((upper - lower) / 100.0, 0.01)
                    value = st.slider(
                        label,
                        min_value=float(lower),
                        max_value=float(upper),
                        value=float(np.clip(current_val, lower, upper)),
                        step=float(step),
                    )
                edited_row.at[edited_row.index[0], feature] = value

    base_probability = float(champion.predict_proba(base_row)[0, 1])
    edited_probability = float(champion.predict_proba(edited_row)[0, 1])
    delta = edited_probability - base_probability

    result_cols = st.columns([1.1, 1.2])
    with result_cols[0]:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=edited_probability * 100,
                number={"suffix": "%"},
                title={"text": "Predicted churn probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 15], "color": "#eff6ff"},
                        {"range": [15, 30], "color": "#dbeafe"},
                        {"range": [30, 50], "color": "#bfdbfe"},
                        {"range": [50, 75], "color": "#93c5fd"},
                        {"range": [75, 100], "color": "#60a5fa"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 4},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=330, margin=dict(l=15, r=15, t=60, b=10), paper_bgcolor="white")
        st.plotly_chart(gauge, use_container_width=True)

    with result_cols[1]:
        metric_html = f"""
        <div class="metric-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
            {metric_card("Base profile risk", f"{base_probability:.1%}", probability_label(base_probability))}
            {metric_card("Scenario risk", f"{edited_probability:.1%}", probability_label(edited_probability))}
            {metric_card("Risk change", f"{delta:+.1%}", "Versus the base profile")}
            {metric_card("Decision threshold", f"{threshold:.2f}", "Current operating cutoff")}
        </div>
        """
        st.markdown(metric_html, unsafe_allow_html=True)

        state_text = "above" if edited_probability >= threshold else "below"
        st.markdown(
            f"""
            <div class="callout">
            This scenario sits <strong>{state_text}</strong> the current operating threshold and is classified as
            <strong>{probability_label(edited_probability).lower()}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview = pd.DataFrame(
            {
                "Field": [fmt_feature(col) for col in features],
                "Base profile": [base_row.iloc[0][col] for col in features],
                "Scenario": [edited_row.iloc[0][col] for col in features],
            }
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)


# ----------------------------
# View builders
# ----------------------------
def render_hero(meta: DataMeta) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Portfolio dashboard</div>
            <h1>{APP_TITLE}</h1>
            <p>
                Insight-first churn analysis with customer risk segmentation, model comparison,
                operating-threshold evaluation, and scenario testing. Source: {meta.source_label}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(df: pd.DataFrame, bundle: Dict[str, Any], target_col: str) -> None:
    y = coerce_binary_target(df[target_col])
    overall_rate = float(y.mean())
    monthly_col = next((col for col in df.columns if "monthly" in str(col).lower() and "charge" in str(col).lower()), None)
    tenure_col = next((col for col in df.columns if "tenure" in str(col).lower()), None)

    high_risk_count = int((bundle["champion_proba"] >= bundle["recommended_threshold"]).sum())
    high_risk_share = high_risk_count / len(bundle["champion_proba"]) if len(bundle["champion_proba"]) else np.nan
    champion_auc = bundle["comparison"].iloc[0]["ROC AUC"]

    monthly_value = (
        pd.to_numeric(df[monthly_col], errors="coerce").median()
        if monthly_col is not None
        else np.nan
    )
    tenure_value = (
        pd.to_numeric(df[tenure_col], errors="coerce").median()
        if tenure_col is not None
        else np.nan
    )

    cards = [
        metric_card("Customer base", f"{len(df):,}", "Records available for analysis"),
        metric_card("Observed churn rate", f"{overall_rate:.1%}", f"Target column: {fmt_feature(target_col)}"),
    ]
    if pd.notna(monthly_value):
        cards.append(metric_card("Median monthly charge", f"{monthly_value:,.2f}", "Central commercial profile"))
    if pd.notna(tenure_value):
        cards.append(metric_card("Median tenure", f"{tenure_value:,.1f}", "Typical customer lifespan"))
    cards.extend(
        [
            metric_card("Customers flagged at threshold", f"{high_risk_count:,}", f"{high_risk_share:.1%} of the scored holdout sample"),
            metric_card("Champion model ROC AUC", f"{champion_auc:.3f}", bundle["champion_name"]),
        ]
    )

    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.05], gap="large")
    with left:
        section_open(
            "Data-led summary",
            "Start with what the customer base is already telling us before introducing model output.",
        )
        insights = build_data_insight_sentences(df, target_col)
        st.markdown("<ul class='insight-list'>" + "".join([f"<li>{item}</li>" for item in insights]) + "</ul>", unsafe_allow_html=True)
        segment_table = compute_segment_table(df, target_col, min_count=max(20, int(len(df) * 0.02)))
        if not segment_table.empty:
            top_segments = segment_table.head(8).copy()
            top_segments["label"] = top_segments["feature_label"] + " = " + top_segments["segment"].astype(str)
            fig = px.bar(
                top_segments.sort_values("churn_rate"),
                x="churn_rate",
                y="label",
                orientation="h",
                text="customers",
                labels={"churn_rate": "Churn rate", "label": "High-risk segment"},
                title="Highest-risk customer segments",
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(height=380)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(light_plotly(fig), use_container_width=True)
        else:
            st.info("No segment-level churn pattern could be computed from the available fields.")
        section_close()

    with right:
        section_open(
            "Model-led summary",
            "Model output is presented after the observed customer patterns so the dashboard remains insight-led.",
        )
        model_insights = build_model_insight_sentences(bundle)
        st.markdown("<ul class='insight-list'>" + "".join([f"<li>{item}</li>" for item in model_insights]) + "</ul>", unsafe_allow_html=True)

        risk_bands = bundle["risk_bands"].copy()
        if not risk_bands.empty:
            fig = px.bar(
                risk_bands,
                x="band",
                y="observed_churn_rate",
                text="customers",
                labels={"band": "Predicted risk band", "observed_churn_rate": "Observed churn rate"},
                title="Observed churn rate by model risk band",
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=380)
            st.plotly_chart(light_plotly(fig), use_container_width=True)

        threshold = bundle["recommended_threshold"]
        st.markdown(
            f"""
            <div class="callout">
                The dashboard recommends an operating threshold of <strong>{threshold:.2f}</strong> because it produces
                the strongest balance of precision and recall on the holdout sample.
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_close()


def render_data_insights(df: pd.DataFrame, target_col: str) -> None:
    segment_table = compute_segment_table(df, target_col, min_count=max(20, int(len(df) * 0.02)))
    numeric_bands = compute_numeric_band_table(df, target_col, max_features=4)
    snapshot = key_numeric_snapshot(df, target_col)

    row1_col1, row1_col2 = st.columns([1.15, 0.95], gap="large")
    with row1_col1:
        section_open(
            "Where churn concentrates",
            "This view ranks the customer groups with the highest observed churn rate, while filtering out very small groups.",
        )
        if not segment_table.empty:
            display = segment_table.head(20).copy()
            display["churn_rate"] = display["churn_rate"].map(lambda x: f"{x:.1%}")
            display["lift_vs_overall"] = display["lift_vs_overall"].map(lambda x: f"{x:.2f}x")
            display = display.rename(
                columns={
                    "feature_label": "Feature",
                    "segment": "Segment value",
                    "customers": "Customers",
                    "churn_rate": "Churn rate",
                    "lift_vs_overall": "Lift vs overall",
                }
            )
            st.dataframe(display[["Feature", "Segment value", "Customers", "Churn rate", "Lift vs overall"]], use_container_width=True, hide_index=True)

            chart_data = segment_table.head(12).copy()
            chart_data["label"] = chart_data["feature_label"] + " = " + chart_data["segment"].astype(str)
            fig = px.bar(
                chart_data.sort_values("churn_rate"),
                x="churn_rate",
                y="label",
                orientation="h",
                color="feature_label",
                labels={"churn_rate": "Churn rate", "label": "Customer segment", "feature_label": "Feature"},
                title="Highest-risk segments",
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(light_plotly(fig), use_container_width=True)
        else:
            st.info("No categorical or low-cardinality columns were suitable for segment analysis.")
        section_close()

    with row1_col2:
        section_open(
            "Commercial and lifecycle snapshot",
            "This compares average numeric values for customers who stayed versus customers who churned.",
        )
        if not snapshot.empty:
            snapshot_display = snapshot.copy()
            for col in ["Stayed", "Churned", "Difference"]:
                snapshot_display[col] = snapshot_display[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "Not available")
            st.dataframe(snapshot_display, use_container_width=True, hide_index=True)
        else:
            st.info("No suitable numeric fields were found for a stayed-versus-churned snapshot.")
        section_close()

        section_open(
            "Numeric churn gradients",
            "These bands show how churn changes across the distribution of the most informative numeric variables.",
        )
        if not numeric_bands.empty:
            feature_choice = st.selectbox(
                "Numeric driver",
                options=numeric_bands["feature"].drop_duplicates().tolist(),
                format_func=fmt_feature,
            )
            chart_data = numeric_bands[numeric_bands["feature"] == feature_choice].copy()
            fig = px.bar(
                chart_data,
                x="band",
                y="churn_rate",
                text="customers",
                labels={"band": fmt_feature(feature_choice) + " band", "churn_rate": "Churn rate"},
                title=f"Churn gradient across {fmt_feature(feature_choice)}",
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=380)
            st.plotly_chart(light_plotly(fig), use_container_width=True)
        else:
            st.info("No strong numeric gradient was detected from the available fields.")
        section_close()


def render_model_insights(bundle: Dict[str, Any]) -> None:
    comparison = bundle["comparison"].copy()
    comparison_display = comparison.copy()
    percent_cols = [
        "Accuracy @ 0.50",
        "Balanced accuracy @ 0.50",
        "Precision @ 0.50",
        "Recall @ 0.50",
        "F1 @ 0.50",
    ]
    for col in comparison_display.columns:
        if col in percent_cols:
            comparison_display[col] = comparison_display[col].map(lambda x: f"{x:.1%}")
        elif col != "Model":
            comparison_display[col] = comparison_display[col].map(lambda x: f"{x:.3f}")

    top_left, top_right = st.columns([1.05, 0.95], gap="large")
    with top_left:
        section_open(
            "Model comparison",
            "Two baselines are trained on the same split so the final dashboard can justify its model choice rather than assuming one model is best.",
        )
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)

        fig = px.bar(
            comparison,
            x="Model",
            y="ROC AUC",
            text="ROC AUC",
            title="Champion selection by ROC AUC",
            labels={"ROC AUC": "ROC AUC"},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=320)
        st.plotly_chart(light_plotly(fig), use_container_width=True)
        section_close()

    with top_right:
        section_open(
            "Top model drivers",
            "Permutation importance measures which raw features reduce model discrimination the most when their information is disrupted.",
        )
        importance = bundle["importance"].head(12).copy()
        if not importance.empty:
            fig = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="feature_label",
                orientation="h",
                labels={"importance": "AUC decrease after permutation", "feature_label": "Feature"},
                title="Most influential features",
            )
            fig.update_layout(height=480)
            st.plotly_chart(light_plotly(fig), use_container_width=True)
        else:
            st.info("Feature importance is not available for the current configuration.")
        section_close()

    lower_left, lower_right = st.columns([1.0, 1.0], gap="large")
    with lower_left:
        section_open(
            "Threshold operating view",
            "The recommended threshold is selected from the holdout sweep, but the full precision-recall trade-off is still visible.",
        )
        champion_name = bundle["champion_name"]
        threshold_df = bundle["model_results"][champion_name]["thresholds"].copy()
        long_df = threshold_df.melt(
            id_vars="threshold",
            value_vars=["precision", "recall", "f1"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.line(
            long_df,
            x="threshold",
            y="Score",
            color="Metric",
            markers=True,
            title="Precision, recall, and F1 across thresholds",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.add_vline(x=bundle["recommended_threshold"], line_dash="dash")
        fig.update_layout(height=360)
        st.plotly_chart(light_plotly(fig), use_container_width=True)

        threshold = st.slider(
            "Operating threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(bundle["recommended_threshold"]),
            step=0.01,
        )
        row = threshold_df.iloc[(threshold_df["threshold"] - threshold).abs().argsort()[:1]].iloc[0]
        metrics_table = pd.DataFrame(
            [
                {"Metric": "Precision", "Value": f"{row['precision']:.1%}"},
                {"Metric": "Recall", "Value": f"{row['recall']:.1%}"},
                {"Metric": "F1", "Value": f"{row['f1']:.1%}"},
                {"Metric": "Balanced accuracy", "Value": f"{row['balanced_accuracy']:.1%}"},
                {"Metric": "Customers flagged", "Value": f"{row['predicted_positive_share']:.1%}"},
            ]
        )
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        section_close()

    with lower_right:
        section_open(
            "Classification result at the selected threshold",
            "This converts the probability output into a final operating decision and shows the trade-off in customer-level counts.",
        )
        cm_df = confusion_at_threshold(
            bundle["y_test"].to_numpy(),
            bundle["champion_proba"],
            threshold,
        )
        cm_long = cm_df.reset_index().melt(id_vars="index", var_name="Prediction", value_name="Customers")
        cm_long = cm_long.rename(columns={"index": "Actual"})
        fig = px.density_heatmap(
            cm_long,
            x="Prediction",
            y="Actual",
            z="Customers",
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion matrix",
        )
        fig.update_layout(height=360)
        st.plotly_chart(light_plotly(fig), use_container_width=True)

        risk_bands = bundle["risk_bands"].copy()
        risk_bands["observed_churn_rate_label"] = risk_bands["observed_churn_rate"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            risk_bands.rename(
                columns={
                    "band": "Risk band",
                    "customers": "Customers",
                    "observed_churn_rate": "Observed churn rate",
                    "avg_model_score": "Average model score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        section_close()

    section_open(
        "Model narrative",
        "This is the concise evaluation summary a recruiter or hiring manager should be able to read in under a minute.",
    )
    model_points = build_model_insight_sentences(bundle)
    st.markdown("<ul class='insight-list'>" + "".join([f"<li>{item}</li>" for item in model_points]) + "</ul>", unsafe_allow_html=True)
    section_close()


def render_appendix(df: pd.DataFrame, meta: DataMeta, target_col: str, bundle: Dict[str, Any]) -> None:
    missing_share = (
        df.isna().mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing share"})
    )
    profile_rows = []
    for col in df.columns:
        profile_rows.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing share": df[col].isna().mean(),
                "Unique values": df[col].nunique(dropna=True),
            }
        )
    profile_df = pd.DataFrame(profile_rows).sort_values(["Missing share", "Unique values"], ascending=[False, False])

    section_open(
        "Technical appendix",
        "Raw previews, column checks, and model configuration are kept here so the main story remains insight-led.",
    )
    tech_metrics = f"""
    <div class="metric-grid">
        {metric_card("Source", meta.source_label, "Loaded into the current session")}
        {metric_card("Rows", f"{meta.rows:,}", "Before target split")}
        {metric_card("Columns", f"{meta.cols:,}", "Before feature pruning")}
        {metric_card("Target column", fmt_feature(target_col), "Positive class is interpreted as churn = 1")}
    </div>
    """
    st.markdown(tech_metrics, unsafe_allow_html=True)
    section_close()

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        section_open("Column health", "Missingness and cardinality remain available for technical review.")
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        section_close()

    with right:
        section_open("Missingness ranking", "Columns are sorted by missing share so technical issues stay visible without leading the story.")
        missing_display = missing_share.copy()
        missing_display["Missing share"] = missing_display["Missing share"].map(lambda x: f"{x:.1%}")
        st.dataframe(missing_display, use_container_width=True, hide_index=True)
        section_close()

    section_open("Raw preview", "Raw rows stay in the appendix instead of the main insight flow.")
    st.dataframe(df.head(250), use_container_width=True, hide_index=True)
    section_close()

    section_open("Model configuration", "This documents the final modeling setup used in the main dashboard.")
    config_df = pd.DataFrame(
        [
            {"Setting": "Champion model", "Value": bundle["champion_name"]},
            {"Setting": "Recommended threshold", "Value": f"{bundle['recommended_threshold']:.2f}"},
            {"Setting": "Train rows", "Value": f"{len(bundle['X_train']):,}"},
            {"Setting": "Test rows", "Value": f"{len(bundle['X_test']):,}"},
            {"Setting": "Scored features", "Value": f"{bundle['X'].shape[1]:,}"},
        ]
    )
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    section_close()


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.sidebar.title("Controls")
    st.sidebar.caption("Load data, select the target, and set the modeling split.")

    use_upload = st.sidebar.toggle("Use uploaded file", value=False)
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV only. Common delimiters are auto-detected.",
    )

    try:
        raw_df, meta = load_dataset(use_upload, uploaded)
        df = clean_dataframe(raw_df)
    except Exception as exc:
        st.error("The dataset could not be loaded.")
        st.write(str(exc))
        return

    if df.empty:
        st.error("The dataset is empty.")
        return

    candidates = detect_target_candidates(df)
    default_target = candidates[0] if candidates else df.columns[-1]

    st.sidebar.subheader("Model settings")
    target_col = st.sidebar.selectbox(
        "Target column",
        options=list(df.columns),
        index=list(df.columns).index(default_target),
    )
    test_size = st.sidebar.slider("Holdout share", min_value=0.15, max_value=0.40, value=0.25, step=0.01)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    try:
        bundle = fit_analysis_bundle(df, target_col, test_size, int(seed))
    except Exception as exc:
        st.error("The analysis bundle could not be built.")
        st.write(str(exc))
        return

    render_hero(meta)

    tabs = st.tabs(
        [
            "Executive Summary",
            "Data Insights",
            "Model Insights",
            "Scenario Studio",
            "Appendix",
        ]
    )

    with tabs[0]:
        render_executive_summary(df, bundle, target_col)

    with tabs[1]:
        render_data_insights(df, target_col)

    with tabs[2]:
        render_model_insights(bundle)

    with tabs[3]:
        section_open(
            "Scenario studio",
            "Use a real customer profile from the holdout sample as a starting point, then test how profile changes move churn risk.",
        )
        scenario_editor(bundle, threshold=bundle["recommended_threshold"])
        section_close()

    with tabs[4]:
        render_appendix(df, meta, target_col, bundle)


if __name__ == "__main__":
    main()
