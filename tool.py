# tool.py  — Streamlit predictor + optimizer with file-path sidebar & auto-discovery
# Run:  streamlit run tool.py

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import differential_evolution
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

# ==========================
# Defaults (can be overridden in the sidebar)
# ==========================
DEFAULT_EXCEL = "Training_data.xlsx"
DEFAULT_MODEL_PKL = "best_model.pkl"
DEFAULT_MODEL_META = "best_model_meta.json"

PCTL_LO, PCTL_HI = 2.5, 97.5        # robust default bounds (percentiles)
DEFAULT_TIME_CAP = 450.0            # cap time slider upper bound by default
SEED = 42                           # optimizer seed


# ==========================
# Small helpers
# ==========================
@dataclass
class FeatureNames:
    run: str
    temp: str
    rh: str


def safe_filename(s: str, maxlen: int = 120) -> str:
    s = str(s).replace("\\", "_").replace("/", "_")
    s = re.sub(r'[<>:\"|?*]+', "_", s)
    s = re.sub(r"\s{2,}", " ", s).strip().strip(".")
    return s[:maxlen]


def final_estimator_from_pipeline(model):
    """Return (prep, sel, est) if the model is a Pipeline/TransformedTargetRegressor."""
    reg = model
    if isinstance(model, TransformedTargetRegressor):
        reg = model.regressor_  # unwrap to Pipeline if present
    if isinstance(reg, Pipeline):
        steps = reg.named_steps
        prep = steps.get("scale", steps.get("prep", None))
        sel = steps.get("sel", None)
        est = steps.get("model", None)
        return prep, sel, est
    return None, None, reg


def transform_X_for_estimator(model, X_df):
    """Apply the same preprocessing/selection inside the training pipeline."""
    prep, sel, _ = final_estimator_from_pipeline(model)
    X = X_df.values
    if prep is not None and hasattr(prep, "transform"):
        X = prep.transform(X)
    if sel is not None and hasattr(sel, "transform"):
        X = sel.transform(X)
    return X


def hat_setup_from_train(model, X_train_np):
    """Precompute pieces to evaluate leverage h for new points with same standardization."""
    mu = np.nanmean(X_train_np, axis=0)
    sd = np.nanstd(X_train_np, axis=0)
    sd[sd < 1e-12] = 1e-12
    ones = np.ones((X_train_np.shape[0], 1))
    Z = np.hstack([ones, (X_train_np - mu) / sd])  # [n, p+1]
    ZTZ_inv = np.linalg.pinv(Z.T @ Z)
    return mu, sd, ZTZ_inv


def leverage_of_point_from_setup(model, row_df, mu, sd, ZTZ_inv):
    Xu = transform_X_for_estimator(model, row_df)  # [1, p]
    Z = np.hstack([np.ones((1, 1)), (Xu - mu) / sd])  # [1, p+1]
    return float(Z @ ZTZ_inv @ Z.T)


def _latest_match(patterns):
    """Return path to the newest file matching any of the glob patterns, or None."""
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    if not matches:
        return None
    matches = [Path(m) for m in matches if Path(m).is_file()]
    if not matches:
        return None
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def _resolve_or_find(user_path, name, fallback_patterns):
    """If user_path exists, use it. Else try to auto-find via patterns. Else stop the app."""
    if user_path and Path(user_path).is_file():
        return user_path
    auto = _latest_match(fallback_patterns)
    if auto:
        st.info(f"Auto-found **{name}** at: `{auto}`")
        return auto
    st.error(
        f"❌ Could not find **{name}**.\n\n"
        f"Tried path: `{user_path or ''}` and searched patterns: {fallback_patterns}\n\n"
        "Please enter a valid path in the sidebar or move the file next to this script."
    )
    st.stop()


# ==========================
# Cached loader (uses sidebar paths; tries auto-discovery)
# ==========================
@st.cache_resource(show_spinner=True)
def load_artifacts(excel_path, model_pkl, model_meta):
    # Resolve or auto-discover
    excel_path = _resolve_or_find(
        excel_path, "Training_data.xlsx",
        ["Training_data.xlsx", "**/Training_data.xlsx", "runs/**/Training_data.xlsx"]
    )
    model_meta = _resolve_or_find(
        model_meta, "best_model_meta.json",
        ["best_model_meta.json", "**/best_model_meta.json", "runs/**/best_model_meta.json"]
    )
    model_pkl = _resolve_or_find(
        model_pkl, "best_model.pkl",
        ["best_model.pkl", "**/best_model.pkl", "runs/**/best_model.pkl"]
    )

    # Load meta + model + data
    meta = json.loads(Path(model_meta).read_text(encoding="utf-8"))
    features = meta["features"]
    targets = meta["targets"]

    fns = FeatureNames(
        run=[c for c in features if "Run time" in c][0],
        temp=[c for c in features if "Temperature" in c][0],
        rh=[c for c in features if "RH" in c][0],
    )

    model = joblib.load(model_pkl)
    df = pd.read_excel(excel_path)[features + targets].dropna().reset_index(drop=True)

    # Robust bounds for sliders
    pbounds: Dict[str, Tuple[float, float]] = {}
    for col in [fns.run, fns.temp, fns.rh]:
        q = np.quantile(df[col], [PCTL_LO / 100.0, PCTL_HI / 100.0])
        pbounds[col] = (float(q[0]), float(q[1]))

    # AD (Williams) setup
    X_used = transform_X_for_estimator(model, df[features])
    p_used = X_used.shape[1]
    n_tr = X_used.shape[0]
    h_star = 0.039
    mu, sd, ZTZ_inv = hat_setup_from_train(model, X_used)

    return model, features, targets, fns, df, pbounds, (mu, sd, ZTZ_inv, h_star)


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Water Uptake & Daily Production — Predictor & Optimizer", layout="wide")
st.title("Water Uptake & Daily Production — Predictor & Optimizer")

# --- Sidebar: let the user set file paths ---
st.sidebar.header("📂 Model & Data Paths")
sb_excel = st.sidebar.text_input("Excel file", value=DEFAULT_EXCEL)
sb_pkl = st.sidebar.text_input("Model file (.pkl)", value=DEFAULT_MODEL_PKL)
sb_meta = st.sidebar.text_input("Meta file (.json)", value=DEFAULT_MODEL_META)

# Load everything (with caching & auto-discovery)
model, features, targets, fns, df, percentile_bounds, ad_pack = load_artifacts(sb_excel, sb_pkl, sb_meta)
(mu, sd, ZTZ_inv, h_star) = ad_pack

with st.expander("ℹ️ Loaded Artifacts", expanded=False):
    st.write(f"**Features (order):** {features}")
    st.write(f"**Targets:** {targets}")
    st.write(f"**Williams threshold (h∗):** {h_star:.6g}")

col_left, col_right = st.columns([1, 1])

# ------------------------ Predictor Pane ------------------------
with col_left:
    st.subheader("🔮 Predict")

    # Robust slider ranges
    rt_lo, rt_hi = percentile_bounds[fns.run]
    tp_lo, tp_hi = percentile_bounds[fns.temp]
    rh_lo, rh_hi = percentile_bounds[fns.rh]

    # Optional cap on time
    cap_time = st.checkbox(f"Cap '{fns.run}' at {DEFAULT_TIME_CAP:.0f} min", value=True)
    if cap_time:
        rt_hi = min(rt_hi, DEFAULT_TIME_CAP)

    # Sliders
    rt = st.slider(fns.run, float(rt_lo), float(rt_hi), float(np.clip((rt_lo + rt_hi) / 2, rt_lo, rt_hi)))
    tp = st.slider(fns.temp, float(tp_lo), float(tp_hi), float(np.clip((tp_lo + tp_hi) / 2, tp_lo, tp_hi)))
    rh = st.slider(fns.rh, float(rh_lo), float(rh_hi), float(np.clip((rh_lo + rh_hi) / 2, rh_lo, rh_hi)))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame({fns.run: [rt], fns.temp: [tp], fns.rh: [rh]})[features]
        y = np.asarray(model.predict(row)).ravel()
        preds = {targets[i]: float(y[i]) for i in range(len(targets))}
        h = leverage_of_point_from_setup(model, row, mu, sd, ZTZ_inv)

        st.success("Prediction complete.")
        st.write("**Inputs**")
        st.json({fns.run: rt, fns.temp: tp, fns.rh: rh})
        st.write("**Predictions**")
        st.json(preds)
        st.write(f"**Leverage h** = {h:.6g} &nbsp;&nbsp; **h∗** = {h_star:.6g} &nbsp;&nbsp; **In-domain:** {h <= h_star}")

# ------------------------ Optimizer Pane ------------------------
with col_right:
    st.subheader("🚀 Optimize")

    mode = st.radio("Objective", ["Maximize water uptake", "Maximize daily production", "Weighted (both)"], index=0)
    w1 = w2 = 0.5
    if mode == "Weighted (both)":
        w1 = st.number_input("Weight for water uptake (w1)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w2 = st.number_input("Weight for daily production (w2)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    enforce_ad = st.checkbox("Enforce Applicability Domain (leverage ≤ h∗)", value=False)

    st.markdown("**Search bounds** (defaults from robust percentiles; optionally narrow):")
    c1, c2, c3 = st.columns(3)
    with c1:
        rt_min = st.number_input(f"{fns.run} min", value=float(percentile_bounds[fns.run][0]))
        rt_max = st.number_input(f"{fns.run} max", value=float(min(percentile_bounds[fns.run][1], DEFAULT_TIME_CAP)))
    with c2:
        tp_min = st.number_input(f"{fns.temp} min", value=float(percentile_bounds[fns.temp][0]))
        tp_max = st.number_input(f"{fns.temp} max", value=float(percentile_bounds[fns.temp][1]))
    with c3:
        rh_min = st.number_input(f"{fns.rh} min", value=float(percentile_bounds[fns.rh][0]))
        rh_max = st.number_input(f"{fns.rh} max", value=float(percentile_bounds[fns.rh][1]))

    max_iter = st.slider("Max iterations (Differential Evolution)", 50, 500, 200, 10)

    if st.button("Run Optimization", type="primary"):
        # Assemble bounds in model feature order
        bmap: Dict[str, Tuple[float, float]] = {
            fns.run: (rt_min, rt_max),
            fns.temp: (tp_min, tp_max),
            fns.rh: (rh_min, rh_max),
        }
        lb = [bmap.get(col, (float(df[col].min()),))[0] for col in features]
        ub = [bmap.get(col, (None, float(df[col].max())))[1] for col in features]
        bounds = list(zip(lb, ub))

        def objective(x_vec):
            row = pd.DataFrame([x_vec], columns=features)
            y = np.asarray(model.predict(row)).ravel()
            if mode == "Maximize water uptake":
                score = y[0]
            elif mode == "Maximize daily production":
                score = y[1] if len(y) > 1 else y[0]
            else:
                y1 = y[0]
                y2 = y[1] if len(y) > 1 else y[0]
                score = w1 * y1 + w2 * y2

            if enforce_ad:
                h = leverage_of_point_from_setup(model, row, mu, sd, ZTZ_inv)
                if h > h_star:
                    return -(score) + 1e6 * (h - h_star) ** 2  # penalty if outside AD
            return -(score)

        res = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=int(max_iter),
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=SEED,
            polish=True,
            updating="deferred",
            workers=1,
        )

        sol = {features[i]: float(res.x[i]) for i in range(len(features))}
        ybest = np.asarray(model.predict(pd.DataFrame([res.x], columns=features))).ravel()
        preds = {targets[i]: float(ybest[i]) for i in range(len(targets))}
        h_sol = leverage_of_point_from_setup(model, pd.DataFrame([res.x], columns=features), mu, sd, ZTZ_inv)

        st.success("Optimization complete.")
        st.write(f"**Objective (max)** = {float(-res.fun):.6g}  |  **Success:** {res.success}")
        st.write("**Best Inputs**")
        st.json({fns.run: sol[fns.run], fns.temp: sol[fns.temp], fns.rh: sol[fns.rh]})
        st.write("**Predictions at Optimum**")
        st.json(preds)
        st.write(f"**Leverage h** = {h_sol:.6g}  |  **h∗** = {h_star:.6g}  |  **In-domain:** {h_sol <= h_star}")

        # Offer JSON download
        out_dict = {
            "success": res.success,
            "objective_max": float(-res.fun),
            "inputs": {fns.run: sol[fns.run], fns.temp: sol[fns.temp], fns.rh: sol[fns.rh]},
            "predictions": preds,
            "leverage_h": h_sol,
            "h_star": h_star,
            "in_domain": bool(h_sol <= h_star),
            "bounds": bmap,
            "mode": mode,
            "weights": {"w1": w1, "w2": w2} if mode == "Weighted (both)" else None,
        }
        st.download_button(
            "Download result (.json)",
            data=json.dumps(out_dict, indent=2),
            file_name=f"opt_result_{safe_filename(mode)}.json",
            mime="application/json",
        )

# ------------------------ Notes ------------------------
with st.expander("Notes & Tips", expanded=False):
    st.markdown(
        f"""
- Defaults use **inner {PCTL_LO}–{PCTL_HI} percentiles** from your data for robust slider bounds.
- Toggle **Applicability Domain** to keep solutions within the Williams domain (leverage ≤ h∗ = {h_star:.4g}).
- Optimizer: **Differential Evolution** (global). Increase iterations for tougher landscapes.
- Targets are taken in this order from your meta file:  
  **[0]** = *Water uptake (L/kg)*, **[1]** = *Daily water production rate (L/kg.day)* (adjust naming if yours differs).
        """
    )
