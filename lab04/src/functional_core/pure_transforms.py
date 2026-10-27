import numpy as np
import pandas as pd
from typing import Dict, Tuple, NamedTuple

class LearnedPriors(NamedTuple):
    numeric_medians: Dict[str, float]
    categorical_modes: Dict[str, str]
    clipping_bounds: Dict[str, Tuple[float, float]]

def compute_train_statistics(df: pd.DataFrame, num_cols: list, cat_cols: list) -> LearnedPriors:
    meds = {}
    bounds = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        meds[c] = float(s.median())
        bounds[c] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

    modes = {}
    for c in cat_cols:
        clean = df[c].astype(str).str.strip().replace("?", np.nan)
        m = clean.dropna().mode()
        modes[c] = m.iloc[0] if not m.empty else "Missing"

    return LearnedPriors(numeric_medians=meds, categorical_modes=modes, clipping_bounds=bounds)

def apply_learned_priors(df: pd.DataFrame, priors: LearnedPriors, num_cols: list, cat_cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(priors.numeric_medians[c])
        lo, hi = priors.clipping_bounds[c]
        out[c] = out[c].clip(lower=lo, upper=hi)

    for c in cat_cols:
        out[c] = out[c].astype(str).str.strip().replace("?", np.nan).fillna(priors.categorical_modes[c])
    return out

def derive_analytical_features(df: pd.DataFrame) -> pd.DataFrame:
    augmented = df.copy()
    augmented["capital_net"] = augmented["capital-gain"] - augmented["capital-loss"]
    denom = np.maximum(augmented["age"] - 16.0, 1.0)
    augmented["hours_per_age"] = augmented["hours-per-week"] / denom
    return augmented
