import pandas as pd
import numpy as np
import sqlite3
import pandas_ta as ta
from typing import Dict, List
from scipy.signal import argrelextrema

# ============================================================
#  USER‑TUNABLE HYPER‑PARAMETERS
# ============================================================

# Indicator window lengths expressed in *minutes* (key == timeframe)
LENGTHS: Dict[int, Dict[str, int]] = {
    1:  {"sma": 9,  "rsi": 12, "macd_fast": 6,  "macd_slow": 12,
         "macd_sig": 9, "bb": 9,  "long_ma": 27},
    5:  {"sma": 45, "rsi": 60, "macd_fast": 30, "macd_slow": 60,
         "macd_sig": 45, "bb": 45, "long_ma": 24},
    15: {"sma": 135, "rsi": 180, "macd_fast": 90, "macd_slow": 180,
         "macd_sig": 135, "bb": 135, "long_ma": 21},
    30: {"sma": 270, "rsi": 360, "macd_fast": 180, "macd_slow": 360,
         "macd_sig": 270, "bb": 270, "long_ma": 18},
    60: {"sma": 540, "rsi": 720, "macd_fast": 360, "macd_slow": 720,
         "macd_sig": 540, "bb": 540, "long_ma": 21},
}

#  Range/consolidation threshold as % of price
RANGE_PCT: Dict[int, float] = {1: 0.25, 5: 0.20, 15: 0.15, 30: 0.12, 60: 0.10}

#  Minimum distance (#bars) between consecutive identical targets
MIN_DIST: Dict[int, int] = {1: 9, 5: 9, 15: 6, 30: 5, 60: 4}

#  NEW – Maximum distance (#bars) we will tolerate between alternating
#        peaks ↔ valleys before we force an additional switch.
MAX_SEG_DIST: Dict[int, int] = {1: 60, 5: 40, 15: 30, 30: 25, 60: 20}

#  Distance used *inside* a consolidation segment to increase the
#  frequency of peaks / valleys.
RANGE_MIN_DIST: Dict[int, int] = {1: 4, 5: 4, 15: 3, 30: 3, 60: 2}

#  ADX threshold below which we call the market "range / consolidation"
ADX_THRESH: float = 18.0

# ============================================================
#  GLOBAL SETTINGS
# ============================================================
SEQ_LEN: int  = 128          # full sequence length fed to the NN
LOOKBACK: int = SEQ_LEN // 3 # universal window for *all* indicators
SLOPE_FEATURES: int = SEQ_LEN // 2   # 256 High-slopes + 256 Low-slopes

# ============================================================
#  FEATURE ENGINEERING
# ============================================================

def add_new_features(
    df: pd.DataFrame,
    close_col: str,
    high_col: str,
    low_col: str,
    open_col: str,
    tf: int,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """
    Causal technical indicators using the common LOOKBACK horizon.
    """
    if df[close_col].isna().all() or len(df) < max(50, lookback):
        return df

    # ---- indicator windows (now *identical* for every timeframe) ----
    p = {
        "sma":        lookback // 3,
        "rsi":        lookback // 2,
        "macd_fast":  max(2, lookback // 8),
        "macd_slow":  lookback // 4,
        "macd_sig":   max(2, lookback // 6),
        "bb":         lookback // 3,
        "long_ma":    lookback,
    }

    # ------------------------------------------------------------------
    # Classical / momentum indicators
    # ------------------------------------------------------------------
    df["New_SMA"]  = ta.sma(df[close_col], length=p["sma"])
    df["New_RSI"]  = ta.rsi(df[close_col], length=p["rsi"])

    macd = ta.macd(df[close_col],
                   fast=p["macd_fast"],
                   slow=p["macd_slow"],
                   signal=p["macd_sig"])
    df["New_MACD"]        = macd[f"MACD_{p['macd_fast']}_{p['macd_slow']}_{p['macd_sig']}"]
    df["New_MACD_Signal"] = macd[f"MACDs_{p['macd_fast']}_{p['macd_slow']}_{p['macd_sig']}"]

    bbands = ta.bbands(df[close_col], length=p["bb"], std=2)
    df["New_BB_Upper"]  = bbands[f"BBU_{p['bb']}_2.0"]
    df["New_BB_Lower"]  = bbands[f"BBL_{p['bb']}_2.0"]
    df["New_BB_Middle"] = bbands[f"BBM_{p['bb']}_2.0"]
    df["New_BB_Width"]  = (df["New_BB_Upper"] - df["New_BB_Lower"]) / df["New_BB_Middle"]

    #  Long moving average & slope
    df["New_MA_Long"]          = ta.sma(df[close_col], length=p["long_ma"])
    df["New_MA_Long_Slope"]    = df["New_MA_Long"].diff()
    df["New_MA_Long_Slope_Sm"] = df["New_MA_Long_Slope"].rolling(p["long_ma"]).mean()

    #  Faster / slower EMA spread (still keep tf-relative flavour)
    df["New_EMA_Fast"]  = ta.ema(df[close_col], length=max(2, tf * 2))
    df["New_EMA_Slow"]  = ta.ema(df[close_col], length=max(4, tf * 4))
    df["New_EMA_Trend"] = df["New_EMA_Fast"] - df["New_EMA_Slow"]

    #  Volatility & trend strength
    df["New_ATR"]          = ta.atr(df[high_col], df[low_col], df[close_col], length=lookback)
    df["New_ATR_Percent"]  = df["New_ATR"] / df[close_col] * 100
    df["New_NATR"]         = ta.natr(df[high_col], df[low_col], df[close_col], length=lookback)

    df["New_Trend_Strength"] = (
        (df[close_col] - df[close_col].rolling(window=lookback).mean()) /
        df[close_col].rolling(window=lookback).std()
    )

    #  Oscillators
    stoch = ta.stoch(df[high_col], df[low_col], df[close_col],
                     k=lookback, d=max(3, lookback // 3))
    if not stoch.empty:
        df["New_Stoch_K"] = stoch.iloc[:, 0]
        df["New_Stoch_D"] = stoch.iloc[:, 1]

    df["New_ROC"]  = ta.roc(df[close_col], length=lookback)
    df["New_MOM"]  = ta.mom(df[close_col], length=lookback)
    df["New_CCI"]  = ta.cci(df[high_col], df[low_col], df[close_col], length=lookback)
    df["New_WILLR"] = ta.willr(df[high_col], df[low_col], df[close_col], length=lookback)

    #  Directional movement / ADX
    adx_df = ta.adx(df[high_col], df[low_col], df[close_col], length=lookback)
    df["New_ADX"]    = adx_df[f"ADX_{lookback}"]
    df["New_DI_Pos"] = adx_df[f"DMP_{lookback}"]
    df["New_DI_Neg"] = adx_df[f"DMN_{lookback}"]

    #  Price / volatility regime & distances
    df["New_HL_Range"] = (df[high_col] - df[low_col]) / df[close_col] * 100
    df["New_CO_Range"] = (df[close_col] - df[open_col]) / df[open_col] * 100

    #  Volatility regime (short vs long window)
    vol_short = lookback // 2
    df["New_Vol_Regime"] = (
        df[close_col].rolling(window=vol_short).std() /
        df[close_col].rolling(window=lookback).std()
    )

    df["New_Price_Distance"] = (
        (df[close_col] - df[close_col].rolling(window=vol_short).mean()) /
        df[close_col].rolling(window=vol_short).std()
    )

    #  Normalised support / resistance position
    df["New_Support_Level"]    = df[low_col].rolling(window=vol_short).min()
    df["New_Resistance_Level"] = df[high_col].rolling(window=vol_short).max()
    df["New_Price_Position"] = (
        (df[close_col] - df["New_Support_Level"]) /
        (df["New_Resistance_Level"] - df["New_Support_Level"])
    )

    #  Momentum of price & RSI
    df["New_Price_Mom"] = df[close_col].diff(lookback)
    df["New_RSI_Mom"]   = df["New_RSI"].diff(lookback)

    # ------------------------------------------------------------------
    #  Clean-up
    # ------------------------------------------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ------------------------------------------------------------
#  NEW 512-bar slope features (256 Highs, 256 Lows)
# ------------------------------------------------------------
def add_slope_features(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    seq_len: int = SEQ_LEN,
) -> pd.DataFrame:
    """
    Compute the causal High/Low slope block (seq_len/2 * 2 columns)
    and write / overwrite it in one shot so               │
        – no duplicate column names occur,               │
        – the DataFrame is not fragmented.               │
    """
    half = seq_len // 2

    # -------- build every slope series in memory ---------
    slopes = {
        # High-based slopes
        **{
            f"New_SlopeH_{lag}": (df[high_col] - df[high_col].shift(lag)) / lag
            for lag in range(1, half + 1)
        },
        # Low-based slopes
        **{
            f"New_SlopeL_{lag}": (df[low_col] - df[low_col].shift(lag)) / lag
            for lag in range(1, half + 1)
        },
    }

    slopes_df = pd.DataFrame(slopes, index=df.index)

    # ----- single vectorised assignment (overwrites if present) -----
    df[slopes_df.columns] = slopes_df

    return df

# ============================================================
#  TARGET‑BUILDING
# ============================================================

def _nearest_extreme(hi, lo, start, end, want_peak=True):
    """
    Return the index of the highest HIGH (peak) or lowest LOW (valley)
    between start and end (inclusive).  If hi/lo arrays are large this
    is still O(window) because the window is tiny (<= 40 bars).
    """
    if want_peak:
        rel = np.argmax(hi[start:end+1])
    else:
        rel = np.argmin(lo[start:end+1])
    return int(start + rel)

def detect_peaks_valleys(
    group_data: pd.DataFrame,
    tf: int,
    close_col: str,
    high_col: str,
    low_col: str,
    guidance: np.ndarray = None
) -> pd.Series:
    """
    Generate two‑class targets:
        1000  -> Trend‑Up    (valley → peak)
        5000  -> Trend‑Down  (peak → valley)
    The algorithm:
        1.  Find local maxima/minima on a long MA using a forward window.
        2.  Build alternating peak/valley markers with a minimal spacing.
        3.  Convert every segment between markers into Trend‑Up/Down.
        4.  If the segment looks like consolidation
              (small amplitude **or** low ADX) we *split it*
              into short alternating runs instead of adding a
              dedicated "range" class.
    """

    target = pd.Series(0, index=group_data.index)

    if "New_MA_Long" not in group_data.columns:
        return target  # nothing to do

    ma_long  = group_data["New_MA_Long"].values

    # ---------------------------------------------------------
    #  High / Low / Close arrays are needed for early refining
    # ---------------------------------------------------------
    hi_vals    = group_data[high_col].values
    lo_vals    = group_data[low_col].values
    close_vals = group_data[close_col].values

    rsi_data = group_data.get("New_RSI",   pd.Series(index=group_data.index, dtype="float64")).values
    bb_upper = group_data.get("New_BB_Upper", pd.Series(index=group_data.index, dtype="float64")).values
    bb_lower = group_data.get("New_BB_Lower", pd.Series(index=group_data.index, dtype="float64")).values
    macd_val = group_data.get("New_MACD",  pd.Series(index=group_data.index, dtype="float64")).values
    macd_sig = group_data.get("New_MACD_Signal", pd.Series(index=group_data.index, dtype="float64")).values
    adx_data = group_data.get("New_ADX",   pd.Series(index=group_data.index, dtype="float64")).values

    n_bars = len(ma_long)

    # Work with *positional* indices (0…n_bars‑1) to avoid mismatches
    # when the DataFrame keeps its original, gapped index.
    target_arr = np.zeros(n_bars, dtype=int)

    #  Forward‑window size scales with tf
    fw = {1: 20, 5: 15, 15: 12, 30: 9, 60: 6}.get(tf, 5)

    #  Rolling std for tolerance
    vol_window   = max(12, fw)
    rolling_std  = pd.Series(ma_long).rolling(vol_window).std().bfill().values
    rolling_std[np.isnan(rolling_std)] = 0

    raw_signals: List[tuple] = []

    # ========================================================
    #  Step‑1: obtain preliminary peak/valley points
    # ========================================================
    for i in range(n_bars - fw):
        peak_score, valley_score = 0.0, 0.0

        fwd_slice     = ma_long[i : i + fw + 1]
        ma_local_max  = fwd_slice.max()
        ma_local_min  = fwd_slice.min()
        local_range   = ma_local_max - ma_local_min
        local_mean    = np.mean(fwd_slice) if np.mean(fwd_slice) != 0 else 1e-12
        buffer_factor = np.clip(local_range / local_mean, 0.2, 0.6)
        tol           = buffer_factor * rolling_std[i]

        near_max = ma_long[i] >= ma_local_max - tol
        near_min = ma_long[i] <= ma_local_min + tol
        if near_max:
            peak_score += 1.0
        if near_min:
            valley_score += 1.0

        #  Add soft votes
        if rsi_data[i] > 65:
            peak_score += 0.35
        elif rsi_data[i] < 35:
            valley_score += 0.35

        if bb_upper[i] and ma_long[i] >= bb_upper[i]:
            peak_score += 0.25
        if bb_lower[i] and ma_long[i] <= bb_lower[i]:
            valley_score += 0.25

        if macd_val[i] > macd_sig[i]:
            peak_score += 0.25
        else:
            valley_score += 0.25

        # ----------------------------------------------------
        #  Decide if this bar is a candidate signal
        # ----------------------------------------------------
        if max(peak_score, valley_score) >= 1.0:
            if peak_score > valley_score:
                raw_signals.append((i, "peak"))
            elif valley_score > peak_score:
                raw_signals.append((i, "valley"))
            else:  # tie‑break
                slope = ma_long[i] - ma_long[i - 1] if i > 0 else 0
                raw_signals.append((i, "peak" if slope < 0 else "valley"))

    #  Sort + deduplicate
    raw_signals.sort(key=lambda x: x[0])

    # ========================================================
    #  Step‑2: enforce minimal distance & alternate types
    # ========================================================
    min_dist = MIN_DIST.get(tf, 4)
    final_sig: List[tuple] = []
    last_idx, last_type = -1_000_000, None
    for sig_idx, sig_type in raw_signals:
        if (sig_idx - last_idx) < min_dist:
            continue
        if not final_sig or sig_type != last_type:
            final_sig.append((sig_idx, sig_type))
            last_idx, last_type = sig_idx, sig_type

    # ========================================================
    #  NEW Step‑2b:  refine every signal to the *true* high/low
    #  --------------------------------------------------------
    #  – For a peak → move to the bar that owns the highest HIGH
    #  – For a valley→ move to the bar that owns the lowest  LOW
    #  The search span is symmetrical:  ±refine_w bars
    # ========================================================
    if final_sig:
        refine_w = {1: 30, 5: 25, 15: 20, 30: 15, 60: 10}.get(tf, 10)
        refined_sig: List[tuple] = []
        for pos, typ in final_sig:
            left  = max(0, pos - refine_w)
            right = min(n_bars - 1, pos + refine_w)
            if typ == "peak":
                # highest HIGH inside [left, right]
                new_pos = int(np.argmax(hi_vals[left : right + 1]) + left)
            else:  # valley
                # lowest  LOW  inside [left, right]
                new_pos = int(np.argmin(lo_vals[left : right + 1]) + left)
            refined_sig.append((new_pos, typ))

        #  De‑duplicate (two signals could collapse onto the same bar)
        refined_sig.sort(key=lambda x: x[0])
        final_sig = []
        for p, t in refined_sig:
            if not final_sig or p != final_sig[-1][0]:
                # keep alternation flag; if same type repeats, skip
                if not final_sig or t != final_sig[-1][1]:
                    final_sig.append((p, t))

    if not final_sig:
        return target  # no signal found

    rng_pct_thr = RANGE_PCT.get(tf, 0.12)

    # ========================================================
    #  Step‑3: segment‑wise labelling (2 classes; ranges split)
    # ========================================================
    for seg_i, (cur_pos, cur_type) in enumerate(final_sig):
        nxt_pos = (
            final_sig[seg_i + 1][0] if seg_i < len(final_sig) - 1 else n_bars
        )

        slice_hi  = hi_vals[cur_pos:nxt_pos]
        slice_lo  = lo_vals[cur_pos:nxt_pos]
        slice_adx = adx_data[cur_pos:nxt_pos]

        amplitude_pct = (
            (np.nanmax(slice_hi) - np.nanmin(slice_lo)) /
            close_vals[cur_pos] * 100
        )
        mean_adx = np.nanmean(slice_adx)

        is_range = (amplitude_pct < rng_pct_thr) or (mean_adx < ADX_THRESH)

        if is_range:
            # ------------------------------
            # Consolidation → higher freq.
            # ------------------------------
            rng_step = RANGE_MIN_DIST.get(tf, max(1, MIN_DIST.get(tf, 2)//2))
            toggle_label = 5000 if cur_type == "peak" else 1000
            start = cur_pos
            while start < nxt_pos:
                end = min(start + rng_step, nxt_pos)
                target_arr[start:end] = toggle_label
                toggle_label = 1000 if toggle_label == 5000 else 5000
                start = end
        else:
            # Normal trend-segment
            label = 5000 if cur_type == "peak" else 1000
            max_seg = MAX_SEG_DIST.get(tf, 40)
            seg_len = nxt_pos - cur_pos
            
            if seg_len > max_seg:
                # Split a very long trend segment into shorter alternating blocks
                # with each block placed at a local extreme
                step = max_seg
                toggle_label = 5000 if cur_type == "peak" else 1000
                start = cur_pos
                
                while start < nxt_pos:
                    end = min(start + step, nxt_pos)
                    
                    # Find local extreme within this window
                    if toggle_label == 5000:  # Looking for peak
                        # Find index of highest high in this segment
                        relative_idx = np.nanargmax(hi_vals[start:end+1])
                    else:  # Looking for valley (1000)
                        # Find index of lowest low in this segment
                        relative_idx = np.nanargmin(lo_vals[start:end+1])
                    
                    # Convert relative index to absolute
                    cut_point = start + max(1, int(relative_idx))
                    cut_point = min(cut_point, nxt_pos)
                    
                    # Label this segment
                    target_arr[start:cut_point] = toggle_label
                    
                    # Switch to opposite label for next segment
                    toggle_label = 1000 if toggle_label == 5000 else 5000
                    start = cut_point
            else:
                # Segment is short enough, just use the original label
                target_arr[cur_pos:nxt_pos] = label

    # ----------------------------------------------------------------
    #  Build the return Series and convert 0 → NaN (unlabelled)
    # ----------------------------------------------------------------
    target_series = pd.Series(target_arr, index=group_data.index, dtype="float64")
    target_series.replace(0, np.nan, inplace=True)

    return target_series

# ============================================================
#  DATA‑PROCESSING PIPELINE
# ============================================================

def process_table(df: pd.DataFrame, timeframes: List[int]) -> pd.DataFrame:
    """
    Build features & two‑class targets for every timeframe present in *df*.
    """
    #  Remove any previous Target4
    if "Target4" in df.columns:
        df.drop(columns="Target4", inplace=True)

    # ---------------------------------------------------------
    #  Align every timeframe to the *first* 60‑minute bar
    # ---------------------------------------------------------
    first_60_idx = df.loc[df["Group"] == 60].index.min()
    if first_60_idx is not None:
        df = df.loc[first_60_idx:].copy()

    # ---------------------------------------------------------
    #  Create an *absolute* time‑key so every timeframe that
    #  represents the same bar (Month, Day, Hour, Minute) can be
    #  aggregated together later on.
    # ---------------------------------------------------------
    df["TimeKey"] = (
        df["Month"].astype(int)  * 10_000_000
      + df["Day"].astype(int)    *    100_000
      + df["Hour"].astype(int)   *      1_000
      + df["Minute"].astype(int)
    )

    #  We will also keep a dict of "guidance labels" coming from the
    #  next‑higher timeframe
    higher_guidance: Dict[int, np.ndarray] = {}

    #  Feature engineering for every timeframe
    for tf in sorted(timeframes, reverse=True):        # 60 → 30 → 15 …
        col_prefix   = f"{tf}m_"
        req = [f"{col_prefix}{fld}_0" for fld in ("Close", "High", "Low", "Open")]

        if not all(c in df.columns for c in req):
            continue  # dataframe does not include this timeframe

        close_col, high_col, low_col, open_col = req
        mask = df["Group"] == tf
        if mask.sum() == 0:
            continue

        grp_df = df.loc[mask].copy()

        #  ----------------------------------------------------
        #  Feature engineering (legacy indicators)
        #  ----------------------------------------------------
        grp_df = add_new_features(
            grp_df, close_col, high_col, low_col, open_col,
            tf, lookback=LOOKBACK
        )

        #  ----------------------------------------------------
        #  NEW slope-based features
        #  ----------------------------------------------------
        grp_df = add_slope_features(grp_df, high_col, low_col, seq_len=SEQ_LEN)
        
        #  Drop rows with missing critical new features
        critical_feats = ["New_MA_Long", "New_RSI", "New_ADX"]
        grp_df.dropna(subset=[c for c in critical_feats if c in grp_df.columns],
                      inplace=True)
        if grp_df.empty:
            continue

        #  ----------------------------------------------------
        #  Target generation WITH optional higher‑TF guidance
        #  ----------------------------------------------------
        guidance = None
        if (tf + 1) in higher_guidance:        # e.g. 30's guide available when tf==15
            guidance = higher_guidance[tf + 1]

        grp_df["Target_tmp"] = detect_peaks_valleys(
            grp_df,
            tf,
            close_col,
            high_col,
            low_col,
            guidance
        )

        #  Save this label stream so the immediately lower timeframe can see it
        higher_guidance[tf] = grp_df["Target_tmp"].values

        #  Write back to main df
        df.drop(index=df.loc[mask].index, inplace=True)
        df = pd.concat([df, grp_df], axis=0)

    # ---------------------------------------------------------
    #  FUSION: majority vote across all timeframes on TimeKey
    # ---------------------------------------------------------
    pivot = (
        df[["TimeKey", "Group", "Target_tmp"]]
          .dropna(subset=["Target_tmp"])
          .pivot_table(index="TimeKey", columns="Group", values="Target_tmp", aggfunc="first")
    )

    def _vote(row):
        vals = row.dropna().astype(int)
        if vals.empty:
            return np.nan
        # majority
        top = vals.value_counts()
        if top.iloc[0] > 1:
            return top.idxmax()
        # tie → choose highest timeframe available
        return vals.loc[sorted(vals.index, reverse=True)[0]]

    fused = pivot.apply(_vote, axis=1)
    df["Target4"] = df["TimeKey"].map(fused).astype("float64")

    # One last forward‑fill for intra‑segment rows
    df.sort_index(inplace=True)
    df["Target4"] = df["Target4"].ffill()

    # ---------------------------------------------------------
    #  Remove the temporary per‑TF label column
    # ---------------------------------------------------------
    if "Target_tmp" in df.columns:
        df.drop(columns="Target_tmp", inplace=True)

    #  ----------------------------------------------------------------
    #  Remove rows where *any* New_* feature is NaN OR Target4 is NaN
    #  ----------------------------------------------------------------
    new_cols = [c for c in df.columns if c.startswith("New_")]
    df.dropna(subset=new_cols + ["Target4"], inplace=True)

    #  ----------------------------------------------------------------
    #  Post‑processing: merge short runs (now for 1000/5000)
    #  ----------------------------------------------------------------
    for tf in df["Group"].unique():
        mask = df["Group"] == tf
        arr  = df.loc[mask, "Target4"].values
        valid_mask = np.isin(arr, [1000, 5000])

        if not valid_mask.any():
            continue

        runs, N = [], len(arr)
        i = 0
        while i < N:
            if valid_mask[i]:
                v = arr[i]
                s = i
                while i + 1 < N and arr[i + 1] == v:
                    i += 1
                e = i
                runs.append((s, e, v))
            i += 1

        min_dist = MIN_DIST.get(tf, 4)
        merged = []
        for run in runs:
            s, e, v = run
            run_len = e - s + 1
            if run_len < min_dist:
                if merged and merged[-1][2] == v:
                    merged[-1] = (merged[-1][0], e, v)
                elif (idx := runs.index(run)) < len(runs) - 1 and runs[idx + 1][2] == v:
                    #  let next run absorb this one
                    continue
                elif merged:
                    #  merge into previous regardless of value
                    ps, pe, pv = merged[-1]
                    merged[-1] = (ps, e, pv)
                else:
                    merged.append((s, e, v))
            else:
                merged.append(run)

        #  Apply merged runs back to the array
        new_arr = np.zeros_like(arr)
        for s, e, v in merged:
            new_arr[s : e + 1] = v
        df.loc[mask, "Target4"] = new_arr

    #  ----------------------------------------------------------------
    #  Column ordering:  Original ‑> New_* ‑> other target columns
    #  ----------------------------------------------------------------
    new_cols = [c for c in df.columns if c.startswith("New_")]
    other_tg = [c for c in df.columns if c.startswith("Target") and c != "Target4"]
    #  TimeKey is internal only – keep it out of the final table
    base_cols = [
        c for c in df.columns
        if c not in new_cols + other_tg + ["Target4", "TimeKey", "Target_tmp"]
    ]
    df = df[base_cols + new_cols + other_tg + ["Target4"]]

    return df

# ============================================================
#  DATABASE IO
# ============================================================

def process_database(src_db: str, dst_db: str) -> None:
    """
    Iterate over every table in *src_db*, enrich with features & targets
    and persist into *dst_db*.
    """
    src_conn = sqlite3.connect(src_db)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", src_conn
    )["name"].tolist()

    timeframes = [1, 5, 15, 30, 60]

    for tbl in tables:
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", src_conn)
        df = process_table(df, timeframes)

        dst_conn = sqlite3.connect(dst_db)
        df.to_sql(tbl, dst_conn, if_exists="replace", index=False)
        dst_conn.close()

    src_conn.close()
    print(f"✓ Processing complete – results saved to {dst_db}")

# ============================================================
#  EXECUTION ENTRY‑POINT
# ============================================================

if __name__ == "__main__":
    #  Example paths – adjust as needed
    SRC_DB = r"C:\Users\crgon\OneDrive\Desktop\trading\data\backup\processed_multi_stacked_XAUUSD_Train_5_15_30_60.db"
    DST_DB = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_new_method.db"

    process_database(SRC_DB, DST_DB)