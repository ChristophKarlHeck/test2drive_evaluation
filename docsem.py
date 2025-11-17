import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ============================================================
# 1) LOAD + MERGE DATA
# ============================================================

users = pd.read_csv("docsem/users.csv")
trials = pd.read_csv("docsem/reactionTestTrials.csv")
sessions = pd.read_csv("docsem/reactionTestSessions.csv")

merged = trials.merge(
    sessions,
    left_on="sessionId",
    right_on="id",
    suffixes=("_trial", "_session")
)

merged["timestamp_dt"] = pd.to_datetime(merged["timestamp"], unit="ms")

# ============================================================
# 2) REMOVE NON-DRINKERS
# ============================================================

bac_per_user = merged.groupby("userId")["breathAlcoholValue"].max()
drinkers = bac_per_user[bac_per_user > 0].index
merged = merged[merged["userId"].isin(drinkers)]

# ============================================================
# 3) REMOVE FIRST TWO SESSIONS
# ============================================================

first_two_sessions = (
    merged.groupby("userId")["sessionId"]
    .unique().map(lambda x: x[:2])
)
sessions_to_remove = set(sid for arr in first_two_sessions for sid in arr)
merged = merged[~merged["sessionId"].isin(sessions_to_remove)]

# ============================================================
# 4) PIPELINE A — include misses as max error
# ============================================================

merged_A = merged.copy()

merged_A["taps_missed_flag"] = (merged_A["reactionTimeMs"] < 0).astype(int)

merged_A["reactionTimeMs_fixed"] = merged_A["reactionTimeMs"].copy()
merged_A.loc[merged_A["reactionTimeMs"] < 0, "reactionTimeMs_fixed"] = 1.0

merged_A["distance_fixed"] = merged_A["distanceFromTargetNormalized"]
merged_A.loc[merged_A["taps_missed_flag"] == 1, "distance_fixed"] = 1.0

# ============================================================
# 5) PIPELINE B — exclude misses
# ============================================================

# ---- Step 1: compute taps missed BEFORE filtering ----
taps_missed_count = (
    merged.groupby(["userId","sessionId"])["reactionTimeMs"]
    .apply(lambda s: (s < 0).sum())
)

# ---- Step 2: remove missed trials for B ----
merged_B = merged[merged["reactionTimeMs"] >= 0].copy()

# Keep clean RT and distances
merged_B["reactionTimeMs_fixed"] = merged_B["reactionTimeMs"]
merged_B["distance_fixed"] = merged_B["distanceFromTargetNormalized"]

# ---- Step 3: taps missed will be re-added after aggregation ----
merged_B["taps_missed_flag"] = 0   # 0 for all remaining trials

# ============================================================
# 6) SESSION AGGREGATION
# ============================================================

def compute_session_means(df):
    return (
        df.groupby(["userId","sessionId","breathAlcoholValue"])
        .agg(
            meanDistance=("distance_fixed","mean"),
            meanReactionTimeMs=("reactionTimeMs_fixed","mean"),
            dist_ge_01=("distance_fixed", lambda x: (x>=0.1).sum()),
            taps_missed=("taps_missed_flag","sum"),
        )
        .reset_index()
    )

session_A = compute_session_means(merged_A)

# Pipeline B (means exclude misses)
session_B = compute_session_means(merged_B)

# ---- Add correct taps_missed back to pipeline B ----
session_B["taps_missed"] = session_B.apply(
    lambda row: taps_missed_count.loc[(row["userId"], row["sessionId"])],
    axis=1
)

# ============================================================
# 7) HELPER — plot 4 subplots
# ============================================================

metrics = [
    ("meanDistance",        "Mean Distance From Target"),
    ("meanReactionTimeMs",  "Mean Reaction Time (ms)"),
    ("dist_ge_01",          "Count distance ≥ 0.1"),
    ("taps_missed",         "Missed Taps"),
]

def plot_four_panel(df, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, ylabel) in zip(axes, metrics):

        x = df["breathAlcoholValue"]
        y = df[col]

        mask = ~y.isna()
        x = x[mask]
        y = y[mask]

        # scatter
        ax.scatter(x, y, alpha=0.6, s=25)

        # regression
        if len(np.unique(x)) > 1:
            a, b = np.polyfit(x, y, 1)
            order = np.argsort(x)
            ax.plot(x.iloc[order], a*x.iloc[order] + b,
                    color="red", linewidth=2)

        ax.set_title(ylabel)
        ax.set_xlabel("Breath Alcohol Value (BAC)")
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# ============================================================
# 8) PRODUCE BOTH FIGURES
# ============================================================

plot_four_panel(session_A, "INCLUDE MISSES AS MAX ERROR")
plot_four_panel(session_B, "EXCLUDE MISSES")

import statsmodels.formula.api as smf

# ============================================================
# RUN MIXED MODELS (LMMs) FOR BOTH PIPELINES
# ============================================================

def mixed_model_plot(df, col, ylabel, title_suffix, ax):
    """
    Fit mixed model y ~ BAC + (1 | user),
    and plot:
    - scatter
    - global regression line
    - subject-specific BLUP lines
    """

    data = df.dropna(subset=[col, "breathAlcoholValue"]).copy()

    # Fit mixed effects model
    formula = f"{col} ~ breathAlcoholValue"
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["userId"],
        re_formula="1"
    ).fit(reml=True)

    # ---- Print effects ----
    print("\n===================================================")
    print(f" Mixed Model for {ylabel} — {title_suffix}")
    print("===================================================")
    print(model.summary())

    fe_intercept = model.fe_params["Intercept"]
    fe_slope = model.fe_params["breathAlcoholValue"]
    random_effects = model.random_effects

    # =====================================================
    # PLOTTING
    # =====================================================

    # Scatter points
    ax.scatter(
        data["breathAlcoholValue"], data[col],
        alpha=0.5, s=25, color="black"
    )

    # ---- Subject-specific lines (BLUP) ----
    for user, re in random_effects.items():
        user_data = data[data["userId"] == user]
        if len(user_data) < 2:
            continue

        # Range of X for this subject
        x_vals = np.linspace(
            user_data["breathAlcoholValue"].min(),
            user_data["breathAlcoholValue"].max(),
            20
        )

        # Random-intercept model → slope is global
        subj_intercept = fe_intercept + re["Group"]
        subj_slope = fe_slope

        y_vals = subj_intercept + subj_slope * x_vals

        ax.plot(x_vals, y_vals, alpha=0.5, linewidth=1.5)

    # ---- Global regression line ----
    x_sorted = np.linspace(data["breathAlcoholValue"].min(),
                           data["breathAlcoholValue"].max(), 40)
    y_global = fe_intercept + fe_slope * x_sorted

    ax.plot(
        x_sorted, y_global,
        color="red", linewidth=3, label="Global Mixed Model"
    )

    ax.set_title(f"{ylabel} — {title_suffix}")
    ax.set_xlabel("Breath Alcohol Value (BAC)")
    ax.set_ylabel(ylabel)
    ax.legend()



print("\n###############  PIPELINE A — INCLUDE MISSES  ###############")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (col, ylabel) in zip(axes, metrics):
    mixed_model_plot(session_A, col, ylabel, "INCLUDE MISSES", ax)

plt.tight_layout()
plt.show()


print("\n###############  PIPELINE B — EXCLUDE MISSES  ###############")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (col, ylabel) in zip(axes, metrics):
    mixed_model_plot(session_B, col, ylabel, "EXCLUDE MISSES", ax)

plt.tight_layout()
plt.show()


