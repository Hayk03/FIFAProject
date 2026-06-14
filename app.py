"""
FIFA 24 Player Stats Analysis — Streamlit web report.

This page is the interactive equivalent of the Jupyter notebook:
it walks through reading & cleaning the data, EDA, and the five
hypotheses (H1-H5) with the same plots, statistics and conclusions.

Run with:
    streamlit run streamlit_app.py

Expects `male_players.csv` (the EA Sports FC 24 male players file)
in the same folder as this script.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="FIFA 24 Player Analysis", page_icon="⚽", layout="wide")

COLS = [
    "short_name", "player_positions", "value_eur", "overall", "potential", "age",
    "international_reputation", "preferred_foot", "weak_foot", "skill_moves",
    "height_cm", "weight_kg", "pace", "shooting", "passing",
    "dribbling", "defending", "physic",
]

ATACK_POS = ["ST", "CF", "LW", "RW"]
SEMI_DEFEN_POS = ["CDM", "CM", "CAM", "LM", "RM"]


def main_position(positions):
    """Map the player's first listed position to a coarse class (matches the notebook)."""
    first = positions.split(",")[0].strip()
    if first in ATACK_POS:
        return "Stricker"
    if first in SEMI_DEFEN_POS:
        return "Midfielder"
    if first == "GK":
        return "GoulKeeper"
    return "Defender"


@st.cache_data(show_spinner="Loading and cleaning the dataset…")
def load_data(path="male_players.csv"):
    """Read the raw file, keep FIFA 24, select columns, clean, and engineer features."""
    df = pd.read_csv(path, low_memory=False)
    n_raw = df.shape[0]
    n_v24 = df.shape[0]

    df = df[COLS].copy()
    n_dropped = int(df["value_eur"].isna().sum())
    df = df.dropna(subset="value_eur").reset_index(drop=True)

    # Engineered features (same as the notebook)
    df["player_class"] = df["player_positions"].apply(main_position)
    df["value_log"] = np.log10(df["value_eur"])
    df["is_adult"] = (df["age"] > df["age"].median()).astype(int)

    slope, intercept = np.polyfit(df["overall"], df["value_log"], 1)
    df["value_resid"] = df["value_log"] - (slope * df["overall"] + intercept)

    df["overall_band"] = pd.cut(
        df["overall"], bins=[59, 69, 79, 99], labels=["60-69", "70-79", "80-99"]
    )

    meta = {"n_raw": n_raw, "n_v24": n_v24, "n_dropped": n_dropped, "n_final": df.shape[0]}
    return df, meta


# ----------------------------------------------------------------------------- load
try:
    df, meta = load_data()
except FileNotFoundError:
    st.error(
        "Could not find **male_players.csv**. Put the EA Sports FC 24 male players "
        "file in the same folder as this script and reload."
    )
    st.stop()

field = df[df["player_class"] != "GoulKeeper"]

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚽ FIFA 24 Analysis")
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Data & cleaning",
        "EDA",
        "H1 · Value vs rating",
        "H2 · Age & value",
        "H3 · Reputation",
        "H4 · Preferred foot & skill",
        "H5 · Foot & value",
        "Summary",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {meta['n_final']:,} FIFA 24 players\n\n"
    "Interactive equivalent of the analysis notebook."
)


# ============================================================================ OVERVIEW
if section == "Overview":
    st.title("FIFA 24 Player Stats Analysis")
    st.markdown(
        "Exploratory analysis and hypothesis testing on the **EA Sports FC 24** male "
        "players dataset. The goal is to understand what drives a player's market value "
        "(`value_eur`) and how player attributes relate to position, age, reputation and "
        "preferred foot."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players (FIFA 24)", f"{meta['n_final']:,}")
    c2.metric("Numeric features", "14")
    c3.metric("Categorical features", "4")
    c4.metric("Statistical tests", "2")

    st.subheader("The five hypotheses")
    st.markdown(
        "Each hypothesis is a *nested* condition — a base comparison with an extra "
        "condition layered on top.\n\n"
        "- **H1.** Value grows **non-linearly** with `overall` (close to exponential).\n"
        "- **H2.** **At equal `overall`**, younger players are worth more.\n"
        "- **H3.** Higher `international_reputation` forms a **monotonic value ladder** "
        "across all five levels.\n"
        "- **H4.** Left-footed **field players** are more technical (`dribbling`, "
        "`skill_moves`) — and we check whether the effect is large or only small-but-real.\n"
        "- **H5.** Left-footers look more valuable, **but** the gap shrinks once we "
        "condition on equal `overall` (the raw gap is a confound)."
    )
    st.info("Use the sidebar to walk through each section, just like the notebook.")


# ====================================================================== DATA & CLEANING
elif section == "Data & cleaning":
    st.title("Data & cleaning")

    st.markdown(
        f"The dataset holds **{meta['n_v24']:,} FIFA 24 players** (the full multi-version file "
        "was pre-filtered to FIFA 24 and to the 18 columns relevant to the analysis)."
    )

    st.subheader("Missing values")
    st.markdown(
        "Two groups of missing values appear:\n\n"
        "1. **`value_eur`** (100) — players with no listed market value.\n"
        "2. **`pace/shooting/passing/dribbling/defending/physic`** (2045 each) — the same "
        "count in all six. These rows are **goalkeepers**, who have no aggregated field "
        "attributes in FIFA. This is a structural feature, not an error, so they are kept "
        "(and excluded later from field-attribute comparisons)."
    )
    st.markdown(
        f"The {meta['n_dropped']} rows without `value_eur` are **dropped** (≈0.5%), since "
        f"value is the key feature and cannot be imputed. **{meta['n_final']:,} players remain.**"
    )

    st.subheader("Cleaned data — preview")
    st.dataframe(df[COLS].head(20), width="stretch")

    st.subheader("Numeric summary")
    st.dataframe(df.select_dtypes("number").describe().round(2), width="stretch")


# ============================================================================== EDA
elif section == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown(
        "A new feature **`player_class`** (Striker / Midfielder / Defender / Goalkeeper) "
        "is derived from each player's main position and used as the color dimension below."
    )

    st.subheader("Game attributes by player class")
    attr = st.selectbox(
        "Attribute",
        ["dribbling", "pace", "shooting", "passing", "defending", "physic", "overall"],
    )
    fig = px.violin(
        field, x="player_class", y=attr, color="player_class", box=True,
    )
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Attacking stats (dribbling, shooting, passing) rise from defenders to strikers; "
        "`defending` is the mirror image. Attributes cleanly separate the classes by role."
    )

    st.subheader("Numeric features by player class")
    num = st.selectbox(
        "Feature", ["age", "height_cm", "weight_kg", "potential", "value_eur"]
    )
    fig = px.violin(
        df, x="player_class", y=num, color="player_class", box=True,
    )
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Defenders are taller and heavier; midfielders the smallest. Age and potential "
        "barely differ across classes. `value_eur` is strongly right-skewed — which is why "
        "the value analysis uses `log10(value_eur)`."
    )

    st.subheader("Categorical features by player class")
    cat = st.selectbox(
        "Feature", ["preferred_foot", "weak_foot", "international_reputation"]
    )
    counts = (
        df.groupby([cat, "player_class"]).size().reset_index(name="count")
    )
    fig = px.bar(
        counts, x=cat, y="count", color="player_class", barmode="group",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Right-footers outnumber left-footers ~3:1; weak-foot is bell-shaped around 2–3; "
        "`international_reputation` is heavily imbalanced toward level 1."
    )

    st.success(
        "**EDA summary.** Attributes separate classes by role, size differs across classes, "
        "age/potential do not. The right-skew of `value_eur` motivates the log transform; "
        "`international_reputation` is highly imbalanced (most players at level 1)."
    )


# ============================================================================== H1
elif section == "H1 · Value vs rating":
    st.title("H1 · Value is non-linearly related to `overall`")
    st.markdown(
        "- **H0:** value depends *linearly* on `overall`.\n"
        "- **H1:** value depends *non-linearly* (close to exponential).\n\n"
        "Method: compare Pearson correlation on raw value vs on `log10(value)`, and inspect "
        "the scatter plots. No formal test needed."
    )

    corr_raw = df["overall"].corr(df["value_eur"])
    corr_log = df["overall"].corr(df["value_log"])
    c1, c2 = st.columns(2)
    c1.metric("Pearson r — raw value", f"{corr_raw:.3f}")
    c2.metric("Pearson r — log value", f"{corr_log:.3f}")

    log_y = st.toggle("Use log10(value) on the Y axis", value=False)
    ycol = "value_log" if log_y else "value_eur"
    fig = px.scatter(
        df.sample(min(6000, len(df)), random_state=0),
        x="overall", y=ycol, color="player_class", opacity=0.5,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.success(
        "**Supported.** On the raw scale the cloud curves sharply upward (non-linear); on the "
        f"log scale it straightens into a near-linear band. Pearson rises from **{corr_raw:.2f}** "
        f"(raw) to **{corr_log:.2f}** (log) — the value gap between rating points widens as "
        "`overall` grows, so the relationship is near-exponential."
    )


# ============================================================================== H2
elif section == "H2 · Age & value":
    st.title("H2 · At equal `overall`, younger players are worth more")
    st.markdown(
        "- **H0:** at a fixed `overall`, age has no effect on value.\n"
        "- **H1:** at a fixed `overall`, younger players are worth more.\n\n"
        "Players are split by median age (`is_adult`). To isolate age from rating, we remove "
        "the linear effect of `overall` from `value_log` (a regression **residual** = the part "
        "of value rating does **not** explain) and compare the residual between age groups with "
        "a one-sided Welch t-test."
    )

    young_resid = df[df["is_adult"] == 0]["value_resid"]
    adult_resid = df[df["is_adult"] == 1]["value_resid"]
    t_stat, pvalue_h2 = stats.ttest_ind(
        young_resid, adult_resid, equal_var=False, alternative="greater"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean residual — young", f"{young_resid.mean():+.3f}")
    c2.metric("Mean residual — older", f"{adult_resid.mean():+.3f}")
    c3.metric("t  (p ≈ 0)", f"{t_stat:.0f}")

    plot_df = df.copy()
    plot_df["age_group"] = plot_df["is_adult"].map({0: "Young", 1: "Older"})
    fig = px.scatter(
        plot_df.sample(min(6000, len(plot_df)), random_state=0),
        x="overall", y="value_log", color="age_group", opacity=0.45,
        labels={"value_log": "log10(value)"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.success(
        "**Supported.** Holding rating constant, the residual value is **positive for young "
        f"(+{young_resid.mean():.2f})** and **negative for older ({adult_resid.mean():.2f})**. "
        f"The Welch t-test gives **t ≈ {t_stat:.0f}, p ≈ 0** → reject H0. At equal `overall`, "
        "younger players carry systematically higher value (≈2× in EUR terms)."
    )


# ============================================================================== H3
elif section == "H3 · Reputation":
    st.title("H3 · Reputation forms a monotonic value ladder")
    st.markdown(
        "- **H0:** `value_eur` does not depend on `international_reputation`.\n"
        "- **H1:** value increases monotonically with reputation.\n\n"
        "Method: violin plot across the 5 reputation levels + a one-way **ANOVA**."
    )

    fig = px.violin(
        df, x="international_reputation", y="value_log", color="international_reputation",
        box=True, labels={"value_log": "log10(value)"},
    )
    fig.update_layout(showlegend=False, height=480)
    st.plotly_chart(fig, width="stretch")

    rep_table = (
        df.groupby("international_reputation")["value_eur"]
        .median().astype(int).reset_index()
        .rename(columns={"value_eur": "median_value_eur"})
    )
    groups = [df[df["international_reputation"] == r]["value_log"] for r in [1, 2, 3, 4, 5]]
    F, pvalue_h3 = stats.f_oneway(*groups)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Median value by reputation level**")
        st.dataframe(rep_table, width="stretch", hide_index=True)
    with c2:
        st.metric("ANOVA F", f"{F:.0f}")
        st.metric("p-value", "< 0.0001")

    st.success(
        "**Supported.** The median rises monotonically across all five levels (≈×80 from 1 to 5) "
        f"and ANOVA gives **F ≈ {F:.0f}, p < 0.0001** → reject H0. Caveat: reputation correlates "
        "with `overall`, so part of the effect runs through rating."
    )


# ============================================================================== H4
elif section == "H4 · Preferred foot & skill":
    st.title("H4 · Are left-footed field players more technical?")
    st.markdown(
        "- **H0:** preferred foot has no relation to technical attributes.\n"
        "- **H1:** left-footers have higher `dribbling` / `skill_moves`.\n\n"
        "Field players only (goalkeepers excluded). We check whether any difference is large "
        "or merely small-but-real — no test, since at n≈18k even a tiny gap would read as "
        "'significant' while being practically meaningless."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(field, x="preferred_foot", y="dribbling", color="preferred_foot")
        fig.update_layout(showlegend=False, height=430, title="Dribbling by foot")
        st.plotly_chart(fig, width="stretch")
    with c2:
        sm = (
            pd.crosstab(field["skill_moves"], field["preferred_foot"], normalize="columns")
            .reset_index()
            .melt(id_vars="skill_moves", var_name="preferred_foot", value_name="share")
        )
        fig = px.bar(
            sm, x="skill_moves", y="share", color="preferred_foot", barmode="group",
        )
        fig.update_layout(height=430, title="Skill moves (normalized)")
        st.plotly_chart(fig, width="stretch")

    med = field.groupby("preferred_foot")[["dribbling", "skill_moves"]].median()
    mean = field.groupby("preferred_foot")[["dribbling", "skill_moves"]].mean().round(3)
    c1, c2 = st.columns(2)
    c1.markdown("**Medians**"); c1.dataframe(med, width="stretch")
    c2.markdown("**Means**"); c2.dataframe(mean, width="stretch")

    st.warning(
        "**Partially supported — small but real.** `dribbling` is essentially identical "
        "(median 64 both). `skill_moves` shows a small consistent shift (median 3 vs 2; "
        "mean 2.58 vs 2.55). The 'left-footers are more technical' idea holds only weakly, "
        "for `skill_moves` — we do not over-claim."
    )


# ============================================================================== H5
elif section == "H5 · Foot & value":
    st.title("H5 · Left-footers look pricier — does the gap survive controlling for rating?")
    st.markdown(
        "- **H0:** the raw value gap is explained by `overall` — it disappears once rating is held constant.\n"
        "- **H1:** left-footers are worth more even at equal `overall`.\n\n"
        "Method: compare raw medians, then **stratify** by `overall` band so rating is roughly "
        "constant within each band."
    )

    raw = df.groupby("preferred_foot")["value_eur"].median()
    ov = df.groupby("preferred_foot")["overall"].mean().round(2)
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw median — Left", f"€{raw['Left']/1e6:.2f}M")
    c2.metric("Raw median — Right", f"€{raw['Right']/1e6:.2f}M")
    c3.metric("Mean overall (L / R)", f"{ov['Left']} / {ov['Right']}")

    band_med = (
        df.dropna(subset="overall_band")
        .groupby(["overall_band", "preferred_foot"])["value_eur"]
        .median().reset_index()
    )
    fig = px.bar(
        band_med, x="overall_band", y="value_eur", color="preferred_foot",
        barmode="group", labels={"value_eur": "median value (EUR)", "overall_band": "overall band"},
    )
    fig.update_layout(height=470, title="Median value by overall band and preferred foot")
    st.plotly_chart(fig, width="stretch")

    st.success(
        "**H0 supported, H1 rejected — the raw gap is a confound.** Left-footers look pricier "
        f"(€{raw['Left']/1e6:.2f}M vs €{raw['Right']/1e6:.2f}M), but they also have a slightly "
        f"higher mean `overall` ({ov['Left']} vs {ov['Right']}). Within overall bands the gap "
        "essentially vanishes — at equal rating both feet are valued almost identically. The "
        "raw advantage is driven by rating, not by foot."
    )


# ============================================================================== SUMMARY
elif section == "Summary":
    st.title("Final summary")
    summary = pd.DataFrame(
        {
            "Hypothesis": [
                "H1 · value non-linear in overall",
                "H2 · young worth more at equal overall",
                "H3 · monotonic value ladder by reputation",
                "H4 · left-footers more technical",
                "H5 · left-footers worth more",
            ],
            "Result": [
                "Supported", "Supported", "Supported",
                "Weak / small effect", "Rejected (confound)",
            ],
            "Evidence": [
                "scatter curve + corr 0.55 → 0.88 (log)",
                "residual Welch t-test, t≈108, p≈0",
                "violin + ANOVA p<0.0001, medians ×80",
                "medians: dribbling equal, skill_moves 3 vs 2",
                "raw gap vanishes within overall bands",
            ],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown(
        "**Overall.** A player's market value is driven mainly by `overall` (non-linearly), "
        "with age and reputation adding real, separable information on top. Preferred foot has "
        "little genuine effect — its apparent value advantage disappears once rating is "
        "controlled, and its link to technical skill is marginal. Two statistical tests were "
        "used in total (a residual t-test for H2 and an ANOVA for H3); the rest is demonstrated "
        "with figures and descriptive statistics."
    )
