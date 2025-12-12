import sqlite3
from typing import Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

DISTINCT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def get_dataframe(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        ld.end_to_end_latency,
        ld.load as load,
        ld.num_spec_tokens,
        ld.input_tokens,
        tm.model_name AS target_model_name,
        dm.model_name AS sd_model_name,
        sd.sd_method_type AS sd_method_type,
        ds.dataset_type AS dataset_type
    FROM ld_performances ld
    JOIN sd_setups ss ON ld.sd_setup_id = ss.sd_setup_id
    JOIN models tm ON ss.target_model_id = tm.model_id
    JOIN models dm ON ss.sd_model_id = dm.model_id
    JOIN sd_methods sd ON ss.sd_method_id = sd.sd_method_id
    JOIN datasets ds ON ss.dataset_id = ds.dataset_id
    """
    df = pd.read_sql_query(query, conn)
    return df


def add_setup_type(df: pd.DataFrame) -> pd.DataFrame:
    df["setup_type"] = df.apply(
        lambda row: (
            f"{row['target_model_name']}_{row['sd_method_type']}_sd_tokens={row['num_spec_tokens']}"
            if row["sd_method_type"] != ""
            else f"{row['target_model_name']}"
        ),
        axis=1,
    )
    return df


def plot_3d_latency_vs_load_vs_input_tokens(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    df = df.sort_values(["load"])

    fig = go.Figure()

    df["setup_type_id"] = pd.factorize(df["setup_type"])[0]
    unique_setups = sorted(df["setup_type_id"].unique())

    for i, setup_id in enumerate(unique_setups):
        group_df = df[df["setup_type_id"] == setup_id]
        setup_type = group_df["setup_type"].iloc[0]
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        fig.add_trace(
            go.Scatter3d(
                x=group_df["input_tokens"],
                y=group_df["load"],
                z=group_df["end_to_end_latency"],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.9),
                name=f"Setup {setup_type}",
                customdata=np.stack(
                    (
                        group_df["input_tokens"],
                        group_df["load"],
                        group_df["end_to_end_latency"],
                        group_df["setup_type"],
                    ),
                    axis=-1,
                ),
                hovertemplate="Number of Prompt Tokens: %{customdata[0]}<br>"
                + "VUS: %{customdata[1]}<br>"
                + "Latency: %{customdata[2]} (us)<br>"
                + "Setup Type: %{customdata[3]}<br>",
            )
        )

        if len(group_df) >= 3:
            xi = np.linspace(
                group_df["input_tokens"].min(), group_df["input_tokens"].max(), 30
            )
            yi = np.linspace(group_df["load"].min(), group_df["load"].max(), 30)
            xi, yi = np.meshgrid(xi, yi)

            zi = griddata(
                (group_df["input_tokens"], group_df["load"]),
                group_df["end_to_end_latency"],
                (xi, yi),
                method="linear",
            )

            fig.add_trace(
                go.Surface(
                    x=xi,
                    y=yi,
                    z=zi,
                    opacity=1,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    visible=True,
                    showlegend=True,
                    name=f"Surface {setup_type}",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="Number of Prompt Tokens",
            yaxis_title="VUS",
            zaxis_title="Latency (us)",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.show()

    if save_path:
        fig.write_html(save_path)


def analyze_latency_improvements(
    df: pd.DataFrame,
    model_name: str,
    vus_in_range: Optional[Set] = None,
    input_tokens_in_range: Optional[Set] = None,
    show_distribution: bool = False,
) -> pd.DataFrame:
    filtered_df = df[df.target_model_name == model_name].copy()

    if vus_in_range is not None:
        if isinstance(vus_in_range, int):
            filtered_df = filtered_df[filtered_df["load"] == vus_in_range]
        else:
            min_vus, max_vus = vus_in_range
            filtered_df = filtered_df[
                (filtered_df["load"] >= min_vus) & (filtered_df["load"] <= max_vus)
            ]

    if input_tokens_in_range is not None:
        if isinstance(input_tokens_in_range, int):
            filtered_df = filtered_df[
                filtered_df["input_tokens"] == input_tokens_in_range
            ]
        else:
            min_tokens, max_tokens = input_tokens_in_range
            filtered_df = filtered_df[
                (filtered_df["input_tokens"] >= min_tokens)
                & (filtered_df["input_tokens"] <= max_tokens)
            ]

    if filtered_df.empty:
        print("No data in the specified range")
        return None

    baseline_df = filtered_df[filtered_df.setup_type == model_name][
        ["load", "input_tokens", "end_to_end_latency"]
    ]
    baseline_df = baseline_df.rename(columns={"end_to_end_latency": "baseline_latency"})

    df_with_baseline = pd.merge(
        filtered_df, baseline_df, on=["load", "input_tokens"], how="left"
    )

    df_with_baseline["latency_improvement_pct"] = (
        (df_with_baseline["baseline_latency"] - df_with_baseline["end_to_end_latency"])
        / df_with_baseline["baseline_latency"]
        * 100
    )

    improvements_df = df_with_baseline.dropna(subset=["latency_improvement_pct"])

    median_improvements = (
        improvements_df.groupby("setup_type")["latency_improvement_pct"]
        .median()
        .reset_index()
    )
    median_improvements = median_improvements.rename(
        columns={"latency_improvement_pct": "median_improvement_pct"}
    )

    vus_in_range_str = vus_in_range if vus_in_range else "All"
    input_tokens_in_range_str = (
        input_tokens_in_range if input_tokens_in_range else "All"
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Median Latency Improvements {model_name} by Setup Type (VUS: {vus_in_range_str}, Input Tokens: {input_tokens_in_range_str})",
            (
                "Distribution of Latency Improvements by Setup Type"
                if show_distribution
                else ""
            ),
        ),
        vertical_spacing=0.2,
    )

    setup_types = median_improvements["setup_type"].unique()

    for i, setup_type in enumerate(setup_types):
        value = median_improvements[median_improvements["setup_type"] == setup_type][
            "median_improvement_pct"
        ].iloc[0]
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        fig.add_trace(
            go.Bar(
                x=[setup_type],
                y=[value],
                name=setup_type,
                marker_color=color,
                text=[f"{value:.1f}%"],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Setup Type", row=1, col=1)
    fig.update_yaxes(title_text="Median Improvement (%)", row=1, col=1)

    if show_distribution:
        for i, setup_type in enumerate(improvements_df["setup_type"].unique()):
            if setup_type != "single_model_setup":
                setup_data = improvements_df[
                    improvements_df["setup_type"] == setup_type
                ]["latency_improvement_pct"]
                color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

                fig.add_trace(
                    go.Histogram(
                        x=setup_data,
                        name=setup_type,
                        marker_color=color,
                        opacity=0.7,
                        nbinsx=25,
                    ),
                    row=2,
                    col=1,
                )

        fig.update_xaxes(title_text="Improvement (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_layout(
        height=1500, showlegend=True, legend=dict(orientation="h", xanchor="right", x=1)
    )

    fig.show()


def get_color_mapping(df):
    unique_setups = sorted(df["setup_type"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_setups))
    return {setup: palette[i] for i, setup in enumerate(unique_setups)}


def plot_2d_latency_vs_context(
    df: pd.DataFrame | None,
    dataset_type: str,
    model_name: str,
):
    dataset = df[
        (df.dataset_type == dataset_type)
        & (df.target_model_name == model_name)
        & (df.load == 1)
    ]
    color_map = get_color_mapping(dataset)
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    dataset = dataset.sort_values(["input_tokens"])
    sns.lineplot(
        data=dataset,
        x="input_tokens",
        y="end_to_end_latency",
        hue="setup_type",
        palette=color_map,
        marker="o",
        alpha=0.8,
        ax=ax,
    )

    for setup_type, color in color_map.items():
        subset = dataset[dataset["setup_type"] == setup_type]
        n_points = len(subset)
        if n_points > 2:
            label_points = subset.iloc[[0, n_points // 2, -1]]
        else:
            label_points = subset

        for _, row in label_points.iterrows():
            ax.text(
                row["input_tokens"],
                row["end_to_end_latency"],
                f"{row['num_spec_tokens']}",
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5),
            )

    ax.set_xlabel("Context", fontsize=12)
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=12)
    ax.set_title(
        f"Latency vs Context for {model_name}\n Dataset {dataset_type}", fontsize=14
    )

    handles = [
        plt.Line2D([0], [0], color=color_map[setup], lw=2) for setup in color_map
    ]
    plt.legend(handles, color_map.keys(), title="Setup Type")

    plt.tight_layout()
    plt.show()
