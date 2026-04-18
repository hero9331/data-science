from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

STATE_CENTROIDS = {
    "AL": (32.8, -86.7),
    "AK": (61.3, -152.4),
    "AZ": (34.1, -111.9),
    "AR": (34.9, -92.3),
    "CA": (36.1, -119.6),
    "CO": (39.0, -105.7),
    "CT": (41.5, -72.7),
    "DE": (39.3, -75.5),
    "DC": (38.9, -77.0),
    "FL": (27.7, -81.6),
    "GA": (33.0, -83.6),
    "HI": (21.0, -157.8),
    "ID": (44.2, -114.4),
    "IL": (40.3, -88.9),
    "IN": (39.8, -86.2),
    "IA": (42.0, -93.4),
    "KS": (38.5, -98.3),
    "KY": (37.6, -84.6),
    "LA": (31.1, -91.8),
    "ME": (44.6, -69.3),
    "MD": (39.0, -76.8),
    "MA": (42.2, -71.8),
    "MI": (43.3, -84.5),
    "MN": (45.6, -93.9),
    "MS": (32.7, -89.6),
    "MO": (38.4, -92.2),
    "MT": (46.9, -110.4),
    "NE": (41.1, -98.2),
    "NV": (38.3, -117.0),
    "NH": (43.4, -71.5),
    "NJ": (40.2, -74.5),
    "NM": (34.8, -106.2),
    "NY": (42.1, -74.9),
    "NC": (35.6, -79.8),
    "ND": (47.5, -100.5),
    "OH": (40.3, -82.7),
    "OK": (35.5, -96.9),
    "OR": (44.5, -122.1),
    "PA": (40.5, -77.2),
    "RI": (41.6, -71.5),
    "SC": (33.8, -80.9),
    "SD": (44.2, -99.4),
    "TN": (35.7, -86.6),
    "TX": (31.0, -97.5),
    "UT": (40.1, -111.8),
    "VT": (44.0, -72.7),
    "VA": (37.7, -78.1),
    "WA": (47.4, -121.4),
    "WV": (38.4, -80.9),
    "WI": (44.2, -89.6),
    "WY": (42.7, -107.3),
}

STATE_ABBR = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}


st.set_page_config(
    page_title="Truck Demand Forecast Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_metrics.csv")


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test_predictions.csv")


@st.cache_data
def load_forecast() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "forecast_2025.csv")


@st.cache_data
def load_model_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_dataset.csv")


def add_state_abbr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["origin_abbr"] = df["origin_state"].map(STATE_ABBR)
    df["dest_abbr"] = df["dest_state"].map(STATE_ABBR)
    return df


def make_line_map(data: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    top_df = data.nlargest(60, value_col).copy()
    if top_df.empty:
        return fig

    max_value = top_df[value_col].max()
    for _, row in top_df.iterrows():
        origin = STATE_CENTROIDS.get(row["origin_abbr"])
        dest = STATE_CENTROIDS.get(row["dest_abbr"])
        if not origin or not dest:
            continue
        line_width = 0.5 + (row[value_col] / max_value) * 7
        fig.add_trace(
            go.Scattergeo(
                lon=[origin[1], dest[1]],
                lat=[origin[0], dest[0]],
                mode="lines",
                line=dict(width=line_width, color="rgba(22, 93, 255, 0.35)"),
                hoverinfo="text",
                text=(
                    f"{row['origin_state_name']} -> {row['dest_state_name']}"
                    f"<br>{value_col}: {row[value_col]:,.0f}"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=[coords[1] for coords in STATE_CENTROIDS.values()],
            lat=[coords[0] for coords in STATE_CENTROIDS.values()],
            text=list(STATE_CENTROIDS.keys()),
            mode="markers+text",
            textposition="top center",
            marker=dict(size=5, color="#ff6b6b"),
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        geo=dict(scope="usa", projection_type="albers usa", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
    )
    return fig


metrics = load_metrics()
test_predictions = add_state_abbr(load_test_predictions())
forecast = add_state_abbr(load_forecast())

pred_cols = [c for c in test_predictions.columns if c.endswith("_predicted_tons")]
label_map = {
    "linear_regression_predicted_tons": "Linear Regression",
    "random_forest_predicted_tons": "Random Forest",
}

best_row = metrics.sort_values("rmse").iloc[0]
best_pred_col = f"{best_row['model']}_predicted_tons"

# Naive baseline: use current-year tons as next-year prediction.
naive_prediction = load_model_dataset()
naive_test = naive_prediction[naive_prediction["year"] == int(best_row["test_year"])].copy()
naive_rmse_value = float(
    ((naive_test["target_next_year_tons"] - naive_test["tons"]) ** 2).mean() ** 0.5
)
naive_mae_value = float((naive_test["target_next_year_tons"] - naive_test["tons"]).abs().mean())

st.title("미국 트럭 물동량 예측 대시보드")
st.caption("FAF5 + 거리 + CPI 기반 기본 예측 결과")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("선택 모델", str(best_row["model"]))
with col2:
    st.metric("RMSE", f"{best_row['rmse']:,.1f}")
with col3:
    st.metric("MAE", f"{best_row['mae']:,.1f}")
with col4:
    st.metric("R2", f"{best_row['r2']:.4f}")

st.info(
    "쉽게 말하면, 현재 모델은 2023년 정보를 보고 2024년 물동량을 맞혀봤고 "
    f"평균 절대오차(MAE)는 약 {best_row['mae']:.0f} thousand tons입니다. "
    "다만 테스트가 1개 연도(2024)라서 아직 과신하면 안 됩니다."
)

left, right = st.columns([1, 1])
with left:
    st.subheader("모델 비교")
    compare_df = metrics.copy()
    compare_df = pd.concat(
        [
            compare_df,
            pd.DataFrame(
                [
                    {
                        "model": "naive_last_year_tons",
                        "train_year_start": None,
                        "train_year_end": None,
                        "test_year": int(best_row["test_year"]),
                        "target_year": int(best_row["target_year"]),
                        "rmse": naive_rmse_value,
                        "mae": naive_mae_value,
                        "r2": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    compare_df = compare_df.sort_values("mae", na_position="last").reset_index(drop=True)
    st.dataframe(compare_df, use_container_width=True)

with right:
    st.subheader("쉽게 보는 해석")
    if naive_mae_value < float(best_row["mae"]):
        comparison_text = (
            "현재 기본 모델은 아직 `작년 물동량을 그대로 다음 해 값으로 쓰는 단순 기준선`보다 낫지 않습니다."
        )
    else:
        comparison_text = (
            "현재 기본 모델은 단순 기준선보다 더 낫게 나왔습니다."
        )
    st.markdown(
        f"""
        - `Linear Regression`이 현재 가장 좋게 나왔습니다.
        - 현재 검증은 `2023 -> 2024` 한 번만 했습니다.
        - 단순 기준선도 같이 봐야 합니다:
          올해 물동량을 그대로 내년 값으로 본 naive baseline
        - naive baseline MAE: `{naive_mae_value:,.1f}`
        - naive baseline RMSE: `{naive_rmse_value:,.1f}`
        - {comparison_text}
        """
    )

st.subheader("실제값 vs 예측값")
selected_model_label = st.selectbox(
    "비교할 모델",
    options=[label_map.get(col, col) for col in pred_cols],
    index=[label_map.get(col, col) for col in pred_cols].index(label_map.get(best_pred_col, best_pred_col)),
)
selected_pred_col = next(col for col, label in label_map.items() if label == selected_model_label)

scatter_fig = px.scatter(
    test_predictions,
    x="actual_next_year_tons",
    y=selected_pred_col,
    hover_data=["origin_state_name", "dest_state_name", "actual_year"],
    labels={
        "actual_next_year_tons": "Actual tons",
        selected_pred_col: "Predicted tons",
    },
    title=f"{selected_model_label}: 2024 실제값 vs 예측값",
)
max_value = max(
    float(test_predictions["actual_next_year_tons"].max()),
    float(test_predictions[selected_pred_col].max()),
)
scatter_fig.add_trace(
    go.Scatter(
        x=[0, max_value],
        y=[0, max_value],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="perfect fit",
    )
)
scatter_fig.update_layout(height=520)
st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("2025 예측 Top 경로")
only_interstate = st.checkbox("주 내부 이동 제외", value=False)
forecast_view = forecast.copy()
if only_interstate:
    forecast_view = forecast_view[forecast_view["origin_state"] != forecast_view["dest_state"]]

top_n = st.slider("상위 몇 개 경로를 볼지", min_value=10, max_value=100, value=20, step=10)
top_forecast = forecast_view.nlargest(top_n, "predicted_tons").copy()
st.dataframe(
    top_forecast[
        [
            "origin_state_name",
            "dest_state_name",
            "prediction_year",
            "selected_model",
            "predicted_tons",
        ]
    ],
    use_container_width=True,
)

map_fig = make_line_map(
    forecast_view,
    value_col="predicted_tons",
    title="2025 예측 물동량 상위 Corridor 지도",
)
st.plotly_chart(map_fig, use_container_width=True)

st.subheader("검증 데이터 오차 큰 경로")
error_view = test_predictions.copy()
error_view["abs_error"] = (
    error_view["actual_next_year_tons"] - error_view[selected_pred_col]
).abs()
st.dataframe(
    error_view.nlargest(20, "abs_error")[
        [
            "origin_state_name",
            "dest_state_name",
            "actual_year",
            "actual_next_year_tons",
            selected_pred_col,
            "abs_error",
        ]
    ],
    use_container_width=True,
)
