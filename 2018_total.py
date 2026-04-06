import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import platform

from faf_parquet import filter_truck, read_faf5_parquet

# 운영체제에 따라 폰트 다르게 설정하기
os_name = platform.system()
if os_name == "Windows":
    plt.rc("font", family="Malgun Gothic")  # 윈도우
elif os_name == "Darwin":
    plt.rc("font", family="AppleGothic")  # 맥(Mac)
else:
    plt.rc("font", family="NanumGothic")  # 리눅스 (스트림릿 클라우드)

# 마이너스(-) 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_faf() -> pd.DataFrame:
    return read_faf5_parquet()


@st.cache_data
def load_cpi() -> pd.DataFrame:
    return pd.read_csv("data/CPIAUCSL_PC1.csv")


@st.cache_data
def load_sctg2_description() -> dict:
    try:
        meta = pd.read_excel(
            "data/FAF5_metadata.xlsx",
            sheet_name="Commodity (SCTG2)",
        )
        meta = meta.rename(columns={"Numeric Label": "sctg2", "Description": "description"})
        meta["sctg2"] = pd.to_numeric(meta["sctg2"], errors="coerce").astype("Int64")
        meta = meta.dropna(subset=["sctg2", "description"])
        return dict(zip(meta["sctg2"].astype(int), meta["description"].astype(str)))
    except Exception:
        return {}


# 1. 데이터 불러오기
faf_raw = load_faf()
cpi_raw = load_cpi()
SCTG2_DESC_MAP = load_sctg2_description()

# 2. 결측치 제거 + 트럭 필터링
required_cols = ["dms_mode", "sctg2", "tons_2018"]
missing_cols = [c for c in required_cols if c not in faf_raw.columns]

if missing_cols:
    st.error(f"FAF 데이터에 필요한 컬럼이 없습니다: {missing_cols}")
    st.stop()

clean_df = faf_raw.dropna(subset=required_cols)
truck_df = filter_truck(clean_df)

# 정제 전/후 데이터 개수 비교
col1, col2, col3 = st.columns(3)
col1.metric(label="정제 전 (원본) 행 수", value=f"{len(faf_raw):,}")
col3.metric(label="트럭 필터링 후 행 수", value=f"{len(truck_df):,}")

# 3. 2018년 트럭 운송 품목 전체 가로 막대 그래프
target_year_col = "tons_2018"

top10_truck = (
    truck_df.groupby("sctg2", as_index=True)[target_year_col]
    .sum()
    .sort_values(ascending=False)
)

top10_df = top10_truck.reset_index().rename(columns={"sctg2": "sctg2_code"})
top10_df["sctg2_num"] = pd.to_numeric(top10_df["sctg2_code"], errors="coerce").astype("Int64")
top10_df["label"] = top10_df["sctg2_num"].map(SCTG2_DESC_MAP)
top10_df["label"] = top10_df["label"].fillna(top10_df["sctg2_code"].astype(str))

n_items = len(top10_df)
fig_w, fig_h = 20, max(14, n_items * 0.55)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

sns.barplot(
    y="label",
    x=target_year_col,
    data=top10_df,
    order=top10_df["label"],
    palette="Blues_r",
    ax=ax,
)

# 막대 끝부분에 값(tons_2018) 표시
vals = top10_df[target_year_col].values
for i, patch in enumerate(ax.patches):
    if i >= len(vals):
        break
    val = vals[i]
    if pd.isna(val):
        continue
    x_end = patch.get_x() + patch.get_width()
    y_center = patch.get_y() + patch.get_height() / 2
    ax.annotate(
        f"{val:,.0f}",
        (x_end, y_center),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )

ax.set_xlabel("물동량 (천 톤, thousand tons)", fontsize=14)
ax.set_ylabel("품목(설명/코드)", fontsize=14)
ax.set_title("2018년 트럭 운송 품목별 물동량 (전체)", fontweight="bold", fontsize=16)
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=12)

from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)

cur_xlim = ax.get_xlim()
ax.set_xlim(cur_xlim[0], cur_xlim[1] * 1.08)

st.pyplot(fig, use_container_width=True)

# 항목 부연 설명
st.write("")
st.write("부연 설명")
st.write("")
st.write("- `label`: 품목 코드(`sctg2`)에 해당하는 품목 설명(없으면 코드 그대로 표시). 막대의 y축입니다.")
st.write("- `tons_2018`: 2018년 트럭 운송 물동량 합계. 단위는 `thousand tons`(천 톤)이며 막대의 x축입니다.")
st.write("- 트럭 추출: `faf_parquet.filter_truck` → `dms_mode == 1`.")
st.write("- 결측치 제거: `dms_mode`, `sctg2`, 해당 연도 `tons_연도` 중 하나라도 NaN이면 해당 행을 제외한 뒤 트럭만 필터링합니다.")
