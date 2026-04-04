import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 및 마이너스 기호 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기준 (맥은 'AppleGothic')
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_faf() -> pd.DataFrame:
    url = "https://github.com/bnn05195/data-science/releases/download/v1.0/FAF5.parquet"
    return pd.read_parquet(url)


@st.cache_data
def load_cpi() -> pd.DataFrame:
    # CPI 데이터는 크기가 작으므로 깃허브에 함께 올렸다고 가정하고 진행합니다.
    # 만약 이 파일도 못 찾는다고 에러가 나면 이 파일도 깃허브에 꼭 올려주세요.
    return pd.read_csv("data/CPIAUCSL_PC1.csv")


@st.cache_data
def load_sctg2_description() -> dict:
    try:
        meta = pd.read_excel(
            "data/FAF5_metadata.xlsx",
            sheet_name="Commodity (SCTG2)",
        )
        # columns: ['Numeric Label', 'Description']
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

# 2. 결측치 제거 + 트럭 필터링 (실제 CSV 컬럼명 사용)
#    - FAF: dms_mode (운송수단 코드), sctg2 (품목 코드), tons_2024 (2024년 물동량)
required_cols = ["dms_mode", "sctg2", "tons_2024"]
missing_cols = [c for c in required_cols if c not in faf_raw.columns]

if missing_cols:
    st.error(f"FAF 데이터에 필요한 컬럼이 없습니다: {missing_cols}")
    st.stop()

# (1) 필요한 컬럼 기준 결측치 제거
clean_df = faf_raw.dropna(subset=required_cols)

# (2) 트럭만 필터링: 메타 기준 dms_mode == 1 을 트럭으로 가정
truck_df = clean_df[clean_df["dms_mode"].astype(str) == "1"].copy()

# 정제 전/후 데이터 개수 비교
col1, col2, col3 = st.columns(3)
col1.metric(label="정제 전 (원본) 행 수", value=f"{len(faf_raw):,}")
col3.metric(label="트럭 필터링 후 행 수", value=f"{len(truck_df):,}")

# 3. 2024년 트럭 운송 품목 Top 10 막대 그래프 (제목 X)
target_year_col = "tons_2024"

top10_truck = (
    truck_df.groupby("sctg2", as_index=True)[target_year_col]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
top10_df = top10_truck.reset_index().rename(columns={"sctg2": "sctg2_code"})
top10_df["sctg2_num"] = pd.to_numeric(top10_df["sctg2_code"], errors="coerce").astype("Int64")
top10_df["label"] = top10_df["sctg2_num"].map(SCTG2_DESC_MAP)
top10_df["label"] = top10_df["label"].fillna(top10_df["sctg2_code"].astype(str))

sns.barplot(y="label", x=target_year_col, data=top10_df, palette="Blues_r", ax=ax)

# 막대 끝부분에 값(tons_2024) 표시
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
        xytext=(5, 0),  # 막대 오른쪽으로 약간 띄우기
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )

ax.set_xlabel("물동량 (천 톤, thousand tons)", fontsize=12)
ax.set_ylabel("품목(설명/코드)", fontsize=12)
ax.set_title("2024년 트럭 운송 품목 Top 10", fontweight="bold", fontsize=14)

from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)

cur_xlim = ax.get_xlim()
ax.set_xlim(cur_xlim[0], cur_xlim[1] * 1.08)

st.pyplot(fig)

# 항목 부연 설명
st.write("")
st.write("부연 설명")
st.write("")
st.write("- `label`: 품목 코드(`sctg2`)에 해당하는 품목 설명(없으면 코드 그대로 표시). 막대의 y축입니다.")
st.write("- `tons_2024`: 2024년 트럭 운송 물동량 합계. 단위는 `thousand tons`(천 톤)이며 막대의 x축입니다.")
st.write("- 트럭 추출 기준: `dms_mode == 1` 인 행만 사용합니다.")
st.write("- 결측치 제거 기준: `dms_mode`, `sctg2`, `tons_2024` 중 하나라도 NaN이면 해당 행을 제외합니다.")