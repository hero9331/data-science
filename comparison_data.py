import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

YEARS = list(range(2018, 2025))
TONS_COLS = [f"tons_{y}" for y in YEARS]

# 2. 결측치 제거(FAF) + 트럭 필터링
required_cols = ["dms_mode", "sctg2"] + TONS_COLS
missing_cols = [c for c in required_cols if c not in faf_raw.columns]

if missing_cols:
    st.error(f"FAF 데이터에 필요한 컬럼이 없습니다: {missing_cols}")
    st.stop()

if "observation_date" not in cpi_raw.columns or "CPIAUCSL_PC1" not in cpi_raw.columns:
    st.error("CPI 데이터에 `observation_date` 또는 `CPIAUCSL_PC1` 컬럼이 없습니다.")
    st.stop()

clean_df = faf_raw.dropna(subset=required_cols)
truck_df = filter_truck(clean_df)

# 연도별 FAF: 트럭·모든 품목(sctg2) 물동량 합계 (천 톤)
faf_by_year = [float(truck_df[f"tons_{y}"].sum()) for y in YEARS]

# 연도별 CPI: 해당 연도 12개월 CPIAUCSL_PC1 평균
cpi_df = cpi_raw.dropna(subset=["observation_date", "CPIAUCSL_PC1"]).copy()
cpi_df["year"] = pd.to_datetime(cpi_df["observation_date"], errors="coerce").dt.year
annual_cpi = cpi_df.groupby("year", as_index=False)["CPIAUCSL_PC1"].mean(numeric_only=True)
annual_cpi = annual_cpi.set_index("year")["CPIAUCSL_PC1"]
cpi_by_year = [float(annual_cpi.loc[y]) if y in annual_cpi.index else np.nan for y in YEARS]

# 3. 세로 막대: 가로축=연도, 연도마다 2개(왼쪽 FAF 합계, 오른쪽 CPI 연평균) — 단위가 달라 이중 y축 사용
x = np.arange(len(YEARS), dtype=float)
width = 0.36

fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
ax2 = ax.twinx()

bars_faf = ax.bar(x - width / 2, faf_by_year, width, label="FAF 트럭 총 물동량", color="steelblue")
bars_cpi = ax2.bar(x + width / 2, cpi_by_year, width, label="CPI 연평균 (12개월)", color="darkorange")

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_xlabel("연도", fontsize=14)
ax.set_ylabel("물동량 (천 톤, thousand tons)", fontsize=12, color="steelblue")
ax2.set_ylabel("CPIAUCSL_PC1 (연평균)", fontsize=12, color="darkorange")
ax.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax.set_title("연도별 FAF 트럭 총 물동량 vs CPI 연평균 (2018–2024)", fontweight="bold", fontsize=15)

from matplotlib.ticker import ScalarFormatter

for axis in (ax, ax2):
    axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axis.yaxis.get_major_formatter().set_scientific(False)

# 막대 위에 값 표시
for bar in bars_faf:
    h = bar.get_height()
    if not np.isfinite(h):
        continue
    ax.annotate(
        f"{h:,.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="steelblue",
    )

for bar in bars_cpi:
    h = bar.get_height()
    if not np.isfinite(h):
        continue
    ax2.annotate(
        f"{h:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="darkorange",
    )

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")

fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# 부연 설명
st.write("")
st.write("부연 설명")
st.write("")
st.write("- 왼쪽 막대(파란색): FAF는 `required_cols`(dms_mode, sctg2, tons_2018~2024)에 대해 결측 행을 제거한 뒤, `filter_truck`으로 트럭(`dms_mode==1`)만 두고 모든 품목(`sctg2`)의 `tons_연도`를 합한 값입니다. 단위는 천 톤(thousand tons)입니다.")
st.write("- 오른쪽 막대(주황색): `CPIAUCSL_PC1.csv`에서 `observation_date`, `CPIAUCSL_PC1` 결측 행을 제거한 뒤, 같은 연도의 월별 값 12개를 평균한 연평균입니다.")
st.write("- 물동량과 CPI의 척도가 달라 같은 y축에 두면 왜곡되므로, 왼쪽 축은 물동량, 오른쪽 축은 CPI로 따로 읽습니다.")
