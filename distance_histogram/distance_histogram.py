import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import platform
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------
# 스타일 및 설정
# ---------------------------------------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

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

# ---------------------------------------------------------
# 데이터 로드 및 전처리 
# ---------------------------------------------------------
@st.cache_data
def load_faf():
    url = "https://github.com/bnn05195/data-science/releases/download/v1.0/FAF5.parquet"
    return pd.read_parquet(url)

# 지도 좌표 및 약어 데이터
STATE_CENTROIDS = {
    'AL': [32.8, -86.7], 'AK': [61.3, -152.4], 'AZ': [34.1, -111.9], 'AR': [34.9, -92.3], 'CA': [36.1, -119.6],
    'CO': [39.0, -105.7], 'CT': [41.5, -72.7], 'DE': [39.3, -75.5], 'FL': [27.7, -81.6], 'GA': [33.0, -83.6],
    'HI': [21.0, -157.8], 'ID': [44.2, -114.4], 'IL': [40.3, -88.9], 'IN': [39.8, -86.2], 'IA': [42.0, -93.4],
    'KS': [38.5, -98.3], 'KY': [37.6, -84.6], 'LA': [31.1, -91.8], 'ME': [44.6, -69.3], 'MD': [39.0, -76.8],
    'MA': [42.2, -71.8], 'MI': [43.3, -84.5], 'MN': [45.6, -93.9], 'MS': [32.7, -89.6], 'MO': [38.4, -92.2],
    'MT': [46.9, -110.4], 'NE': [41.1, -98.2], 'NV': [38.3, -117.0], 'NH': [43.4, -71.5], 'NJ': [40.2, -74.5],
    'NM': [34.8, -106.2], 'NY': [42.1, -74.9], 'NC': [35.6, -79.8], 'ND': [47.5, -100.5], 'OH': [40.3, -82.7],
    'OK': [35.5, -96.9], 'OR': [44.5, -122.1], 'PA': [40.5, -77.2], 'RI': [41.6, -71.5], 'SC': [33.8, -80.9],
    'SD': [44.2, -99.4], 'TN': [35.7, -86.6], 'TX': [31.0, -97.5], 'UT': [40.1, -111.8], 'VT': [44.0, -72.7],
    'VA': [37.7, -78.1], 'WA': [47.4, -121.4], 'WV': [38.4, -80.9], 'WI': [44.2, -89.6], 'WY': [42.7, -107.3],
    'DC': [38.9, -77.0]
}
STATE_ABBR = {
    '1': 'AL', '2': 'AK', '4': 'AZ', '5': 'AR', '6': 'CA', '8': 'CO', '9': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
    '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY'
}

faf_raw = load_faf()

year_cols = [c for c in faf_raw.columns if c.startswith("tons_")]
available_years = sorted([c.replace("tons_", "") for c in year_cols])

truck_df = faf_raw.dropna(subset=["dms_mode", "dms_orig", "dms_dest"]).copy()
truck_df = truck_df[truck_df["dms_mode"] == 1]
truck_df["orig_state_clean"] = (truck_df["dms_orig"] // 10).astype(int).astype(str)
truck_df["dest_state_clean"] = (truck_df["dms_dest"] // 10).astype(int).astype(str)

# ---------------------------------------------------------
# UI 및 하버사인 거리 계산 로직
# ---------------------------------------------------------
st.header("운송 거리에 따른 트럭 물동량 분포")

selected_year = st.selectbox("연도 선택", options=available_years, index=0)
target_year_col = f"tons_{selected_year}"

# 주 내 이동(거리=0) 제외하고 주간 경로 집계
corridor_df = truck_df[truck_df['orig_state_clean'] != truck_df['dest_state_clean']].copy()
corridor_tons = corridor_df.groupby(['orig_state_clean', 'dest_state_clean'])[target_year_col].sum().reset_index()

corridor_tons['orig_abbr'] = corridor_tons['orig_state_clean'].map(STATE_ABBR)
corridor_tons['dest_abbr'] = corridor_tons['dest_state_clean'].map(STATE_ABBR)
corridor_tons = corridor_tons.dropna(subset=['orig_abbr', 'dest_abbr'])

corridor_tons['orig_lat'] = corridor_tons['orig_abbr'].apply(lambda x: STATE_CENTROIDS.get(x, [0,0])[0])
corridor_tons['orig_lon'] = corridor_tons['orig_abbr'].apply(lambda x: STATE_CENTROIDS.get(x, [0,0])[1])
corridor_tons['dest_lat'] = corridor_tons['dest_abbr'].apply(lambda x: STATE_CENTROIDS.get(x, [0,0])[0])
corridor_tons['dest_lon'] = corridor_tons['dest_abbr'].apply(lambda x: STATE_CENTROIDS.get(x, [0,0])[1])

# 하버사인 공식 함수 정의
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # 마일 기준
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# 거리 계산 적용
corridor_tons['distance_miles'] = corridor_tons.apply(
    lambda row: calculate_distance(row['orig_lat'], row['orig_lon'], row['dest_lat'], row['dest_lon']), axis=1
)

# ---------------------------------------------------------
# 히스토그램 시각화 
# ---------------------------------------------------------
fig_dist = px.histogram(
    corridor_tons,
    x="distance_miles",
    y=target_year_col,
    nbins=40,
    # marginal="box" 부분을 삭제하여 상단 박스 플롯을 제거합니다.
    color_discrete_sequence=['#2E86C1'],
    labels={"distance_miles": "직선 거리 (Miles)", target_year_col: "물동량 합계(천 톤)"}
)

fig_dist.update_layout(
    title=f"<b>[{selected_year}] 운송 거리에 따른 트럭 물동량 집중도</b>",
    xaxis_title="주요 도시 간 운송 거리 (Miles)",
    yaxis_title="합계 물동량 (천 톤)",
    bargap=0.1,
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------
# 그래프 부연 설명 
# ---------------------------------------------------------
st.markdown("""
* **메인 막대그래프 (히스토그램):** 가로축(운송 거리)을 일정 구간으로 나누어, 각 거리 구간마다 트럭에 실린 화물이 얼마나 되는지 쌓아 올린 그래프입니다. 
* **분석 인사이트:** 막대가 가장 높게 솟은 구간(약 0~500마일 사이)이 트럭 운송 수요가 가장 집중된 '핵심 운영 거리'입니다. 거리가 멀어질수록 철도나 항공 등 다른 운송 수단과의 경쟁으로 인해 트럭 물동량이 급격히 감소하는 경향을 확인할 수 있습니다.
""")