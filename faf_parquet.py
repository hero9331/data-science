"""
FAF5.parquet (GitHub release v1.0) 로딩 및 스키마 정보.

실측 (pandas.read_parquet 로 원격 URL 직접 읽음):
  - 행 수: 2,494,901
  - 열 수: 38
  - dms_mode, sctg2, tons_2018 ~ tons_2024, value_*, current_value_*, tmiles_* : 결측 0%
  - fr_orig, fr_dest, fr_inmode, fr_outmode 등: 국내외 구간에 따라 결측 비율 높음 (집계 미사용 시 무시 가능)

트럭: FAF 메타 기준 dms_mode == 1

앱 스크립트에서는 보통 `dms_mode`, `sctg2`, `tons_2018`~`tons_2024` 등
집계에 쓰는 열에 대해 `dropna` 한 뒤 `filter_truck`을 적용합니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FAF5_PARQUET_URL = "https://github.com/bnn05195/data-science/releases/download/v1.0/FAF5.parquet"

# 실측 컬럼 목록 (순서 무관)
FAF5_COLUMNS = [
    "fr_orig",
    "dms_orig",
    "dms_dest",
    "fr_dest",
    "fr_inmode",
    "dms_mode",
    "fr_outmode",
    "sctg2",
    "trade_type",
    "dist_band",
    "tons_2018",
    "tons_2019",
    "tons_2020",
    "tons_2021",
    "tons_2022",
    "tons_2023",
    "tons_2024",
    "value_2018",
    "value_2019",
    "value_2020",
    "value_2021",
    "value_2022",
    "value_2023",
    "value_2024",
    "current_value_2018",
    "current_value_2019",
    "current_value_2020",
    "current_value_2021",
    "current_value_2022",
    "current_value_2023",
    "current_value_2024",
    "tmiles_2018",
    "tmiles_2019",
    "tmiles_2020",
    "tmiles_2021",
    "tmiles_2022",
    "tmiles_2023",
    "tmiles_2024",
]


def read_faf5_parquet() -> pd.DataFrame:
    """data/FAF5.parquet 가 있으면 로컬 우선, 없으면 GitHub URL에서 로드."""
    root = Path(__file__).resolve().parent
    local = root / "data" / "FAF5.parquet"
    if local.is_file():
        df = pd.read_parquet(local)
    else:
        df = pd.read_parquet(FAF5_PARQUET_URL)

    missing = set(FAF5_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            "FAF5.parquet 컬럼이 예상과 다릅니다. "
            f"누락: {sorted(missing)}. 파일 버전을 확인하세요."
        )
    return df


def filter_truck(df: pd.DataFrame) -> pd.DataFrame:
    """트럭만 (dms_mode == 1). dms_mode 는 int64."""
    return df.loc[df["dms_mode"] == 1].copy()
