"""Select model features (X) and target (y) from a dataframe."""

import pandas as pd

from demand_forecast.config import FEATURES, TARGET


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def make_x(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].copy()