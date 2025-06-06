import pandas as pd

def load_cpi_data(filepath: str) -> pd.DataFrame:
    CPI = pd.read_csv(filepath)
    CPI['observation_date'] = pd.to_datetime(CPI['observation_date']).dt.to_period('M')
    return CPI

def adjust_for_inflation(df: pd.DataFrame, CPI: pd.DataFrame, target_col: str, date_col: str, reference_month='2024-12') -> pd.Series:
    current_CPI = CPI[CPI['observation_date'] == reference_month]['CPIAUCSL'].values[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M')
    inflation_index = df[date_col].map(CPI.set_index('observation_date')['CPIAUCSL'])
    return df[target_col] * (current_CPI / inflation_index)

def undo_inflation_adjustment(adjusted_series: pd.Series, df: pd.DataFrame, CPI: pd.DataFrame, date_col: str, reference_month='2024-12') -> pd.Series:
    current_CPI = CPI[CPI['observation_date'] == reference_month]['CPIAUCSL'].values[0]
    inflation_index = df[date_col].map(CPI.set_index('observation_date')['CPIAUCSL'])
    return adjusted_series * (inflation_index / current_CPI)