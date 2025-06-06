import pandas as pd

def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'], utc=True)
    df['DateReported'] = pd.to_datetime(df['DateReported'], utc=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ReportingDelta'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
    df['HoursWorkedPerDay'] = df['HoursWorkedPerWeek'] / df['DaysWorkedPerWeek']
    return df