import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from dateutil.easter import easter


# Set of fixed-date holidays (month-day format)
FIXED_HOLIDAYS = {
    "01-01",  # novo leto
    "01-02",  # novo leto
    "02-08",  # Prešeren Day
    "04-27",  # dan upora proti okupatorju
    "05-01",  # praznik dela
    "05-02",  # praznik dela
    "06-25",  # dan drzavnosti
    "08-15",  # marijino vnebozetje
    "10-31",  # dan reformacije
    "11-01",  # dan spomina na mrtve
    "12-25",  # bozic
    "12-26",  # dan samostojnosti in enotnosti
}

# function to calculate when is easter in that year
def get_easter_related_holidays(year: int) -> set:
    easter_sunday = easter(year)
    easter_monday = easter_sunday + timedelta(days=1)
    pentecost_sunday = easter_sunday + timedelta(days=49)

    return {
        easter_sunday,
        easter_monday,
        pentecost_sunday
    }


# Define school holiday rules (approximate common ranges)
# could add jesenske pocitnice, zimske pocitnice (februar) but would need api for that
def is_school_holiday(month_day: str) -> bool:
    return (
        "06-25" <= month_day <= "08-31" or  # summer holidays
        "12-25" <= month_day <= "12-31" or  # Christmas holidays
        "01-01" <= month_day <= "01-02"     # New Year's break
    )

# Main annotation function
def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month_day'] = df['timestamp'].dt.strftime('%m-%d')

    # Initial is_holiday from fixed dates
    df['is_holiday'] = df['month_day'].isin(FIXED_HOLIDAYS).astype(np.int8)

    # Add school holidays
    df['is_school_holiday'] = df['month_day'].apply(is_school_holiday).astype(np.int8)

    # Easter-based holidays
    years = df['timestamp'].dt.year.unique()
    easter_holidays = set()
    for year in years:
        easter_holidays.update(get_easter_related_holidays(year))

    df['date_only'] = df['timestamp'].dt.date
    # Add to is_holiday if the date matches an Easter-related one
    df['is_holiday'] |= df['date_only'].isin(easter_holidays).astype(np.int8)

    # Drop helper columns
    df = df.drop(columns=['month_day', 'date_only'])
    return df
