import pandas as pd


def _is_true(x):
    return x == "t"


def _parse_percentage(x):
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x):
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_energy(energy: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    cols = list(energy.columns)
    cols[0] = "ATX"
    energy.columns = cols
    return energy


