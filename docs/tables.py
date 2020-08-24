"""Functions which extend the functionality of pandas methods."""
from typing import Any, Iterable, Mapping, Union

import pandas as pd


def freq_and_prop(
    series: Union[pd.Series, Iterable[Any]], **value_counts_kwargs: Mapping[Any, Any]
) -> pd.DataFrame:
    """Extends series.value_counts() by adding a proportion column."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    item_name = series.name
    res = pd.concat(
        [
            series.value_counts(
                **value_counts_kwargs
                # since 'count' is a method, better to use
                # 'freq' as the column name
            ).rename("freq"),
            series.value_counts(normalize=True, **value_counts_kwargs).rename("prop"),
        ],
        axis=1,
    )
    res.index.rename(item_name, inplace=True)
    return res


def get_percentile_from_freq(
    freq: pd.Series,
    p: float = 0.5,
    ascending: bool = False,
    include_boundary: Union[bool, None] = None,
) -> pd.Series:
    """Returns all the counts up to a given percentile."""
    freq = freq.sort_values(ascending=ascending)
    cutoff_idx = (freq.cumsum() >= p * freq.sum()).idxmax()

    if include_boundary is None:
        if ascending:
            # If your looking at the smallest, if there is a large
            # jump in count, you will get way too much of the data
            # so probably *dont*
            include_boundary = False
        else:
            # If you're looking at the topx largest, you probably want
            # to *include* the count just larger than the sum (because)
            # this is probably a large proportion of the total count
            include_boundary = True
    if include_boundary:
        return freq.loc[:cutoff_idx]
    return freq.loc[:cutoff_idx].iloc[:-1]
