"""Functions extending matplotlib and seaborn functions."""
from typing import Any, Iterable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tables import get_percentile_from_freq


def remove_tr_spines() -> None:
    """Remove the right and top edges from matplotlib axes."""
    fig = plt.gcf()
    axes = fig.axes
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")


def add_numbers_to_bars(
    ax: Union[plt.Axes, None] = None,
    kind: str = "pct",
    fmt: Union[str, None] = None,
    count_sum: Union[float, None] = None,
) -> None:
    """Add text to an axes labelling bars.

    Set kind to 'pct', 'count' or 'pct_count' to get the percentage, count
    (raw height of the bar), or both, respectively. If you would like to use
    a different total for the height of the bars than is displayed in this plot,
    set a value for count_sum.
    """
    valid_kind = ["pct", "count", "pct_count"]
    if ax is None:
        ax = plt.gca()
    bar_heights = [p.get_height() for p in ax.patches]
    max_height = np.nanmax(bar_heights)
    if count_sum is None:
        count_sum = np.nansum(bar_heights)
    for p in ax.patches:
        height = p.get_height()
        posx = p.get_x() + p.get_width() / 2
        posy = height + 0.001 * max_height
        if kind == "pct":
            value = height / count_sum * 100
            if fmt is not None:
                text = fmt.format(value)
            else:
                text = f"{value:1.1f}%"
        elif kind == "count":
            value = height
            if fmt is not None:
                text = fmt.format(value)
            else:
                text = f"{value:1.0f}"
        elif kind == "pct_count":
            pct = height / count_sum * 100
            cnt = height
            if fmt is not None:
                # Nice little hack allowing user to use pos or kwd args for fmt
                format_dict = dict(count=cnt, percent=pct)
                text = fmt.format(*format_dict.values(), **format_dict)
            else:
                text = f"{cnt:1.0f}, {pct:1.1f}%"
        else:
            raise ValueError(f"kind = {kind} is not valid, choose from {valid_kind}.")
        if not (np.isnan(posx) or np.isnan(posy)):
            ax.text(posx, posy, text, ha="center")


CountplotInput = Union[str, Iterable[Any]]


def countplot_with_numbers(
    var_name: CountplotInput,
    data: Union[pd.DataFrame, None] = None,
    hue: Union[CountplotInput, None] = None,
    kind: str = "pct",
    fmt: Union[str, None] = None,
    sort_cols: Union[bool, None] = None,
    count_sum: Union[float, None] = None,
    tr_spines: bool = False,
    **sns_kwargs: Mapping[Any, Any],
) -> None:
    """Extends sns.countplot by adding numbers to bars.

    var_name is passed to sns.countplot as the 'x' parameter."""
    order = None
    if sort_cols is not None:
        if data is not None:
            series = data[var_name]
        else:
            series = pd.Series(var_name)
        assert sort_cols in ["ascending", "descending"]
        if sort_cols == "descending":
            vc = series.value_counts().sort_values(ascending=False)
        elif sort_cols == "ascending":
            vc = series.value_counts().sort_values()
        order = vc.index
    ax = sns.countplot(x=var_name, hue=hue, data=data, order=order, **sns_kwargs)
    add_numbers_to_bars(ax, kind=kind, fmt=fmt, count_sum=count_sum)
    if not tr_spines:
        remove_tr_spines()


def barplot_with_numbers(
    series: pd.Series,
    kind: str = "pct",
    fmt: Union[str, None] = None,
    sort_cols: Union[bool, None] = None,
    count_sum: Union[float, None] = None,
    tr_spines: bool = False,
    **sns_kwargs: Mapping[Any, Any],
) -> None:
    """Extends sns.barplot by adding numbers to bars.

    Note that series is expected to be a pd.Series.
    """
    if sort_cols is not None:
        if sort_cols == "descending":
            series = series.sort_values(ascending=False)
        elif sort_cols == "ascending":
            series = series.sort_values()
    ax = sns.barplot(series.index, series, **sns_kwargs)
    add_numbers_to_bars(ax, kind=kind, fmt=fmt, count_sum=count_sum)
    if not tr_spines:
        remove_tr_spines()


def plot_freq(
    freq: pd.Series,
    title: Union[str, bool, None] = None,
    tr_spines: bool = False,
    kind: str = "pct",
    **get_percentile_from_freq_kwargs: Mapping[Any, Any],
) -> pd.Series:
    """Plots pd.Series frequency counts.

    The freq is ostensibly obtained using a pd.Series.value_counts() method.
    """
    if get_percentile_from_freq_kwargs:
        count_sum = freq.sum()
        _freq = get_percentile_from_freq(freq, **get_percentile_from_freq_kwargs)
        if len(_freq) == 0:
            print(
                "WARNING: the percentile setting means no frequency "
                "counts are selected, setting include_boundary to True"
            )
            get_percentile_from_freq_kwargs["include_boundary"] = True
            freq = get_percentile_from_freq(freq, **get_percentile_from_freq_kwargs)
        else:
            freq = _freq
    else:
        count_sum = None
    barplot_with_numbers(freq, kind=kind, count_sum=count_sum, tr_spines=tr_spines)
    plt.xticks(rotation=90)
    if title is None or title is False:
        pass
    elif title is True:
        if "p" in get_percentile_from_freq_kwargs:
            p = get_percentile_from_freq_kwargs["p"]
        else:
            p = 0.5
        if "ascending" in get_percentile_from_freq_kwargs:
            ascending = get_percentile_from_freq_kwargs["ascending"]
            if ascending:
                text = "Bottom"
            else:
                text = "Top"
        else:
            text = "Top"
        plt.title(f"{text} {p*100}%")
    elif isinstance(title, str):
        plt.title(title)
    else:
        raise ValueError(f"Invalid title {title}")
    return freq
