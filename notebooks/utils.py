import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_flight_seasonality_graph(data, time_column, grouping_column):
    """Generates line graph with count of values in *data* by *time_column* and *grouping_column*.
    Uses *time_column* on the x-axis and *grouping_column* as hue.

    :param data: Data to be plotted. Must have *time_column* and *grouping_column* among columns.
    :type data: DataFrame
    :param time_column: Column of *data* to use as x-axis for the plot.
    :type time_column: string
    :param grouping_column: Column of *data* to use as hue for the plot.
    :type grouping_column: string
    """
    vc_flights_by_kind_and_time = data[[time_column, grouping_column]].value_counts().reset_index()
    fig = sns.lineplot(data = vc_flights_by_kind_and_time, x = time_column, y = 0, hue=grouping_column)
    fig.set_title(f'Number of flights by {grouping_column} and {time_column}')
    fig.set_ylabel('Number of flights')
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.show()

def get_period_day(x):
    """Returns period of day corresponding to input's hour.
    Times of day are defined as follows:
    morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59)

    :param x: Timestamp for which to assign period of day.
    :type x: Timestamp
    :return: Period of day.
    :rtype: string
    """
    if x.hour >= 5 and x.hour <= 11:
        return 'morning'
    if x.hour >= 12 and x.hour <= 18:
        return 'afternoon'
    return 'night'
