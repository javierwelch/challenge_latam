import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV

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

def get_delay_rate_graph(data, grouping_column):
    """Generates bar graph with delay rate of flight in *data* by *grouping_column*.

    :param data: Data to be plotted. Must have "delay_15" and *grouping_column* among columns.
    :type data: DataFrame
    :param grouping_column: Column of *data* to group delay rates by.
    :type grouping_column: string
    """
    d = data.groupby(grouping_column, as_index = False).mean()[[grouping_column, 'delay_15']]
    d['delay_15'] = d['delay_15']*100
    
    global_mean = np.nanmean(data['delay_15'])*100
    
    fig = sns.barplot(data = d.sort_values('delay_15', ascending = False), 
                    x=grouping_column, y = 'delay_15')
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    fig.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.axhline(global_mean)
    fig.set_title(f"Proportion of Delayed Flights by {grouping_column}")
    fig.set_ylabel("Proportion of Delayed Flights")
    fig.set_xlabel(grouping_column)
    plt.show()

def get_to_hub_flag(flight, foreign_hubs):
    """Checks if given flight is made by a foreign company to their hub.

    :param flight: Information about the flight to be checked.
    :type flight: Series
    :param foreign_hubs: Foreign companies and the city on which their hub is located.
    :type foreign_hubs: DataFrame
    :return: True if the flight is made by a foreign company to their hub.
    :rtype: boolean
    """
    try:
        hub = foreign_hubs.loc[foreign_hubs['OPERA'] == flight['OPERA'], 'SIGLADES'].values[0]
        return hub == flight['SIGLADES']
    except:
        return False

def encode_feats(encoder, feature_df):
    """Encodes features in *feature_df* using *encoder+

    :param encoder: Fitted Encoder
    :type encoder: Encoder
    :param feature_df: Original feature DataFrame
    :type feature_df: DataFrame
    :return: Encoded feature DataFrame
    :rtype: DataFrame
    """
    return pd.DataFrame(data=encoder.transform(feature_df), columns=encoder.get_feature_names_out())

def get_best_model_CV(X_train, y_train, classifier, param_grid, scoring = 'f1_micro', n_jobs = -1):
    """Finds the best parameter combination in *param_grid* for *classifier* through cross validation.

    :param X_train: Input features for the model.
    :type X_train: array-like
    :param y_train: Target labels.
    :type y_train: array-like
    :param classifier: Model to train and optimize.
    :type classifier: sklearn model
    :param param_grid: Grid with possible parameter values for *classifier*
    :type param_grid: dict
    :param scoring: Scoring function used to evaluate the model, defaults to 'f1_micro'
    :type scoring: str, optional
    :param n_jobs: Number of cores to use for cross-validated grid search, defaults to -1
    :type n_jobs: int, optional
    :return: Best model found trained
    :rtype: sklearn model
    """

    clf = GridSearchCV(classifier, param_grid=param_grid, scoring = scoring, n_jobs = n_jobs, verbose = True)
    
    best_clf = clf.fit(X_train, y_train)

    return best_clf