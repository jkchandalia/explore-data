import argparse
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S', level=logging.INFO)
logger = logging.getLogger('datagravity')


class AnomalyDetector:
    def __init__(self, data_path, labels_path, output_path):
        self.data_path = data_path
        self.labels_path = labels_path
        self.output_path = output_path
        self.model = None

    def prep_311_data(self, df, labels):
        """Read in raw 311 data, parse date, and join with labels."""
        dates = list(df['created_date'])
        date_strings = []
        new_dates = []
        for date in dates:
            new_dates.append(date.split('+')[0])
            date_strings.append(date.split(' ')[0])

        df['created_date'] = new_dates
        df['created_date_timestamp'] = pd.to_datetime(df['created_date'])
        df['date_string'] = date_strings
        df['hour'] = df.created_date_timestamp.dt.hour
        df['log_minutes'] = np.log(df.minutes_open + 0.00001)

        df_out = pd.merge(df, labels, left_on='date_string', right_on='date')

        return df_out

    def __aggregate_categorical_feature(self, df, raw_feature, cutoff=0.9, num_features=10):
        """Aggregate categorical data (including null/other) for df[column] with optional cutoffs."""
        df_agg = df.fillna('no_' + raw_feature).groupby(['date', raw_feature]).agg({raw_feature: 'count'}).apply(list).apply(pd.Series)
        df_agg = df_agg.unstack()

        feature_sums = df_agg.sum(axis=0).sort_values(ascending=False)
        percentile90 = np.where(np.cumsum(feature_sums).values / sum(feature_sums) > cutoff)[0][0] + 1
        max_features = min(percentile90, num_features)
        cols = list(feature_sums[0:max_features][raw_feature].index)

        other_cols = list(set(df_agg[raw_feature].columns) - set(cols))
        other_values = df_agg[raw_feature][other_cols].sum(axis=1)
        df_agg = df_agg[raw_feature][cols]
        df_agg['other_' + raw_feature] = other_values
        df_agg = df_agg.div(df_agg.sum(axis=1), axis=0)

        # Calculate entropy metric associated with the categorical variables
        df_chaos = df.groupby(['date']).agg({'chaos': 'min'}).apply(list).apply(pd.Series)

        df_entropy = pd.merge(df_chaos, df_agg, on='date')
        df_base_dist = df_entropy[df_entropy.chaos == False].fillna(0)

        base_distribution = (df_base_dist.mean(axis=0))[1:] + 0.000001

        def calculate_entropy(example_dist, base_dist=base_distribution):
            return entropy(example_dist, base_dist)

        df_agg = df_agg.fillna(0)
        entropy_metric = df_agg.apply(calculate_entropy, axis=1)
        df_agg['entropy_metric_' + raw_feature] = entropy_metric

        return df_agg

    def __make_duplicate_ratio(self, df):
        """Output a ratio of (# of records)/(deduped # of records)."""
        df_dedup = df.drop_duplicates()
        df_dedup = df_dedup.groupby(['date']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_dup = df.groupby(['date']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_out = df_dup
        df_out['duplicate_ratio'] = list(df_dup['date'].values / df_dedup['date'])
        df_out.drop('date', axis=1, inplace=True)
        return df_out

    def __get_dropped_rows_metric(self, df, hour_bin_size=6):
        """Output the fraction of records in each bin throughout the day."""
        df['hour_bins'] = np.floor(df.hour / hour_bin_size)
        df_agg = df.groupby(['date', 'hour_bins']).agg({'hour_bins': 'count'}).apply(list).apply(pd.Series)
        df_agg = df_agg.unstack()
        df_agg = df_agg['hour_bins']
        df_agg = df_agg.div(df_agg.sum(axis=1), axis=0)

        # Calculate entropy metric associated with the categorical variables
        df_chaos = df.groupby(['date']).agg({'chaos': 'min'}).apply(list).apply(pd.Series)

        df_entropy = pd.merge(df_chaos, df_agg, on='date')
        df_base_dist = df_entropy[df_entropy.chaos == False].fillna(0)

        base_distribution = (df_base_dist.mean(axis=0))[1:] + 0.000001

        def calculate_entropy(example_dist, base_dist=base_distribution):
            return entropy(example_dist, base_dist)

        df_agg = df_agg.fillna(0)
        entropy_metric = df_agg.apply(calculate_entropy, axis=1)
        df_agg['entropy_metric_hour_bins'] = entropy_metric

        return df_agg

    def __get_weekday(self, df):
        """Output True if a weekday, else False."""
        df_out = pd.to_datetime(df.date).dt.dayofweek
        df_out.index = df.date
        df_out = pd.DataFrame((df_out < 5).astype(int))
        df_out.columns = ['weekday']
        return df_out

    def __summarize_numerical_feature(self, df, column):
        """Output mean of numerical data in df[column]."""
        df_out = df.fillna(0).groupby(['date']).agg({column: 'mean'}).apply(list).apply(pd.Series)
        df_out = pd.DataFrame(df_out[column])

        return df_out

    def make_train_test_data(self, df, labels):
        """Turn raw data into features and produce test and train datasets."""
        df_count = df.groupby(['date']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_count.columns = ['count']
        df_count = df_count.reset_index()
        df_chaos = df.groupby(['date']).agg({'chaos': 'min'}).apply(list).apply(pd.Series)

        logger.info('Creating categorical features.')
        df_agency = self.__aggregate_categorical_feature(df, 'agency_name')
        df_borough = self.__aggregate_categorical_feature(df, 'borough')
        df_complaint = self.__aggregate_categorical_feature(df, 'complaint_type')
        df_descriptor = self.__aggregate_categorical_feature(df, 'descriptor')
        df_location = self.__aggregate_categorical_feature(df, 'location_type')
        df_dayofweek = self.__get_weekday(labels)

        logger.info('Creating numerical features.')
        df_dup = self.__make_duplicate_ratio(df)
        df_dropped = self.__get_dropped_rows_metric(df)
        df_long = self.__summarize_numerical_feature(df, 'longitude')
        df_lat = self.__summarize_numerical_feature(df, 'latitude')
        df_log_min = self.__summarize_numerical_feature(df, 'log_minutes')

        logger.info('Aggregating numerical features.')
        df_features = pd.merge(df_count, df_chaos, on='date')
        df_features = pd.merge(df_features, df_agency, on='date')
        df_features = pd.merge(df_features, df_borough, on='date')
        df_features = pd.merge(df_features, df_complaint, on='date')
        df_features = pd.merge(df_features, df_descriptor, on='date')
        df_features = pd.merge(df_features, df_location, on='date')
        df_features = pd.merge(df_features, df_dup, on='date')
        df_features = pd.merge(df_features, df_dropped, on='date')
        df_features = pd.merge(df_features, df_dayofweek, on='date')
        df_features = pd.merge(df_features, df_long, on='date')
        df_features = pd.merge(df_features, df_lat, on='date')
        df_features = pd.merge(df_features, df_log_min, on='date')

        # Prep features
        feature_cols = list(df_features.columns)
        feature_cols.remove('chaos')
        feature_cols.remove('date')

        # Feature selection done below
        feature_cols = ['count',
                        'entropy_metric_agency_name',
                        'entropy_metric_borough',
                        'entropy_metric_complaint_type',
                        'entropy_metric_descriptor',
                        'entropy_metric_location_type',
                        'entropy_metric_hour_bins',
                        'duplicate_ratio',
                        'weekday',
                        'longitude',
                        'latitude',
                        'log_minutes']

        df_train = df_features.dropna(subset=['chaos'])
        df_test = df_features.loc[df_features.chaos.isnull(), :].reset_index(drop=True)
        X_train = df_train.fillna(0)[feature_cols]
        X_test = df_test.fillna(0)[feature_cols]
        y_train =df_train.chaos.tolist()

        return X_train, X_test, y_train

    def train_model(self, X_train, y_train, cv=3):
        """Train a random forest model using random search in parameter space and return best model."""
        # Number of trees in random forest
        n_estimators = [10, 30, 100] #int(x) for x in np.linspace(start=100, stop=300, num=100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [1, 2, 3]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        model = RandomForestClassifier(random_state=0)
        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=200, cv=cv, verbose=1,
                                       random_state=42, n_jobs=-1, scoring='roc_auc')
        rf_random.fit(X_train, y_train)
        best_random = rf_random.best_estimator_
        best_random.fit(X_train, y_train)
        self.model = best_random

        return best_random

    def make_predictions(self, model, X):
        """Input a model and dataset and output prediction scores."""
        output = model.predict_proba(X)[:, 1]

        return output

    def visualize_categorical_data(self, df, columns, labels=None):
        """Input a dataframe and a list of columns and produce a heatmap visualization of the dataframe."""
        df_heatmap = df[columns]
        if labels is not None:
            df_heatmap['label'] = labels
        plt.figure()
        plt.pcolor(df_heatmap)
        plt.show()


def main(data_path, label_path, output_path):
    logger.info('Creating instance of class.')
    ad_311 = AnomalyDetector(data_path, label_path, output_path)

    logger.info('Reading in data.')
    raw_data = pd.read_csv(ad_311.data_path)
    labels = pd.read_csv(ad_311.labels_path)
    full_raw_data = ad_311.prep_311_data(raw_data, labels)

    logger.info('Creating training dataset.')
    X_train, X_test, y_train = ad_311.make_train_test_data(full_raw_data, labels)

    logger.info('Training model.')
    best_model = ad_311.train_model(X_train, y_train, cv=3)

    logger.info('Making predictions.')
    X = pd.concat([X_train, X_test]).reset_index(drop=True)
    predictions = ad_311.make_predictions(best_model, X)
    df_scores = pd.DataFrame({'date': labels.date, 'scores': predictions})

    logger.info('Writing out results.')
    df_scores.to_csv(output_path + 'scores.csv', index=False)

    logger.info('AUC for train data: {0:.3f}'.format(roc_auc_score(y_train, predictions[:(len(y_train))])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="Input path to raw 311 data")
    parser.add_argument("-l", "--labels_path", type=str, help="Input path labels data")
    parser.add_argument("-o", "--output_path", type=str, help="Output path for results")
    args = parser.parse_args()

    main(args.data_path, args.labels_path, args.output_path)
