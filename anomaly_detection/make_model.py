# Load needed packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
import argparse


class Anomaly_Detector:
    def __init__(self, data_path, labels_path, output_path):
        self.data_path = data_path
        self.labels_path = labels_path
        self.output_path = output_path
        self.raw_data = pd.read_csv(data_path)
        self.labels = pd.read_csv(labels_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None

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
        df['log_minutes'] = np.log(full_data.minutes_open + 0.00001)

        df_out = pd.merge(df, labels, left_on='date_string', right_on='date')

        return df_out

    def aggregate_categorical_feature(self, df, raw_feature, cols=None, cutoff=0.9, num_features=10):
        """Aggregate categorical data (including null/other) for df[column] with optional cutoffs."""
        df_agg = df.fillna('no_' + raw_feature).groupby(['date', column]).agg({raw_feature: 'count'}).apply(list).apply(pd.Series)
        df_agg = df_agg.unstack()
        if columns is None:
            feature_sums = df_agg.sum(axis=0).sort_values(ascending=False)
            percentile90 = np.where(np.cumsum(feature_sums).values / sum(feature_sums) > cutoff)[0][0] + 1
            max_features = min(percentile90, num_features)
            cols = list(feature_sums[0:max_features][raw_feature].index)
        if 'no_' + raw_feature not in cols:
            cols.append('no_' + raw_feature)

        other_cols = list(set(df_agg[raw_feature].columns) - set(cols))
        other_values = df_agg[raw_feature][other_cols].sum(axis=1)
        df_agg = df_agg[raw_feature][cols]
        df_agg['other_' + raw_feature] = other_values
        df_agg = df_agg.div(df_agg.sum(axis=1), axis=0)

        return df_agg, cols

    def make_duplicate_ratio(self, df):
        """Output a ratio of (# of records)/(deduped # of records)."""
        df_dedup = df.drop_duplicates()
        df_dedup = df_dedup.groupby(['date']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_dup = df.groupby(['date']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_out = df_dup
        df_out['duplicate_ratio'] = list(df_dup['date'].values / df_dedup['date'])
        df_out.drop('date', axis=1, inplace=True)
        return df_out

    def get_dropped_rows_metric(self, df, hour_bin_size=6):
        """Output the fraction of records in each bin throughout the day."""
        df['hour_bins'] = np.floor(df.hour / hour_bin_size)
        df_agg = df.groupby(['date', 'hour_bins']).agg({'hour_bins': 'count'}).apply(list).apply(pd.Series)
        df_agg = df_agg.unstack()
        df_agg = df_agg['hour_bins']
        df_agg = df_agg.div(df_agg.sum(axis=1), axis=0)

        return df_agg

    def get_weekday(self, df):
        """Output True if a weekday, else False."""
        df_out = pd.to_datetime(df.date).dt.dayofweek
        out.index = df.date
        out = pd.DataFrame((out < 5).astype(int))
        out.columns = ['weekday']
        return out

    def summarize_numerical_feature(self, df, column):
        """Output mean of numerical data in df[column]."""
        df_out = df.fillna(0).groupby(['date']).agg({column: 'mean'}).apply(list).apply(pd.Series)
        df_out = pd.DataFrame(df_out[column])
        return df_out

    def make_train_test_data(self, df):
        # Aggregate features
        df_train_raw = df.dropna(subset=['chaos'], inplace=True)
        df_test_raw = df.loc[df.chaos.isnull(), :].reset_index(drop=True)
        df_count = df.groupby(['date', 'chaos']).agg({'date': 'count'}).apply(list).apply(pd.Series)
        df_count.columns = ['count']
        df_count = df_count.reset_index()

        df_agency, agency_features = self.aggregate_categorical_feature(df_train_raw, 'agency_name')
        df_agency_test, agency_features = self.aggregate_categorical_feature(df_test_raw, 'agency_name', cols=agency_features)

        df_borough, borough_features = self.aggregate_categorical_feature(df_train_raw, 'borough')
        df_borough, borough_features = self.aggregate_categorical_feature(df_train_raw, 'borough', cols=borough_features)

        df_complaint, complaint_features = self.aggregate_categorical_feature(df_train_raw, 'complaint_type')
        df_complaint, complaint_features = self.aggregate_categorical_feature(df_train_raw, 'complaint_type', cols=complaint_features)

        df_descriptor, descriptor_features = self.aggregate_categorical_feature(df_train_raw, 'descriptor')
        df_descriptor, descriptor_features = self.aggregate_categorical_feature(df_test_raw, 'descriptor', cols=descriptor_features)

        df_location, location_features = self.aggregate_categorical_feature(df_train_raw, 'location_type')
        df_location_test, location_features = self.aggregate_categorical_feature(df_test_raw, 'location_type', cols=location_features)

        df_dayofweek = self.get_weekday(df)
        df_dup = self.make_duplicate_ratio(df)
        df_dropped = self.dropped_rows_metric(df)
        df_long = self.summarize_numerical_feature(df, 'longitude')
        df_lat = self.summarize_numerical_feature(df, 'latitude')
        df_log_min = self.summarize_numerical_feature(df, 'log_minutes')

        df_train = pd.merge(df_count, df_agency, on='date')
        df_train = pd.merge(df_train, df_borough, on='date')
        df_train = pd.merge(df_train, df_complaint, on='date')
        df_train = pd.merge(df_train, df_descriptor, on='date')
        df_train = pd.merge(df_train, df_location, on='date')
        df_train = pd.merge(df_train, df_dup, on='date')
        df_train = pd.merge(df_train, df_dropped, on='date')
        df_train = pd.merge(df_train, df_dayofweek, on='date')
        df_train = pd.merge(df_train, df_long, on='date')
        df_train = pd.merge(df_train, df_lat, on='date')
        df_train = pd.merge(df_train, df_log_min, on='date')

        df_test = pd.merge(df_count, df_agency_test, on='date')
        df_test = pd.merge(df_test, df_borough_test, on='date')
        df_test = pd.merge(df_test, df_complaint_test, on='date')
        df_test = pd.merge(df_test, df_descriptor_test, on='date')
        df_test = pd.merge(df_test, df_location_test, on='date')
        df_test = pd.merge(df_test, df_dup_test, on='date')
        df_test = pd.merge(df_test, df_dropped_test, on='date')
        df_test = pd.merge(df_test, df_dayofweek_test, on='date')
        df_test = pd.merge(df_test, df_long_test, on='date')
        df_test = pd.merge(df_test, df_lat_test, on='date')
        df_test = pd.merge(df_test, df_log_min_test, on='date')

        # Prep training data
        feature_cols = list(df_train.columns)
        feature_cols.remove('chaos')
        feature_cols.remove('date')

        X_train = df_train.fillna(0)[feature_cols]
        X_test = df_test.fillna(0)[feature_cols]
        y_train =df_train.chaos.tolist()

        return X_train, X_test, y_train

    def select_features(self, X_train, y_train):
        """Use univariate and tree-based methods to do feature selection."""

        # Feature selection
        X_train = df_full.loc[0:150, train_cols]
        y_train = df_full.chaos.astype(int)[0:151]
        X_test = df_full.loc[150:, train_cols]
        y_test = df_full.chaos.astype(int)[151:]
        print(X_train.shape)
        feature_selection_model = SelectKBest(mutual_info_classif, k=20)
        X_train_new = feature_selection_model.fit_transform(X_train, y_train)

        features = pd.DataFrame(X_train.columns, feature_selection_model.get_support())

        clf = ExtraTreesClassifier(n_estimators=40)
        clf = clf.fit(X_train, y_train)
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_train_new = model.transform(X_train)
        X_test_new = model.transform(X_test)
        plt.hist(clf.feature_importances_, 100)
        pd.DataFrame(X_train.columns, clf.feature_importances_)

        return X_train_new

    def train_model(self, X_train, y_train, cv=3):

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
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
        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=cv, verbose=2,
                                       random_state=42, n_jobs=-1, scoring='auc')
        # Train model using some k-fold and hyperparameter tuning
        best_random = rf_random.best_estimator_
        best_random.fit(X_train, y_train)

        return best_random

    def make_predictions(self, model, X):

        # Train model and make predictions

        output = pd.DataFrame(model.predict_proba(X)[:, 1])
        return output

    # Do more exploration that is dataviz or whatnot for feature selection (use first half of data, if that's true)
    def visualize_categorical_data(self, column, labels=None):
        df_heatmap = df_full[['New York City Police Department',
                              'Department of Housing Preservation and Development',
                              'Department of Transportation',
                              'Department of Environmental Protection', 'Department of Buildings',
                              'Department of Parks and Recreation',
                              'Department of Health and Mental Hygiene', 'Department of Sanitation',
                              'Operations Unit - Department of Homeless Services',
                              'Department of Finance']].div(counts, axis=0)

        df_heatmap['label'] = df_full.chaos.astype(int)
        df_heatmap = df_heatmap[['label', 'New York City Police Department',
                                 'Department of Housing Preservation and Development',
                                 'Department of Transportation',
                                 'Department of Environmental Protection', 'Department of Buildings',
                                 'Department of Parks and Recreation',
                                 'Department of Health and Mental Hygiene', 'Department of Sanitation',
                                 'Operations Unit - Department of Homeless Services',
                                 'Department of Finance']]
        plt.figure()
        plt.pcolor(df_heatmap)
        plt.show()

    def visualize_numerical_data(self):
        pass


def main(data_path, label_path, output_path):
    ad_311 = Anomaly_Detector(data_path, label_path, output_path, feature_types)
    raw_data = pd.read_csv(ad_311.data_path)
    labels = pd.read_csv(ad_311.labels_path)
    full_raw_data = ad_311.prep_311_data(raw_data, labels)
    X_train, X_test, y_train = ad_311.make_train_test_data(full_raw_data)
    best_model = ad_311.train_model(Xtrain, y_train, cv=3)
    print(best_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="Input path to raw 311 data")
    parser.add_argument("-l", "--labels_path", type=str, help="Input path labels data")
    parser.add_argument("-o", "--output_path", type=str, help="Output path for results")
    args = parser.parse_args()

    main(args.data_path, args.labels_path, args.output_path)
