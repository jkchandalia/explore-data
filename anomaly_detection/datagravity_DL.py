import tensorflow as tf
import argparse
import datetime
import keras
import logging
import numpy as np
import pandas as pd

from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.models import Model, load_model
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import  StandardScaler, MinMaxScaler



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
        df['minutes'] = df.created_date_timestamp.dt.minute+df.hour*60
        df['day'] = df.created_date_timestamp.dt.dayofweek


        df_out = pd.merge(df, labels, left_on='date_string', right_on='date')

        return df_out

    def __expand_categorical_feature(self, df, raw_feature, cutoff=0.9, num_features=10):
        """Expand categorical data using dummy variables (include null/other for df[column] with optional cutoffs)."""
        df[raw_feature] = df[raw_feature].fillna('no_' + raw_feature)
        df_agg = df.groupby(['date', raw_feature]).agg({raw_feature: 'count'}).apply(list).apply(pd.Series)
        df_agg = df_agg.unstack()

        feature_sums = df_agg.sum(axis=0).sort_values(ascending=False)
        percentile90 = np.where(np.cumsum(feature_sums).values / sum(feature_sums) > cutoff)[0][0] + 1
        max_features = min(percentile90, num_features)
        cols = list(feature_sums[0:max_features][raw_feature].index)
        cols.append('no_' + raw_feature)

        #full_raw_data.borough[~full_raw_data.borough.isin(boroughs)]='other'
        updated_features = df[raw_feature]
        updated_features[~df[raw_feature].isin(cols)] ='other_' + raw_feature
        df[raw_feature] = updated_features
        
        df_feature = pd.get_dummies(df[raw_feature])
        df = pd.merge(df, df_feature, right_index=True, left_index=True)
        df = df.drop(raw_feature, axis=1)
            
        return df

    def make_train_test_data(self, df, labels):
        """Turn raw data into features and produce test and train datasets."""
        raw_categorical_features = ['agency_name', 'borough', 'complaint_type', 'descriptor', 'location_type', 'day']
        df_expand = df
        logger.info('Creating categorical features.')

        for raw_feature in raw_categorical_features:
            logger.info('Expanding ' + raw_feature)
            df_expand = self.__expand_categorical_feature(df_expand, raw_feature)

        date_index = df_expand['date']
        labels = df_expand.chaos
        train_index = ~df_expand.chaos.isnull()
        test_index = df_expand.chaos.isnull()
        autoencoder_index = df_expand.chaos==False

        df_pared = df_expand.drop(['created_date','minutes_open','date_string','set','hour', 'date', 'created_date_timestamp', 'chaos'],axis=1)
        X = df_pared.fillna(0)
        X_autoencoder = X[autoencoder_index]

        logger.info('Scaling features.')
        scaler = MinMaxScaler()
        X_autoencoder_scaled = scaler.fit_transform(X_autoencoder)
        X_scaled = scaler.transform(X)

        return X_scaled, date_index, train_index, test_index, autoencoder_index, scaler

    def train_model(self, X_autoencoder_scaled, num_epochs=2):
        """Train an autoencoder with 10 hidden layers."""
        input_dim = X_autoencoder_scaled.shape[1]
        encoding_dim = 48

        #Network architecture
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(encoding_dim, activation="relu")(input_layer) #,activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(40), activation="relu")(encoder)
        encoder = Dense(int(32), activation="relu")(encoder)
        encoder = Dense(int(24), activation="relu")(encoder)
        encoder = Dense(int(16), activation="relu")(encoder)
        decoder = Dense(int(24), activation='relu')(encoder)
        decoder = Dense(int(32), activation='relu')(decoder)
        decoder = Dense(int(40), activation='relu')(decoder)
        decoder = Dense(int(48), activation='relu')(decoder)
        decoder = Dense(int(encoding_dim), activation='relu')(decoder)
        decoder = Dense(input_dim, activation='tanh')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        logger.info(autoencoder.summary())

        #Train parameters
        nb_epoch = num_epochs
        batch_size = 50
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        #Compile model
        autoencoder.compile(optimizer=adam, loss='mse' )

        logger.info('Starting model training.')
        t_ini = datetime.datetime.now()

        history = autoencoder.fit(X_autoencoder_scaled, X_autoencoder_scaled,
                                epochs=nb_epoch,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_split=0.2,
                                verbose=0
                                )

        logger.info('Finished model training.')
        t_fin = datetime.datetime.now()
        logger.info('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))
        
        self.model = autoencoder
        df_history = pd.DataFrame(history.history)

        return autoencoder, df_history

    def make_predictions(self, model, X, date_index):
        """Input a model and dataset and output prediction scores."""
        predictions = model.predict(X)

        mse = np.mean(np.power(X - predictions, 2), axis=1)
        df_error = pd.DataFrame({'reconstruction_error': mse})

        df_error['date']=date_index

        df_error_date=df_error.groupby(['date']).agg({'reconstruction_error':['max','min','mean','median','std']}).apply(list).apply(pd.Series)
        output = df_error_date.reconstruction_error['std'].values

        return output

def main(data_path, label_path, output_path, num_epochs):
    logger.info('Creating instance of class.')
    ad_311 = AnomalyDetector(data_path, label_path, output_path)

    logger.info('Reading in data.')
    raw_data = pd.read_csv(ad_311.data_path)
    labels = pd.read_csv(ad_311.labels_path)
    full_raw_data = ad_311.prep_311_data(raw_data, labels)

    logger.info('Creating training dataset.')
    X_scaled, date_index, train_index, test_index, autoencoder_index, scaler = ad_311.make_train_test_data(full_raw_data, labels)
    X_autoencoder_scaled = X_scaled[autoencoder_index]
    
    logger.info('Training model.')
    best_model, history = ad_311.train_model(X_autoencoder_scaled, num_epochs)

    logger.info('Making predictions.')
    predictions = ad_311.make_predictions(best_model, X_scaled, date_index)
    print(predictions)
    df_scores = pd.DataFrame({'date': labels['date'], 'scores': predictions})

    logger.info('Writing out results.')
    df_scores.to_csv(output_path + 'scores_DL_test.csv', index=False)
    
    X_train_scaled = X_scaled[train_index]
    date_index_train = date_index[train_index]
    y_train = labels.chaos[~labels.chaos.isnull()].astype(int)
    predictions_train = ad_311.make_predictions(best_model, X_train_scaled, date_index_train)
    logger.info('AUC for train data: {0:.3f}'.format(roc_auc_score(y_train, predictions_train)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="Input path to raw 311 data")
    parser.add_argument("-l", "--labels_path", type=str, help="Input path labels data")
    parser.add_argument("-o", "--output_path", type=str, help="Output path for results")
    parser.add_argument("-n", "--num_epochs", type=int, default=2, help="Number of epochs to train model")
    args = parser.parse_args()

    main(args.data_path, args.labels_path, args.output_path, args.num_epochs)
