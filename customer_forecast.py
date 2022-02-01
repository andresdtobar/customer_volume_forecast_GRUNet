import pandas as pd
import numpy as np
import sys

import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed, RepeatVector, MaxPooling1D, Conv1D, Flatten, Input, Reshape, BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras import optimizers
from keras import backend as K
import keras

from sklearn.preprocessing import MinMaxScaler

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tienda = 'Tienda83'
# python customer_forecasting update Tienda83
# python customer_forecasting train Tienda83
# python customer_forecasting predict Tienda83

action = sys.argv[1]
tienda = sys.argv[2]

def load_data(dataframe, target_col, win_len, pred_points, test_size, normalise_window):

    data = dataframe.loc[:,[target_col]]
    n = data.shape[0]
    data_train = data.iloc[:int(n*(1-test_size)),:]
    data_test = data.iloc[int(n*(1-test_size)):,:]

    sc = MinMaxScaler()     
    if normalise_window:           
        data_train = sc.fit_transform(data_train)
        data_test = sc.transform(data_test)
        
    sequence_length = win_len + pred_points

    X_train, y_train = build_windowed_mat(data_train, win_len, sequence_length)
    X_test, y_test =  build_windowed_mat(data_test, win_len, sequence_length) 

    return X_train, y_train, X_test, y_test, sc  

def build_windowed_mat(data, win_len, sequence_length):
    n = data.shape[0]    

    X = np.array([data[index: index + sequence_length,:] for index in range(n - sequence_length)])

    pred_points = sequence_length - win_len
    y = X[:,-pred_points:]
    X = X[:,:-pred_points]

    return X,y

def sequence_prediction(curr_frame, model, future_points, pred_points, normalize, sc):
    if len(curr_frame.shape) == 1:
        curr_frame = curr_frame[np.newaxis,:,np.newaxis]
    elif len(curr_frame.shape) == 2:
        curr_frame = curr_frame[np.newaxis,:,:]

    predicted = []   
    N = int(np.ceil(future_points/pred_points) )
    for i in range(N):     
        #with tf.device('/cpu:0'):   
        predicted += model.predict(curr_frame)[0,:,0].tolist()
        curr_frame = np.array(curr_frame[0,pred_points:,0].tolist() + predicted[-pred_points:])    
        curr_frame = curr_frame[np.newaxis,:,np.newaxis]
    predicted = np.array(predicted)   
    
    if normalize:
        predicted = np.round(sc.inverse_transform(predicted.reshape(-1, 1)))
    
    return predicted[1:]


class customer_forecast():
    def __init__(self):
        # Define input parameters
        self.target_col = 'No_Clientes'
        self.pred_points = 1
        self.window_len = 24
        self.future_points = 11
        self.normalize = True
        self.test_size = .1

    def update_data(self, tienda):
        #Load old data
        old_data = pd.read_parquet('data/model_data_'+ tienda + 'last.parquet')

        #Load new data
        file = 'nuevos_datos_' + tienda
        new_data = pd.read_csv('data/' + file + '.csv')

        # Prepare new data
        new_data.FechaEntrega = pd.to_datetime(new_data.FechaEntrega)
        new_data.FechaHoraLlegada = pd.to_datetime(new_data.FechaHoraLlegada)
        new_data['Anio'] = new_data.FechaEntrega.dt.year
        new_data['Mes'] = new_data.FechaEntrega.dt.month
        new_data['Dia'] = new_data.FechaEntrega.dt.day
        new_data['Dia_llegada'] = new_data.FechaHoraLlegada.dt.day
        new_data['Hora'] = new_data.FechaHoraLlegada.dt.hour
        new_data['FechaHora'] = new_data.FechaHoraLlegada.dt.strftime(r'%Y-%m-%d:%H')

        new_data = new_data.query('Dia == Dia_llegada')

        #new_data = new_data[['FechaHora', 'Id_Solicitud_Entrega', 'TotalPaquetesEntregados', 'Anio', 'Mes', 'Dia', 'DiaSemana', 'Hora']]
        new_data = new_data[['FechaHora', 'Id_Solicitud_Entrega', 'TotalPaquetesEntregados', 'Anio', 'Mes', 'Dia', 'Hora']]
        new_data = new_data.groupby(['Anio','Mes', 'Dia','Hora']).agg(
            No_Clientes = ('Id_Solicitud_Entrega', 'nunique'), 
            Total_paquetes = ('TotalPaquetesEntregados', 'sum'),
            FechaHora = ('FechaHora', 'min'),    
            ).reset_index().sort_values(['FechaHora']).reset_index()
        new_data = new_data[['FechaHora', 'Anio', 'Mes', 'Dia', 'Hora', 'Total_paquetes', 'No_Clientes']].sort_values('FechaHora')

        # Append new data to old data
        model_df = old_data.append(new_data, ignore_index=True)

        # Save updated data
        model_df.to_parquet('data/model_data_' + tienda + 'last.parquet', index=False)
        model_df.set_index('FechaHora', inplace=True)
        

    def train(self, tienda):          
        # Read prepared data 
        model_df = pd.read_parquet('data/model_data_'+ tienda +'last.parquet')

        # Build windowed data
        X_train, y_train, _, _, _ = load_data(model_df, 'No_Clientes', self.window_len, self.pred_points, self.test_size, self.normalize)

        # Build model
        model = Sequential()        
        n_timesteps,n_features = X_train.shape[1], X_train.shape[2]
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu', input_shape=(n_timesteps, n_features)))    
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))    
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

        model.add(RepeatVector(self.pred_points))

        model.add(GRU(self.window_len, activation='relu', return_sequences=True))
        model.add(Dropout(0.25))    
        model.add(TimeDistributed(Dense(32)))
        model.add(TimeDistributed(Dense(1)))    
        model.compile(loss='mse', metrics='mae', optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4, ))
        print(model.summary()) 

        #Train model
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
        model.fit(X_train, y_train, batch_size=128, epochs=200, validation_split = 0.1, shuffle = False, callbacks = [early_stop])
        #model.fit(X_train, y_train, batch_size=128, epochs=200, validation_data = (X_test, y_test), shuffle = False, callbacks = [early_stop])
        
        # Save model
        model.save('models/Modelo'+ tienda +'Win24ConvEncUVvf')
       

    def predict_next(self, tienda):

        # Load data
        model_df = pd.read_parquet('data/model_data_'+ tienda +'last.parquet')

        # Build windowed data
        _, _, X_test, _, sc = load_data(model_df, 'No_Clientes', self.window_len, self.pred_points, self.test_size, self.normalize)

        # Load model
        model = keras.models.load_model('models/Modelo'+ tienda +'Win24ConvEncUVvf')

        # Make predictions
        predicted = sequence_prediction(X_test[-1,:,:], model, self.future_points, self.pred_points, self.normalize, sc)

        print(predicted)
        np.savetxt('output/predictions.txt', predicted, fmt='%i') 


if __name__ == '__main__':
    model = customer_forecast()

    if action == 'update':
        model.update_data(tienda)
    elif action == 'train':
        model.train(tienda)
    elif action == 'predict':
        model.predict_next(tienda)

