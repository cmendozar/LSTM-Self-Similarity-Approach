from os import name
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

plt.style.use("bmh")

route_path = "/Users/Carlos/Documents/Tesis/"

def file(name,route):
    path =  route_path +'Data/'+ name + ".xlsx"
    return path

def create_features(data):
    df              = pd.DataFrame()
    df['p_t']       = data.close
    df['p_t-1']     = data.close.shift(1)
    df['r_t']       = np.log(data.close/data.close.shift(1))
    df['r_t-1']     = np.log(data.close.shift(1)/data.close.shift(2))
    return df.dropna()

def get_feature(features, steps, i):
    return features.iloc[i:i+steps]
def get_label(target, steps, i):
    return target.iloc[i+steps+2]

file_names =['GDAXI_1day']#['500_1day','AEX_1day', 'AXJO_1day','GDAXI_1day','N225_1day','SSMI_1day']  
#name_index = '500_1day'

list_steps = [88]#[5,22,66,88,126]
list_cells = [100]#[10,25,50,100,150,200]
results    = pd.DataFrame()
predictions = pd.DataFrame()

for name_index in file_names: 
    print(name_index)
    data = pd.read_excel(file(name_index,route_path), index_col='time')
    for steps in list_steps:
        train_cut = 5284-500-252-steps
        val_cut   = 500

        labels   = pd.DataFrame(data = data.close , columns=['close'], index = data.index)
        labels2   = pd.DataFrame(data = data.close , columns=['close'], index = data.index)                      
        features = create_features(data)

        sc_features = MinMaxScaler(feature_range = (0,1))    
        sc_labels   = MinMaxScaler(feature_range = (0,1))

        sc_features.fit(features)
        sc_labels.fit(labels)

        features    = pd.DataFrame(sc_features.transform(features))
        labels      = pd.DataFrame(sc_labels.transform(labels))


        features_set = []
        labels_set   = []

        for i in range(len(features)-steps):
            features_set.append(get_feature(features,steps,i))
            labels_set.append(get_label(labels,steps,i))

        features_set = np.array(features_set)
        labels_set   = np.array(labels_set)

        train_features = features_set[:train_cut]
        val_features   = features_set[train_cut:train_cut + val_cut]
        test_features  = features_set[train_cut+val_cut:]

        train_labels = labels_set[:train_cut]
        val_labels   = labels_set[train_cut:train_cut + val_cut]
        test_labels  = labels_set[train_cut+val_cut:]

        #cell = 10
        for cell in list_cells: 
            keras.backend.clear_session()
            model = keras.Sequential()
            model.add(layers.LSTM(cell,activation = 'tanh', input_shape = (steps,train_features.shape[2],)))
            model.add(layers.Dropout(0.2))
            #model.add(layers.LSTM(cell))
            #model.add(layers.Dropout(0.2))
            model.add(layers.Dense(10))
            model.add(layers.Dense(1))
            model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),loss = "mse")

            model.fit(train_features,train_labels,
                        epochs = 100,
                        verbose = 0,
                        batch_size = 128,
                        validation_data = (val_features,val_labels)
            )
            forecast = pd.DataFrame(sc_labels.inverse_transform(model.predict(test_features,steps = 1)))
            real     = pd.DataFrame(sc_labels.inverse_transform(test_labels))
            mse      = ((real-forecast)**2).mean()
            label    = "(" + str(cell) + ";"+ str(steps) + ")" 
            print(label,": ",mse.iloc[0])
            results[label] = mse 
            predictions = pd.concat([predictions,forecast],axis = 1)
    real.to_excel(route_path + 'Results/lstm/real_'+str(name_index)+'_lstm_1capa.xlsx')
    predictions.to_excel(route_path +'Results/lstm/forecast_'+str(name_index)+'_lstm_1capa.xlsx')
    results.to_excel(route_path + 'Results/lstm/'+str(name_index)+'_lstm_1capa.xlsx')
