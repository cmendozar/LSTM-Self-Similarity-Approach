#!/usr/bin/env python
# coding: utf-8
# autor: Carlos Mendoza R. 
# date: 31-03-2021

import numpy as np                              #  to work with arrays
import pandas as pd                             #  to work with dataframes
#pd.options.plotting.backend = "plotly"
import pandas_datareader.data as web            #  to import website data
import datetime as dt                           #  to set the datetime in a series
import matplotlib.pyplot as plt                 #  to graph the data
from sklearn.preprocessing import MinMaxScaler  #  to preprocesing the data to any range
import warnings
warnings.filterwarnings('ignore')

import keras.backend as K  
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam 

def reset_ahead(features,target,days_ahead):
    temp_data = pd.concat([target,features.shift(days_ahead-1)],axis = 1)
    temp_data = temp_data.dropna()
    return temp_data#.iloc[:,0],temp_data.iloc[:,1:]

def split_features(features, n_steps):
    features = features.values
    if len(features.shape) == 1:
        n_features = 1
        features_samples = features.shape[0]
    else:
        features_samples, n_features = features.shape 
     
    lstm_features = []
    for i in range(features_samples-n_steps):
        lstm_features.append(features[i:i+n_steps][:].reshape(n_steps,n_features))
    lstm_features = np.array(lstm_features)
    return lstm_features

def split_target(target, n_steps):
    target         = target.values
    target_samples = len(target)
    lstm_target    = []
    for i in range(n_steps, target_samples):
        lstm_target.append(target[i])
    lstm_target = np.array(lstm_target)
    return lstm_target

def get_index_fecha(data):
    d = data.reset_index()
    index = d[d['time'] <= dt.datetime(year = 2019, month = 12, day = 31)].iloc[-1].name
    return index 

def features(data):
    data = data.sort_index()
    ftrs = pd.DataFrame()
    ftrs["pt"]   = data.close
    ftrs["pt-1"] = data.close.shift(1)
    ftrs["rt"]   = data.close.pct_change()
    ftrs["rt-1"] = data.close.pct_change().shift(1)
    ftrs = ftrs.dropna()
    return ftrs


indexes = ["500","AEX","AXJO","GDAXI","N225","SSMI"]

for nombre in indexes: 
    
    data1day     = pd.read_excel("Data/"+ nombre+"_1day.xlsx", index_col = 'time')
    data1min     = pd.read_csv("Data/"+ nombre+"_1min.csv",sep = ';', index_col = 'time')
    
    train_cut_1day   = get_index_fecha(data1day)-500
    val_cut_1day     = 500
    
    train_cut_1min   = 1048575-10000
    val_cut_1min   = 10000
    
    train_cut_15min   = 69897-700
    val_cut_15min   = 700
    
    train_cut_30min  = 34949-700
    val_cut_30min    = 700
    
    train_cut_60min  = 17475-700
    val_cut_60min    = 700
    
    data1min.index = pd.to_datetime(data1min.index)
    data15min      = data1min.resample('15min').last().dropna()
    data30min      = data1min.resample('30min').last().dropna()
    data60min      = data1min.resample('60min').last().dropna()
    
    
    features1day  = features(data1day)
    features1min  = features(data1min)
    features15min = features(data15min)
    features30min = features(data30min)
    features60min = features(data60min)
    
    target1day    = pd.DataFrame(features1day.pt)
    target1min    = pd.DataFrame(features1min.pt)
    target15min   = pd.DataFrame(features15min.pt)
    target30min   = pd.DataFrame(features30min.pt)
    target60min   = pd.DataFrame(features60min.pt)
    
    target1daycopy = target1day
    
    sc_features = MinMaxScaler(feature_range = (0,1))    
    sc_target   = MinMaxScaler(feature_range = (0,1))
    
    sc_features.fit(features1day)
    sc_target.fit(target1day)
    
    features1day    = pd.DataFrame(sc_features.transform(features1day))
    target1day      = pd.DataFrame(sc_target.transform(target1day))
    
    features1min    = pd.DataFrame(sc_features.transform(features1min))
    target1min      = pd.DataFrame(sc_target.transform(target1min))
    
    features15min    = pd.DataFrame(sc_features.transform(features15min))
    target15min      = pd.DataFrame(sc_target.transform(target15min))
    
    features30min    = pd.DataFrame(sc_features.transform(features30min))
    target30min      = pd.DataFrame(sc_target.transform(target30min))
    
    features60min    = pd.DataFrame(sc_features.transform(features60min))
    target60min      = pd.DataFrame(sc_target.transform(target60min))
    
    ### SETEAR PARAMETROS
    days_ahead  = 1
    #steps       = 10
    steps_list  = [5,22,66,88,132,252]
    #cell        = 100
    cell_list   = [10,25,50,100,150,200]

    BIG_LOSS_MSE  = pd.DataFrame()
    BIG_LOSS_MAE  = pd.DataFrame()
    BIG_LOSS_RMSE = pd.DataFrame()
    BIG_LOSS_MAPE = pd.DataFrame()
    
    DA_BIG_LOSS   = pd.DataFrame()
    
    for cell in cell_list:
        for steps in steps_list:
            print("Index: " + nombre + " Celdas: " + str(cell) + " Steps: " + str(steps))
            index_test_fecha_1day  = target1daycopy.index[train_cut_1day + val_cut_1day + steps : ]
            
            index_train = target1day.index[ : train_cut_1day] 
            index_val   = target1day.index[train_cut_1day : train_cut_1day + val_cut_1day]
            index_test  = target1day.index[train_cut_1day + val_cut_1day + steps : ]
            
            X = split_features(features1day, steps)
            y = split_target(target1day, steps)
            n_steps,n_features = X.shape[1:]
            
            #SET TRAIN, VAL, TEST DATA 
            X_training   = X[:train_cut_1day][:][:]
            X_validation = X[train_cut_1day : train_cut_1day + val_cut_1day][:][:]
            X_test       = X[train_cut_1day + val_cut_1day : ][:][:]
            
            y_training   = y[:train_cut_1day][:][:]
            y_validation = y[train_cut_1day : train_cut_1day + val_cut_1day][:][:]
            y_test       = y[train_cut_1day + val_cut_1day : ][:][:]
            
            X_training1   = X_training
            X_validation1 = X_validation
            
            # RANDOMIZE DATA FOR BETTER PERFORMANC
            np.random.seed(4)
            random_idx_train = np.random.choice(len(X_training), len(X_training), replace=False)
            
            np.random.seed(4)
            random_idx_validation = np.random.choice(len(X_validation), len(X_validation), replace=False)
            
            X_training = X_training[random_idx_train]
            y_training = y_training[random_idx_train]
            
            X_validation = X_validation[random_idx_validation]
            y_validation = y_validation[random_idx_validation]
            
            K.clear_session()
            np.random.seed(4)
            
            model1day = Sequential()
            model1day.add(LSTM(cell, #numero de neuronas lstm
                        activation="tanh",
                        return_sequences = False,
                        input_shape=(n_steps, n_features,)
                               )
                         )
            model1day.add(Dropout(0.2))
            #model1day.add(LSTM(cell))
            #model1day.add(Dropout(0.2))
            model1day.add(Dense(10))
            model1day.add(Dense(1))
            model1day.compile(optimizer = Adam(learning_rate = 0.001),loss = "mse")
            
            model1day.fit(X_training,y_training,
                      epochs = 100,
                      verbose = 0,
                      batch_size = 128,
                      validation_data = (X_validation,y_validation)
                      )
            history = pd.DataFrame(model1day.history.history)
            history.plot()
            plt.show()
            model1day.save("model1day2.h5")
            
            
            #In[]: FORECAST AND LOSSES
            symbols = ['model 1day']
            
            
            #f_training    = model1day.predict(X_training1)
            #f_validation  = model1day.predict(X_validation1) 
            
            model1day = load_model("model1day2.h5")
            f_forecast1day    = pd.DataFrame(model1day.predict(X_test,steps = 1),columns = symbols)
            
            normal_forecast1day    = pd.DataFrame(sc_target.inverse_transform(f_forecast1day), columns = symbols , index = index_test_fecha_1day)
            #normal_training    = pd.DataFrame(sc_target.inverse_transform(f_training), columns = symbols, index = index_train)
            normal_target1day     = pd.DataFrame(sc_target.inverse_transform(y_test), columns = symbols , index = index_test_fecha_1day)
            
            #pd.set_option('display.float_format', lambda x: '%.10f' % x)
            mse         = ((normal_target1day - normal_forecast1day)**2).mean()
            mae         = (abs(normal_target1day - normal_forecast1day)).mean()
            rmse        = (mse)**0.5
            mape        = (abs(1 - normal_target1day/normal_forecast1day)).mean()
            
            loss         = pd.concat([mse,mae,rmse,mape],axis = 1)
            loss.columns = ["MSE","MAE","RMSE","MAPE"]
            print(loss)
            
            
            graph = pd.concat([normal_forecast1day,normal_target1day], axis = 1)
            graph.columns = ["forecast", "real"]
            graph.plot()
            
            
            index_train2 = target60min.index[ : train_cut_60min]
            index_val2   = target60min.index[train_cut_60min : train_cut_60min + val_cut_60min]
            index_test2  = target60min.index[train_cut_60min + val_cut_60min + steps : ]
            
            X2 = split_features(features60min, steps)
            y2 = split_target(target60min, steps)
            n_steps2,n_features2 = X2.shape[1:]
            
            #SET TRAIN, VAL, TEST DATA 
            X_training2   = X2[:train_cut_60min][:][:]
            X_validation2 = X2[train_cut_60min : train_cut_60min + val_cut_60min][:][:]
            X_test2       = X2[train_cut_60min + val_cut_60min : ][:][:]
            
            y_training2   = y2[:train_cut_60min][:][:]
            y_validation2 = y2[train_cut_60min : train_cut_60min + val_cut_60min][:][:]
            y_test2       = y2[train_cut_60min + val_cut_60min : ][:][:]
            
            X_training3   = X_training2
            X_validation3 = X_validation2
            
            # RANDOMIZE DATA FOR BETTER PERFORMANC
            np.random.seed(4)
            random_idx_train2 = np.random.choice(len(X_training2), len(X_training2), replace=False)
            
            np.random.seed(4)
            random_idx_validation2 = np.random.choice(len(X_validation2), len(X_validation2), replace=False)
            
            X_training2 = X_training2[random_idx_train2]
            y_training2 = y_training2[random_idx_train2]
            
            X_validation2 = X_validation2[random_idx_validation2]
            y_validation2 = y_validation2[random_idx_validation2]
            
            K.clear_session()
            np.random.seed(4)
            
            model1day60min = load_model("model1day2.h5")
            

            model1day60min.fit(X_training2,y_training2,
                      epochs = 100,
                      verbose = 0,
                      batch_size = 128,
                      validation_data = (X_validation2,y_validation2)
                      )
            
            pd.DataFrame(model1day60min.history.history).plot()
            
            model1day60min.save("model1day60min2.h5")
            
            
            symbols = ['model 1day + 60min']
            
            #f_training2    = model1day60min.predict(X_training1)
            #f_validation2  = model1day60min.predict(X_validation1) 
            model1day60min2 = load_model("model1day60min2.h5")
            f_forecast2    = pd.DataFrame(model1day60min2.predict(X_test,steps = 1),columns = symbols)
            
            normal_forecast2    = pd.DataFrame(sc_target.inverse_transform(f_forecast2), columns = symbols , index = index_test_fecha_1day)
            #normal_training    = pd.DataFrame(sc_target.inverse_transform(f_training), columns = symbols, index = index_train)
            normal_target2     = pd.DataFrame(sc_target.inverse_transform(y_test), columns = symbols , index = index_test_fecha_1day)
            
            #pd.set_option('display.float_format', lambda x: '%.10f' % x)
            mse2        = ((normal_target2 - normal_forecast2)**2).mean()
            mae2         = (abs(normal_target2 - normal_forecast2)).mean()
            rmse2        = (mse2)**0.5
            mape2        = (abs(1 - normal_target2/normal_forecast2)).mean()
            
            loss2         = pd.concat([mse2,mae2,rmse2,mape2],axis = 1)
            loss2.columns = ["MSE","MAE","RMSE","MAPE"]
            print(loss2)
            
            
            graph2 = pd.concat([normal_forecast1day,normal_forecast2,normal_target2],axis = 1)
            graph2.columns = ["forecast 1day","forecast 1day + 60 min", "real"]
            graph2.plot()
            
            index_train4 = target30min.index[ : train_cut_30min]
            index_val4   = target30min.index[train_cut_30min : train_cut_30min + val_cut_30min]
            index_test4  = target30min.index[train_cut_30min + val_cut_30min + steps : ]
            
            X4 = split_features(features30min, steps)
            y4 = split_target(target30min, steps)
            n_steps4,n_features4 = X4.shape[1:]
            
            #SET TRAIN, VAL, TEST DATA 
            X_training4   = X4[:train_cut_30min][:][:]
            X_validation4 = X4[train_cut_30min : train_cut_30min + val_cut_30min][:][:]
            X_test4       = X4[train_cut_30min + val_cut_30min : ][:][:]
            
            y_training4   = y4[:train_cut_30min][:][:]
            y_validation4 = y4[train_cut_30min : train_cut_30min + val_cut_30min][:][:]
            y_test4       = y4[train_cut_30min + val_cut_30min : ][:][:]
            
            X_training5   = X_training4
            X_validation5 = X_validation4
            
            # RANDOMIZE DATA FOR BETTER PERFORMANC
            np.random.seed(4)
            random_idx_train4 = np.random.choice(len(X_training4), len(X_training4), replace=False)
            
            np.random.seed(4)
            random_idx_validation4 = np.random.choice(len(X_validation4), len(X_validation4), replace=False)
            
            X_training4 = X_training4[random_idx_train4]
            y_training4 = y_training4[random_idx_train4]
            
            X_validation4 = X_validation4[random_idx_validation4]
            y_validation4 = y_validation4[random_idx_validation4]
            
            K.clear_session()
            np.random.seed(4)
            
            model1day30min = load_model("model1day2.h5")
            
           
            model1day30min.fit(X_training4,y_training4,
                      epochs = 100,
                      verbose = 0,
                      batch_size = 128,
                      validation_data = (X_validation4,y_validation4)
                      )
            
            
            pd.DataFrame(model1day30min.history.history).plot()
            
            
            
            model1day30min.save("model1day30min.h5")
            
            
            #In[]: FORECAST AND LOSSES
            symbols = ['model 1day + 30min']
            
            #f_training2    = model1day60min.predict(X_training1)
            #f_validation2  = model1day60min.predict(X_validation1) 
            model1day30min = load_model("model1day30min.h5")
            f_forecast4    = pd.DataFrame(model1day30min.predict(X_test,steps = 1),columns = symbols)
            
            normal_forecast4    = pd.DataFrame(sc_target.inverse_transform(f_forecast4), columns = symbols , index = index_test_fecha_1day)
            #normal_training    = pd.DataFrame(sc_target.inverse_transform(f_training), columns = symbols, index = index_train)
            normal_target4     = pd.DataFrame(sc_target.inverse_transform(y_test), columns = symbols , index = index_test_fecha_1day)
            
            #pd.set_option('display.float_format', lambda x: '%.10f' % x)
            mse4        = ((normal_target4 - normal_forecast4)**2).mean()
            mae4         = (abs(normal_target4 - normal_forecast4)).mean()
            rmse4        = (mse4)**0.5
            mape4        = (abs(1 - normal_target4/normal_forecast4)).mean()
            
            loss4         = pd.concat([mse4,mae4,rmse4,mape4],axis = 1)
            loss4.columns = ["MSE","MAE","RMSE","MAPE"]
            print(loss4)
            
            
            index_train6 = target15min.index[ : train_cut_15min]
            index_val6   = target15min.index[train_cut_15min : train_cut_15min + val_cut_15min]
            index_test6  = target15min.index[train_cut_15min + val_cut_15min + steps : ]
            
            X6 = split_features(features15min, steps)
            y6 = split_target(target15min, steps)
            n_steps6,n_features6 = X6.shape[1:]
            
            #SET TRAIN, VAL, TEST DATA 
            X_training6   = X6[:train_cut_15min][:][:]
            X_validation6 = X6[train_cut_15min : train_cut_15min + val_cut_15min][:][:]
            X_test6       = X6[train_cut_15min + val_cut_15min : ][:][:]
            
            y_training6   = y6[:train_cut_15min][:][:]
            y_validation6 = y6[train_cut_15min : train_cut_15min + val_cut_15min][:][:]
            y_test6       = y6[train_cut_15min + val_cut_15min : ][:][:]
            
            X_training7   = X_training6
            X_validation7 = X_validation6
            
            # RANDOMIZE DATA FOR BETTER PERFORMANC
            np.random.seed(4)
            random_idx_train6 = np.random.choice(len(X_training6), len(X_training6), replace=False)
            
            np.random.seed(4)
            random_idx_validation6 = np.random.choice(len(X_validation6), len(X_validation6), replace=False)
            
            X_training6 = X_training6[random_idx_train6]
            y_training6 = y_training6[random_idx_train6]
            
            X_validation6 = X_validation6[random_idx_validation6]
            y_validation6 = y_validation6[random_idx_validation6]
            
            K.clear_session()
            np.random.seed(4)
            
            model1day15min = load_model("model1day2.h5")
            
  
            model1day15min.fit(X_training6,y_training6,
                      epochs = 100,
                      verbose = 0,
                      batch_size = 128,
                      validation_data = (X_validation6,y_validation6)
                      )
            
            
            pd.DataFrame(model1day15min.history.history).plot()
            
            
            model1day15min.save("model1day15min.h5")
            
            
            #In[]: FORECAST AND LOSSES
            #symbols_15min = ['model 1day + 15min 1Capa '+ str(cell) +' Celdas ' + str(steps) + 'steps']
            symbols = ['model 1day + 15min']
                       
            #f_training2    = model1day60min.predict(X_training1)
            #f_validation2  = model1day60min.predict(X_validation1) 
            model1day15min = load_model("model1day15min.h5")
            f_forecast6    = pd.DataFrame(model1day15min.predict(X_test,steps = 1),columns = symbols)
            
            normal_forecast6    = pd.DataFrame(sc_target.inverse_transform(f_forecast6), columns = symbols , index = index_test_fecha_1day)
            #normal_training    = pd.DataFrame(sc_target.inverse_transform(f_training), columns = symbols, index = index_train)
            normal_target6      = pd.DataFrame(sc_target.inverse_transform(y_test), columns = symbols , index = index_test_fecha_1day)
            
            #pd.set_option('display.float_format', lambda x: '%.10f' % x)
            mse6        = ((normal_target6 - normal_forecast6)**2).mean()
            mae6         = (abs(normal_target6- normal_forecast6)).mean()
            rmse6        = (mse6)**0.5
            mape6        = (abs(1 - normal_target6/normal_forecast6)).mean()
            
            loss6         = pd.concat([mse6,mae6,rmse6,mape6],axis = 1)
            loss6.columns = ["MSE","MAE","RMSE","MAPE"]
            print(loss6)
            
            
            graph = pd.concat([normal_target2,normal_forecast1day,normal_forecast2,normal_forecast4,normal_forecast6],axis = 1)
            graph.columns = ["Real","Forecast 1day","Forecast 1day + 60min","Forecast 1day + 30min" , "Forecast 1day + 15min"]
            graph.plot()
            
            graph.to_excel("1CAPA-forecast"+nombre+"-"+str(steps)+"-"+ str(cell)+".xlsx")
            
            
            plt.style.use('seaborn')
            LOSS = pd.concat([loss,loss2,loss4,loss6])
            
            print(LOSS)
            
            LOSS2 = pd.concat([loss2,loss4,loss6])
            loss_aux = pd.concat([loss,loss,loss])
            loss_aux.index = ["model 60min","model 30min","model 15min"]
            change = (LOSS2/loss_aux-1)
            change.plot.barh()
            
            LOSS.to_excel("losses"+nombre+"-"+str(steps)+"-"+ str(cell) +".xlsx")
            
            names_losses = ['model 1day        : 1 Capa, '+ str(cell) +' Celdas, ' + str(steps) + 'steps',
                            'model 1day + 60min: 1 Capa, '+ str(cell) +' Celdas, ' + str(steps) + 'steps',
                            'model 1day + 30min: 1 Capa, '+ str(cell) +' Celdas, ' + str(steps) + 'steps',
                            'model 1day + 15min: 1 Capa, '+ str(cell) +' Celdas, ' + str(steps) + 'steps']
            
            LOSS_MSE = pd.DataFrame()
            LOSS_MSE = pd.concat([loss.MSE,loss2.MSE,loss4.MSE,loss6.MSE])
            LOSS_MSE.index = names_losses
            BIG_LOSS_MSE = pd.concat([BIG_LOSS_MSE,LOSS_MSE])
            
            
            LOSS_MAE = pd.DataFrame()
            LOSS_MAE = pd.concat([loss.MAE,loss2.MAE,loss4.MAE,loss6.MAE])
            LOSS_MAE.index = names_losses
            BIG_LOSS_MAE = pd.concat([BIG_LOSS_MAE,LOSS_MAE])
            
            
            LOSS_RMSE = pd.DataFrame()
            LOSS_RMSE = pd.concat([loss.RMSE,loss2.RMSE,loss4.RMSE,loss6.RMSE])
            LOSS_RMSE.index = names_losses
            BIG_LOSS_RMSE = pd.concat([BIG_LOSS_RMSE,LOSS_RMSE])
            
            
            LOSS_MAPE = pd.DataFrame()
            LOSS_MAPE = pd.concat([loss.MAPE,loss2.MAPE,loss4.MAPE,loss6.MAPE])
            LOSS_MAPE.index = names_losses
            BIG_LOSS_MAPE = pd.concat([BIG_LOSS_MAPE,LOSS_MAPE])
            
    #BIG_LOSS_MSE.to_excel("MSE-" + nombre + "-1CAPA.xlsx")
    #BIG_LOSS_MAE.to_excel("MAE-" + nombre + "-1CAPA.xlsx")
    #BIG_LOSS_RMSE.to_excel("RMSE-" + nombre + "-1CAPA.xlsx")
    #BIG_LOSS_MAPE.to_excel("MAPE-" + nombre + "-1CAPA.xlsx")
        
    DA_BIG_LOSS = pd.concat([BIG_LOSS_MSE,BIG_LOSS_MAE,BIG_LOSS_RMSE,BIG_LOSS_MAPE],axis = 1)
    DA_BIG_LOSS.columns = ["MSE","MAE","RMSE","MAPE"]
    DA_BIG_LOSS.to_excel("BIG LOSS 1 CAPA -" + nombre +".xlsx")
    
    print(DA_BIG_LOSS)            
            
