import os

import keras
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from plotly.offline import plot
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Network:
    def _load_n_preprocess(self):
        filename = "female-birth.csv"
        self.df = pd.read_csv(filename)
        self.df = self.df[:-1]

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.rename(columns={"Daily total female births in California, 1959":"Birth"}, inplace=True)
        print("\nINFO: Preprocessing Complete.")

    def _create_generator(self):
        val = self.df['Birth'].values
        val = val.reshape((-1,1))

        split = int(0.8*len(val))

        self.date_train =self. df['Date'][:split]
        self.date_test = self.df['Date'][split:]

        self.birth_train = self.df['Birth'].values[:split]
        self.birth_test = self.df['Birth'].values[split:]

        self.train = val[:split]
        self.test = val[split:]
        self.train_generator = TimeseriesGenerator(self.train, self.train, length=3, batch_size=5)
        self.test_generator = TimeseriesGenerator(self.test, self.test, length=3, batch_size=1)
        print("INFO: Dataset Generated.")

    def _create_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(4,
                activation='relu',
                input_shape=(3,1), return_sequences=False)
        )
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        print("INFO: Model Created")

    def __init__(self):
        self._load_n_preprocess()
        self._create_generator()
        self._create_model()
        self.train_model()
        self.message = ""

    def train_model(self):
        num_epochs = 15
        self.model.fit_generator(self.train_generator, epochs=num_epochs)
        print("INFO: Model trained for {} epochs".format(num_epochs))

    def generate_init_plot(self):
        self.static_predict = self.model.predict_generator(self.test_generator).reshape((-1)).astype('int64')
        print(self.static_predict)
        
        trace1 = go.Scatter(
                    x = self.date_train,
                    y = self.birth_train,
                    mode = 'lines',
                    name = 'Data'
                )
        trace2 = go.Scatter(
                    x = self.date_test,
                    y = self.static_predict,
                    mode = 'lines',
                    name = 'Traditional Forecast'
                )
        layout = go.Layout(
            title = "Traditional Forecasting Model",
            xaxis = {'title':'Date'},
            yaxis = {'title':'Sales'}
        )
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        return plot(fig, output_type='div')

    def get_vals_for_day(self, x):
        x = pd.to_datetime(x, yearfirst=True)
        x_3 = pd.date_range(end=x, periods=4)[:-1]
        val = self.df.loc[self.df['Date'].isin(x_3), 'Birth']
        return val

    def predict(self, x):
    #     x is given date. So (x-2,x-1,x) needs to be found
        print(x)
        val = self.get_vals_for_day(x)
        val = val.values.reshape((1,3,1))
        y = self.model.predict(val)[0][0]
        return y.astype('int64')
    
    def retrain_model(self, date_str, y):
#     x : Date for model retraining
#     y : Corresponding new value
        x = self.get_vals_for_day(date_str)
        x = x.values.reshape((1,3,1))
        y = np.array(y).reshape(1)
        self.model.fit(x, y, epochs=15, verbose=0)

# Regenerate Plot
        dynamic_predict = self.model.predict_generator(self.test_generator).reshape((-1)).astype('int64') 

        trace1 = go.Scatter(
                    x = self.date_train,
                    y = self.birth_train,
                    mode = 'lines',
                    name = 'Data'
                )
        trace2 = go.Scatter(
            x = self.date_test,
            y = dynamic_predict,
            mode = 'lines',
            name = 'Prophet Forecast'
        )
        trace3 = go.Scatter(
            x = self.date_test,
            y = self.static_predict,
            mode = 'lines',
            name = 'Traditional Forecast'
        )
        layout = go.Layout(
            title = 'Prophet Forecasting Model',
            xaxis = {'title':'Date'},
            yaxis = {'title':'Sales'},
            annotations = [dict(
                x = pd.to_datetime(date_str, yearfirst=True),
                y = y,
                xref = 'x',
                yref = 'y',
                text = 'Target',
                showarrow=True,
                font=dict(
                    family='Courier New, monospace',
                    size=16,
                    color='#ffffff'
                ),
                align='center',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363',
                ax=20,
                ay=-30,
                bordercolor='#c7c7c7',
                borderwidth=2,
                borderpad=4,
                bgcolor='#ff7f0e',
                opacity=0.8                
                )]
        )
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        return plot(fig, output_type='div')
