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
        filename = "sales_final.csv"
        self.df = pd.read_csv(filename)
        # self.df = self.df[:-1]

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # time_diff = pd.Timedelta(days=18263)
        # self.df['Date'] += time_diff
        print("\nINFO: Preprocessing Complete.")

    def _create_generator(self):
        val = self.df['Sales'].values
        val = val.reshape((-1,1))

        split = int(0.8*len(val))

        self.date_train =self. df['Date'][:split]
        self.date_test = self.df['Date'][split:]

        self.sales_train = self.df['Sales'].values[:split]
        self.sales_test = self.df['Sales'].values[split:]

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
        self.static_predict = self.model.predict_generator(self.test_generator).reshape((-1))
        print(self.static_predict)
        
        trace1 = go.Scatter(
                    x = self.date_train,
                    y = self.sales_train,
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
            yaxis = {'title':'Sales (Million Dollars)'}
        )
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        return plot(fig, output_type='div')

    def get_vals_for_day(self, x):
        x = pd.to_datetime(x, yearfirst=True)
        x_3 = pd.date_range(end=x, periods=4)[:-1]
        val = self.df.loc[self.df['Date'].isin(x_3), 'Sales']
        return val

    def predict(self, x):
    #     x is given date. So (x-2,x-1,x) needs to be found
        val = self.get_vals_for_day(x)
        val = val.values.reshape((1,3,1))
        y = self.model.predict(val)[0][0]
        return y
    
    def retrain_model(self, date_str, inp_y):
#     x : Date for model retraining
#     y : Corresponding new value
        print("Input Date: ",date_str)
        x = self.get_vals_for_day(date_str)
        print("Dates : ", x)
        x = x.values.reshape((1,3,1))
        y = np.array(inp_y).reshape(1)
        self.model.fit(x, y, epochs=15, verbose=0)

# Regenerate Plot
        dynamic_predict = self.model.predict_generator(self.test_generator).reshape((-1)) 

        trace1 = go.Scatter(
                    x = self.date_train,
                    y = self.sales_train,
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
            yaxis = {'title':'Sales (Million Dollars)'},
            annotations = [dict(
                x = pd.to_datetime(date_str, yearfirst=True),
                y = inp_y,
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
                arrowhead=5,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363',
                ax=-0,
                ay=-70,
                bordercolor='#c7c7c7',
                borderwidth=2,
                borderpad=4,
                bgcolor='#ff7f0e',
                opacity=0.8                
                )]
        )
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        return plot(fig, output_type='div')
