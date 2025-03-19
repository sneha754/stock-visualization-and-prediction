import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.svm import SVR

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

def get_more_price_fig(df):
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date", markers=True)
    fig.update_layout(title_x=0.5)
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode="lines+markers")
    return fig

# Navigation component
item1 = html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),
        html.P("Input the Stock Code", className="name"),
        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code', className="stock-input"),
            html.Button('Submit', id='submit-button')
        ], className="stock-input-1"),
        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            # Stock price button
            html.Button('Get Stock Price', id='stock-price-button'),
            # Indicators button
            html.Button('Get Indicators', id='indicators-button'),
            # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),
            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
        ], className="selectors")
    ],
    className="nav"
)

# Content component
item2 = html.Div(
    [
        html.Div(
            [
                html.Img(id='logo', className='logo'),
                html.H1(id='company-name', className='company-name')
            ],
            className="header"),
        html.Div(id="description"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[item1, item2])

# Callbacks

# Callback to update the data based on the submitted stock code
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("company-name", "children"),
        Output("stock-price-button", "n_clicks"),
        Output("indicators-button", "n_clicks"),
        Output("forecast-button", "n_clicks")
    ],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):
    if n is None or val is None:
        raise PreventUpdate

    ticker = yf.Ticker(val)
    inf = ticker.info

    # Default fallback values
    description = "No data available"
    logo_url = ""
    name = "Unknown Company"

    print(inf)  # Debugging: Print the info dictionary

    if 'longBusinessSummary' in inf:
        description = inf['longBusinessSummary']
    if 'logo_url' in inf:
        logo_url = inf['logo_url']
    if 'longName' in inf:
        name = inf['longName']

    return description, logo_url, name, None, None, None

# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        return [""]
    try:
        if start_date is not None:
            df = yf.download(val, start=start_date, end=end_date)
        else:
            df = yf.download(val)
    except Exception as e:
        return [html.Div(f"Failed to download data: {str(e)}")]

    df.reset_index(inplace=True)
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=1200)  # Adjust the width of the graph here
    return [dcc.Graph(figure=fig, style={'width': '100%'})]

# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        return [""]
    try:
        if start_date is None:
            df_more = yf.download(val)
        else:
            df_more = yf.download(val, start=start_date, end=end_date)
    except Exception as e:
        return [html.Div(f"Failed to download data: {str(e)}")]

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    fig.update_layout(width=1200)  # Adjust the width of the graph here
    return [dcc.Graph(figure=fig, style={'width': '100%'})]

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

def prediction(stock_code, days):
    # Fetch historical data for the stock
    ticker = yf.Ticker(stock_code)
    df = ticker.history(period="1y")  # Fetch data for the past year

    # Ensure we have enough data points
    if df.shape[0] < days:
        raise ValueError("Not enough data points to make predictions")

    # Prepare the data
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateOrdinal'] = df['Date'].map(dt.toordinal)
    X = df[['DateOrdinal']].values
    y = df['Close'].values

    # Create the model
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    
    # Split the data into train and test sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Fit the model
    svr.fit(X_train, y_train)

    # Make predictions
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    future_predictions = svr.predict(future_dates_ordinal)

    # Create a dataframe for the future predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_predictions
    })

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Close'], mode='lines', name='Predicted Close'))
    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Close Price', width=1200)

    return fig

# Callback for displaying forecast
@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if n is None or n_days is None or val is None:
        raise PreventUpdate
    try:
        fig = prediction(val, int(n_days))
        return [dcc.Graph(figure=fig, style={'width': '100%'})]
    except Exception as e:
        return [html.Div(f"Failed to generate forecast: {str(e)}")]

if __name__ == '__main__':
    app.run_server(debug=True)
