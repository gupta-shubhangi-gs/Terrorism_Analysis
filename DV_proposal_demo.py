from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import dash
# from dash import dcc
# from dash import html
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

# Removing outliers
terror_df = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/globalterrorismdb_0718dist.csv', encoding ='latin1')
terror_df = terror_df[terror_df['nkill'] <= 4].reset_index(drop=True)
terror_df = terror_df[terror_df['nwound'] <= 7].reset_index(drop=True)
corr_df = terror_df[['region', 'attacktype1', 'targtype1']].corr()

# Selecting features with high variance
features = [
    'longitude',
    'latitude',
    
    'nwound',
    'nkill',
    
    'natlty1_txt',
    'targtype1_txt',
    'targsubtype1_txt',
    'weaptype1_txt',
    'attacktype1_txt',
]

X = pd.get_dummies(terror_df[features])
X = X.T[X.var() > 0.05].T.fillna(0)
X = X.fillna(0)

app = dash.Dash(__name__)

graph_style = {'width':'100%', 'display':'inline-block'}

scatters = ['Default', 'K-Means']

app.layout = html.Div([
    html.H1('A Visualization of Global Terrorism Incidents (1970-2017)'),
    html.Div([
        html.P('Plotting Options:'),
        dcc.Dropdown(
            id = 'dropdown',
            options = [{'label': i, 'value': i} for i in scatters],
            value = 'Default'
        ),
        dcc.Graph(id ='scatter', style = graph_style)
    ], style = {'width':'50%', 'display':'inline-block'}),

    html.Div([
        html.P('Features to Correlate:'),
        dcc.Checklist(
            id = 'checklist',
            options = [{'label': x, 'value': x} for x in corr_df.columns],
            value = corr_df.columns.tolist(),
            labelStyle={'display': 'flex'}
        ),
        dcc.Graph(id ='heatmap', style = graph_style)
    ], style = {'width':'50%', 'display':'inline-block'})

], style = graph_style)



@app.callback(
    Output('scatter', 'figure'),
    [Input('dropdown', 'value')]
)
def display_plots(name):
    if name == 'Default':
        fig = display_scatter()
    else:
        fig = display_kmeans()
    return fig

@app.callback(
    Output('heatmap', 'figure'),
    [Input('checklist', 'value')]
)
def display_heatmap(cols):
    fig = px.imshow(corr_df[cols])
    return fig


def display_scatter():
    fig = px.scatter(X, x='longitude', y='latitude', title='Global Terrorism Incidents (1970-2017)')
    return fig

def display_kmeans():
    fig = go.Figure()

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(X)

    # predictions from kmeans

    pred = kmeans.predict(X)
    frame = pd.DataFrame(X)
    frame['cluster'] = pred
    frame = frame[['longitude', 'latitude', 'cluster']]

    frame.head()
    frame.columns = ['longitude', 'latitude', 'cluster']

    clusters_centroids = dict()
    clusters_radii = dict()

    X_arr = X.values
    for k in range(0, 6):
        data = frame[frame["cluster"] == k]
    #
    # frame.head()
    # frame.columns = ['longitude', 'latitude', 'cluster']
    color = ['blue', 'green', 'cyan', 'black', 'magenta', 'yellow', 'red']
    for k in range(0, 6):
        data = frame[frame["cluster"] == k]
        fig.add_trace(go.Scatter(x=data["longitude"], y=data["latitude"], mode='markers'))
        clusters_centroids[k] = list(zip(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]))[k]
        fig.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                                 mode='markers', marker_symbol='x', marker_line_color='darkred',
                                 marker_color='orangered'))
        clusters_radii[k] = max(
            [np.linalg.norm(np.subtract(i, clusters_centroids[k])) for i in zip(X_arr[pred == k, 0], X_arr[pred == k, 1])])
        fig.add_shape(type='circle', line_color='blue',
                      x0=min(data['longitude']), y0=min(data['latitude']),
                      x1=max(data['longitude']), y1=max(data['latitude']))


    fig.update_layout(title='Clusters, Centroids and Radii (K-Means)',
                  xaxis_title='longitude',
                  yaxis_title='latitude',
                  showlegend=False)
    return fig

# app.run_server(debug=True, host='0.0.0.0', port=8086, threaded=True)
app.run_server(debug=False)
