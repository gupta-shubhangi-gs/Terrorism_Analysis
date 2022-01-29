from typing import List

import matplotlib

matplotlib.use('Agg')

import dash.dependencies as dd

from io import BytesIO

import base64
from wordcloud import WordCloud
from nltk.corpus import stopwords
from app import app
stopwords = set(stopwords.words('english'))
import re
import string

from sklearn.cluster import KMeans

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import matplotlib.pyplot as plt

fig = plt.figure()
# Removing outliers

from scipy.stats import t
import matplotlib.pyplot as plt

plt.style.use('seaborn')
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats

terr21 = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/2015.csv')
#terr8 = terr21.groupby(by=['Region'], as_index=False)


test = ['Latin America,Caribbean and Eastern Asia',
          'North America and Middle East',
          'Latin America and Caribbean and Southern Asia']
terror_df = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/globalterrorismdb_0718dist.csv',
    encoding='latin1')
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

graph_style = {'width': '100%', 'display': 'inline-block'}

scatters = ['Default', 'kmeans_gmm']

LR = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/modified_globalterrorismdb_0718dist.csv')
LR1 = LR.groupby(['iyear'])['nkill'].sum().reset_index()
X1 = LR1['iyear'].unique()
X2 = np.reshape(X1, (47,-1))
y1 = LR1['nkill']

X_train, X_test, y_train, y_test = train_test_split(X2, y1, random_state=42)
terr2 = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/Modified3.csv')
text1: List[str] = []
count = {}
text2 = []
terr3 = terr2['motive'].fillna('').apply(str)
articles = {'a': '', 'an':'', 'and':'', 'the':'', 'in':''}

for line in terr3.head(100):
    line = line.lower()
    p = re.compile("[" + re.escape(string.punctuation) + "]")
    text1 = p.sub("", line)
    my_list = text1.split()
    word_freq = [my_list.count(i) for i in my_list]
    count = dict(zip(my_list, word_freq))
keys = count.values()
values = count.keys()
df = pd.DataFrame({'word':values,'freq':keys})



print(df)
dfm = pd.DataFrame(df)
print(dfm)
dfm = dfm[~dfm['word'].isin(['The'])]
dfm = dfm[~dfm['word'].isin(['the'])]
dfm = dfm[~dfm['word'].isin(['and'])]
dfm = dfm[~dfm['word'].isin(['in'])]
# dfm = pd.DataFrame({'word': ['apple', 'pear', 'orange'], 'freq': [1,3,9]})
# print(dfm)
models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}
layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='model-name',
                options=[{'label': x, 'value': x}
                         for x in models],
                value='Regression',
                clearable=False
            ),
            dcc.Graph(id="graph"),
            ], className='create_container five columns'),
   html.Div([
        dcc.Dropdown(
            id = 'dropdown',
            options = [{'label': i, 'value': i} for i in scatters],
            value = 'Default'
        ),
        dcc.Graph(id ='scatter', style = graph_style)
    ], className='create_container six columns'),
    ]),

html.Div([
html.Div([
        html.Img(id="image_wc"),
    ], className='create_container five columns'),
html.Div([
    dcc.Dropdown(
        id='hypothesis',
        options = [{'label': i, 'value': i} for i in test],
        value='Latin America,Caribbean and Eastern Asia',
        clearable=False
    ),
    dcc.Graph(id ='test', style = graph_style),

    ], className='create_container six columns'),


    ]),


], id = 'mainContainer', style={'display': 'flex', 'flex-direction': 'column', 'backgroundColor':'black'})


def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='black', width=700, height=500)
    wc.fit_words(d)
    return wc.to_image()

@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(data=dfm).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(
    Output("graph", "figure"),
    [Input('model-name', "value")])
def Linear_Regrssion(name):
    model = models[name]()
    model.fit(X_train, y_train)
    x_range = np.linspace(X2.min(), X2.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')

    ])

    fig.layout.plot_bgcolor = '#010915'
    fig.layout.paper_bgcolor = '#010915'

    fig.update_layout(
        font_color="white",
        title= "Regression and Prediction(Number of kills w.r.t years)",
        xaxis_title="Years",
        yaxis_title="Number Of Kills",
        font_size=10,
        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"
        )
    )

    return fig
# def confusion_matrx(name):
#     model = models[name]()
#     model.fit(X_train, y_train)



@app.callback(
    Output('scatter', 'figure'),
    [Input('dropdown', 'value')]
)
def display_plots(name):
    if name == 'Default':
        fig = display_scatter()
    elif name == 'kmeans_gmm':
        fig = display_kmeans_gmm()
    return fig
def display_scatter():
    fig = px.scatter(X, x='longitude', y='latitude', title='Scatter and Clustering w.r.t Attacks')
    fig.layout.paper_bgcolor = '#010915'
    fig.layout.plot_bgcolor = '#010915'
    fig.update_layout(
        font_color="white",
        font_size=10,

        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"))
    return fig

def display_kmeans_gmm():
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

        fig.layout.paper_bgcolor = '#010915'
        fig.layout.plot_bgcolor = '#010915'
        fig.update_layout(title='Clusters, Centroids and Radii (K-Means)',
                      xaxis_title='longitude',
                      yaxis_title='latitude',
                      showlegend=False,
                      font_color="white",
                      font_size=10,
                      title_font_color="white",
                      legend_title_font_color="green",
                      font=dict(
                            family="sans-serif",
                            size=12,
                            color="white"
            )
        )

    return fig

@app.callback(
    Output('test', 'figure'),
    [Input('hypothesis', 'value')]
)
def display_graphs(name):
    if name == 'Latin America,Caribbean and Eastern Asia':
        fig = display_2000()
    elif name == 'North America and Middle East':
        fig = display_first_and_last_10_decayes()
    elif name == 'Latin America and Caribbean and Southern Asia':
        fig = display_hypothesis()
    return fig



def display_2000():
    dd1 = terr21.loc[terr21['Region'] == 'Latin America and Caribbean']
    dd2 = terr21.loc[terr21['Region'] == 'Eastern Asia']
    data1 = dd1['Happiness Score']
    data2 = dd2['Happiness Score']

    def degreesOfFreedom(X, Y):
        s1 = np.std(X, ddof=1)  # standard deviation of sample 1
        s2 = np.std(Y, ddof=1)  # standard deviation of sample 2
        s1_sq = (s1 ** 2)  # variance of sample 1
        s2_sq = (s2 ** 2)  # variance of sample 2
        n1 = len(X)  # length of sample 1
        n2 = len(Y)  # length of sample 2
        df = ((s1_sq / n1) + (s2_sq / n2)) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
        return (df)

    print('Degrees of freedom for Student-t distribution: ' + str(degreesOfFreedom(data1, data2)))
    ## Define problem constraints/features

    n = degreesOfFreedom(data1, data2)
    dof = n
    alpha = 0.05
    tails = 2

    # Find critical
    # tcrit = st.t.ppf(alpha/tails, dof)
    # print(f"Critical 2-tail t-scores for alpha={alpha} with {dof} degrees of freedom = +- {tcrit}")

    prob = 1 - alpha
    # retrieve value <= probability
    value = t.ppf(prob, dof)
    print("tcritical=", value)

    stat, p = stats.ttest_ind(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('fail to reject H0')
    else:
        print('reject H0')

    ## Visualize against t distribution
    fig = go.Figure()
    xs = np.linspace(t.ppf(0.001, dof), t.ppf(0.999, dof), 100)
    fig = px.line(x=xs, y=st.t.pdf(xs, dof), labels={'x': 'Tscore', 'y': 'prob'})
    fig.add_vline(x=value, line_dash="dash", line_color="green")
    fig.add_vline(x=-value, line_dash="dash", line_color="green")
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[stat],
            y=[p],
            opacity=0.5,
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
            name='Opacity 0.5'
        )
        )
    fig.layout.paper_bgcolor = '#010915'
    fig.layout.plot_bgcolor = '#010915'
    fig.update_layout(
        font_color="white",
        title = "Hypothesis Testing Between Happiness and Region w.r.t Terrorist Events",
        font_size=10,
        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"))
    return fig

def display_first_and_last_10_decayes():
    dd1 = terr21.loc[terr21['Region'] == 'North America']
    dd2 = terr21.loc[terr21['Region'] == 'Middle East and Northern Africa']
    data1 = dd1['Happiness Score']
    data2 = dd2['Happiness Score']

    def degreesOfFreedom(X, Y):
        s1 = np.std(X, ddof=1)  # standard deviation of sample 1
        s2 = np.std(Y, ddof=1)  # standard deviation of sample 2
        s1_sq = (s1 ** 2)  # variance of sample 1
        s2_sq = (s2 ** 2)  # variance of sample 2
        n1 = len(X)  # length of sample 1
        n2 = len(Y)  # length of sample 2
        df = ((s1_sq / n1) + (s2_sq / n2)) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
        return (df)

    print('Degrees of freedom for Student-t distribution: ' + str(degreesOfFreedom(data1, data2)))
    ## Define problem constraints/features

    n = degreesOfFreedom(data1, data2)
    dof = n
    alpha = 0.05
    tails = 2

    # Find critical
    # tcrit = st.t.ppf(alpha/tails, dof)
    # print(f"Critical 2-tail t-scores for alpha={alpha} with {dof} degrees of freedom = +- {tcrit}")

    prob = 1 - alpha
    # retrieve value <= probability
    value = t.ppf(prob, dof)
    print("tcritical=", value)

    stat, p = stats.ttest_ind(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('fail to reject H0')
    else:
        print('reject H0')

    ## Visualize against t distribution
    fig = go.Figure()
    xs = np.linspace(t.ppf(0.001, dof), t.ppf(0.999, dof), 100)
    fig = px.line(x=xs, y=st.t.pdf(xs, dof), labels={'x': 'Tscore', 'y': 'prob'})
    fig.add_vline(x=value, line_dash="dash", line_color="green")
    fig.add_vline(x=-value, line_dash="dash", line_color="green")
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[stat],
            y=[p],
            opacity=0.5,
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
            name='Opacity 0.5'
        )
    )
    fig.layout.paper_bgcolor = '#010915'
    fig.layout.plot_bgcolor = '#010915'
    fig.update_layout(
        font_color="white",
        font_size=10,
        title="Hypothesis Testing Between Happiness and Region w.r.t Terrorist Events",
        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"))
    return fig

def display_hypothesis():
    dd1 = terr21.loc[terr21['Region'] == 'Latin America and Caribbean']
    dd2 = terr21.loc[terr21['Region'] == 'Southern Asia']
    data1 = dd1['Happiness Score']
    data2 = dd2['Happiness Score']

    def degreesOfFreedom(X, Y):
        s1 = np.std(X, ddof=1)  # standard deviation of sample 1
        s2 = np.std(Y, ddof=1)  # standard deviation of sample 2
        s1_sq = (s1 ** 2)  # variance of sample 1
        s2_sq = (s2 ** 2)  # variance of sample 2
        n1 = len(X)  # length of sample 1
        n2 = len(Y)  # length of sample 2
        df = ((s1_sq / n1) + (s2_sq / n2)) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
        return (df)

    print('Degrees of freedom for Student-t distribution: ' + str(degreesOfFreedom(data1, data2)))
    ## Define problem constraints/features

    n = degreesOfFreedom(data1, data2)
    dof = n
    alpha = 0.05
    tails = 2

    # Find critical
    # tcrit = st.t.ppf(alpha/tails, dof)
    # print(f"Critical 2-tail t-scores for alpha={alpha} with {dof} degrees of freedom = +- {tcrit}")

    prob = 1 - alpha
    # retrieve value <= probability
    value = t.ppf(prob, dof)
    print("tcritical=", value)

    stat, p = stats.ttest_ind(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('fail to reject H0')
    else:
        print('reject H0')

    ## Visualize against t distribution
    fig = go.Figure()
    xs = np.linspace(t.ppf(0.001, dof), t.ppf(0.999, dof), 100)
    fig = px.line(x=xs, y=st.t.pdf(xs, dof), labels={'x': 'Tscore', 'y': 'prob'})
    fig.add_vline(x=value, line_dash="dash", line_color="green")
    fig.add_vline(x=-value, line_dash="dash", line_color="green")
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[stat],
            y=[p],
            opacity=0.5,
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
            name='Opacity 0.5'
        )
        )
    fig.layout.paper_bgcolor = '#010915'
    fig.layout.plot_bgcolor = '#010915'
    fig.update_layout(
        font_color="white",
        font_size=10,
        title="Hypothesis Testing Between Happiness and Region w.r.t Terrorist Events",
        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"))
    return fig

#
#
#
#
#
