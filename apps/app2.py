import dash_core_components as dcc
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
import pandas as pd
import dash_html_components as html
import plotly.express as px

from app import app
terror = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/modified_globalterrorismdb_0718dist.csv')
terr2 = pd.read_csv('/Users/shubhangigupta/Desktop/Data Visualization and communication/Project/2-Global terrorism database dashboard in python by plotly dash/Datasets/Modifies_2.csv')
list = terr2.region_txt.unique()
#app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
layout = html.Div([
html.Div([
        html.Div([
            html.P('Select Region', className='fix_label', style= {'color': 'white'}),
            dcc.Dropdown(id = 'w_countries',
                         multi = False,
                         searchable= True,
                         value='South Asia',
                         placeholder= 'Select Region',
                         options= [{'label': c, 'value': c}
                                   for c in (terror['region_txt'].unique())], className='dcc_compon')
            ], className='create_container three columns'),

        html.Div([
            dcc.Graph(id = 'Sunburst_graph', config={'displayModeBar': 'hover'}),
        ], className='create_container six columns'),
        html.Div([
            dcc.Graph(id = 'Vilon', config={'displayModeBar': 'hover'})
        ], className='create_container nine columns')
], className='row flex-display'),
    html.Div([
        html.Div([
            dcc.Graph(id = 'heatmap', config={'displayModeBar': 'hover'})
        ], className='create_container six columns'),
    html.Div([
        dcc.Graph(id='area_map', config={'displayModeBar': 'hover'})
    ], className='create_container five columns'),
]),
], id = 'mainContainer', style={'display': 'flex', 'flex-direction': 'column', 'backgroundColor':'black'})


@app.callback(Output('Sunburst_graph', 'figure'),
              [Input('w_countries','value')],
              )
def display_sunburst(w_countries):
   # terr5 = terr2.groupby(['region_txt', 'attacktype1_txt', 'targtype1_txt'])
    terr2.fillna(method='ffill')
    print(terr2)
    terr6 = terr2[(terr2['region_txt'] == w_countries)]
    fig = px.sunburst(terr6, path=['region_txt', 'attacktype1_txt', 'targtype1_txt'], values='nkill')
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.layout.plot_bgcolor = '#010915'
    fig.layout.paper_bgcolor = '#010915'
    fig.update_layout(
        font_color="white",
        font_size=10,
        title_font_color="white",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"
    )
)
    return fig

@app.callback(Output('Vilon', 'figure'),
              [Input('w_countries','value')],
              )
def display_vilon(w_countries):
    terr7 = terr2[(terr2['region_txt'] == w_countries)]
    terr7['gname'] = terr7['gname'].str.split(' ').str[0]
    terr8 = terr7.groupby(by=['gname', 'region_txt'], as_index=False)['nkill'].sum().sort_values(by='nkill',
                                                                                            ascending=False).head(20)
    # pd.date_range(end = datetime.today(), periods = 100).to_pydatetime().tolist()

    fig = px.scatter(terr8, x="gname", y="nkill",
                     marginal_x="box", marginal_y="violin", color = 'region_txt',
                     title="Click on the legend items!")
    fig.layout.plot_bgcolor = '#010915'
    fig.update_traces(marker_size=10)
    fig.layout.paper_bgcolor = '#010915'
    fig.update_traces(marker_color='orange')
    fig.update_layout(
        font_color="white",
        font_size= 10,
        title_font_color="white",
        legend_title_font_color="green",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"
        )
    )
    return fig


@app.callback(Output('heatmap', 'figure'),
              [Input('w_countries', 'value')]
              )
def update_graph(w_countries):
    terr11 = terr2[(terr2['region_txt'] == w_countries)]
    terr11['weaptype1_txt'] = terr11['weaptype1_txt'].str.split(' ').str[0]
    terr12 = terr11.groupby([ 'weaptype1_txt','region_txt']).size().unstack(fill_value=0)
    fig = px.bar(terr12,color_continuous_scale=px.colors.sequential.Plasma,
                    title="correlation between region and weapon type")


    fig.update_layout(title_font={'size':20}, title_x=0.5)
    fig.update_traces(hovertemplate="Membership bought: %{y}"
                                    "<br>Weeks subscribed: %{x}"
                                    "<br>Cancellations: %{z}<extra></extra>")

    fig.layout.plot_bgcolor = '#010915'
    fig.layout.paper_bgcolor = '#010915'
    fig.update_layout(
        font_color="red",
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
@app.callback(Output('area_map', 'figure'),
              [Input('w_countries', 'value')]
              )
def update_graph(w_countries):
    terr15 = terror[(terror['region_txt'] == w_countries)]
    terr16 = terr15[['region_txt','iyear','attacktype1']]
    #terr4 = terr3.region_txt.value_counts().reset_index(name='Sum of attacks')
    terr17 = terr16.groupby(['region_txt','iyear']).count().sort_values(["attacktype1"], ascending=False).rename(columns={"accident" : "attacktype1"}).reset_index()

    # hist_data = [terr17]
    # group_labels = ['distplot']
    x = terr17['iyear']
    y = terr17['attacktype1']
    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)']
    fig = ff.create_2d_density( x, y, colorscale=colorscale,
      hist_color='rgb(255, 237, 222)')

    fig.layout.plot_bgcolor = '#010915'
    fig.layout.paper_bgcolor = '#010915'
    # fig = px.area(terr17, x="iyear", y="attacktype1", color="region_txt",
    #           line_group="region_txt")
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Attack_count',
        title='Number of Attacks w.r.t Years',
        font=dict(
            family="sans-serif",
            size=12,
            color="white"
        )
    )

    return fig


@app.callback(
    Output('textarea-example-output', 'children'),
    Input('textarea-example', 'value')
)
def update_output(w_countries):
    terr7 = terr2[(terr2['region_txt'] == w_countries)]
    terr7['gname'] = terr7['gname'].str.split(' ').str[0]
    terr8 = terr7.groupby(by=['gname', 'region_txt'], as_index=False)['nkill'].sum().sort_values(by='nkill',
                                                                                  ascending=False).head(20)

    value1 = terr8['gname'].head(n=1)

    return 'You have entered: \n{}'.format(value1)



