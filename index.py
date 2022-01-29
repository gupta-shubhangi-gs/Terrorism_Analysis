import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app

# Connect to your app pages
from apps import app1, app2, app3

#image_filename = '/Users/shubhangigupta/Desktop/img.png' # replace with your own image
#test_base64 = base64.b64encode(open(image_filename, 'rb').read()).decode('ascii')
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    html.Div([
        html.Div([
            html.Div([
                html.H3('Global Terrrorism Database', style={'margin-bottom': '0px', 'color': 'white'}),
                html.H5('1970 - 2017', style={'margin-top': '0px', 'color': 'white'})
            ])

        ], className='six column', id='title')

    ], id='header', className='row flex-display', style={'margin-botttom': '25px'}),

    html.Div([
      html.Div([
        dcc.Link('Global Terrorism', href='/apps/app1', style = {"margin-bottom": "30px", 'padding': '25px', 'fontWeight': 'bold', 'color': 'red','textAlign': 'center'}),
        dcc.Link('Casualities', href='/apps/app2', style = {"margin-bottom": "30px", 'padding': '25px', 'fontWeight': 'bold', 'color': 'red','textAlign': 'center'}),
        dcc.Link('Analysis', href='/apps/app3', style = {"margin-bottom": "30px", 'padding': '25px', 'fontWeight': 'bold', 'color': 'red','textAlign': 'center'}),

        ], className = 'six column', id = "title1"),


    ], id = "header1", className = "row flex-display", style={'margin-botttom': '25px'}),
    # html.Div([
    #     html.Img(src='data:image/png;base64,{}'.format(test_base64)),
    # ], className = 'six column', id = "title2"),

    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app1':
        return app1.layout
    if pathname == '/apps/app2':
        return app2.layout
    if pathname == '/apps/app3':
        return app3.layout
    else:
        return "No Selection"


if __name__ == '__main__':
    app.run_server(debug=False)