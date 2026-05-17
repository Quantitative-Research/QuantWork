import dash
from dash import dcc, html
from typing import List, Dict

REFERENCE_OPTIONS: List[Dict[str, str]] = [
    {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
    {"label": "Euro Stoxx 50 (^STOXX50E)", "value": "^STOXX50E"},
    {"label": "MSCI World (^MSCI)", "value": "^MSCI"},
    {"label": "CAC 40 (^FCHI)", "value": "^FCHI"},
    {"label": "BTC-EUR (BTCEUR)", "value": "BTCEUR"},
    {"label": "ETH-EUR (ETHEUR)", "value": "ETHEUR"},
]

def create_dash_app():
    app = dash.Dash(__name__)
    app.title = "Markowitz Portfolio Dashboard"
    app.layout = html.Div(
        style={
            "maxWidth": "1400px",
            "margin": "auto",
            "fontFamily": "Arial",
            "padding": "20px"
        },
        children=[
            html.H1("Markowitz Portfolio Allocation", style={"textAlign": "center"}),
            html.Hr(),
            html.Div(
                id="control-panel",
                style={
                    "padding": "20px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "10px",
                    "marginBottom": "20px"
                },
                children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px", "marginBottom": "20px"},
                        children=[
                            html.Div([
                                html.Label("Assets (company names or tickers, comma-separated)", style={"fontWeight": "bold"}),
                                dcc.Input(
                                    id="tickers-input",
                                    type="text",
                                    value="Apple, Microsoft, Google, Amazon",
                                    style={"width": "100%", "padding": "8px", "fontSize": "14px"}
                                ),
                            ]),
                            html.Div([
                                html.Label("Settings", style={"fontWeight": "bold"}),
                                dcc.Checklist(
                                    id="allow-short",
                                    options=[{"label": " Allow short selling", "value": "short"}],
                                    value=[],
                                    style={"marginTop": "10px"}
                                ),
                            ]),
                        ]
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
                        children=[
                            html.Div([
                                html.Label("Historical window for calibration (days)", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="historical-window",
                                    min=60,
                                    max=3650,
                                    step=30,
                                    value=730,
                                    marks={60: "2 mo", 365: "1 yr", 730: "2 yr", 1825: "5 yr", 3650: "10 yr"},
                                    tooltip={"placement": "bottom", "always_visible": False}
                                ),
                                html.Div(id="window-display", style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                            ]),
                            html.Div([
                                html.Label("Reference ticker for beta calculation", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="reference-ticker",
                                    options=REFERENCE_OPTIONS,  # type: ignore
                                    value="^GSPC",
                                    placeholder="Select or type a ticker or company name",
                                    searchable=True,
                                    clearable=True,
                                    style={"width": "100%"}
                                ),
                            ]),
                        ]
                    ),
                    html.Div([
                        html.Label("Target annual return (optional)", style={"fontWeight": "bold"}),
                        dcc.Slider(
                            id="target-return",
                            min=0,
                            max=0.25,
                            step=0.01,
                            value=None,
                            marks={i / 100: f"{i}%" for i in range(0, 26, 5)},
                            tooltip={"placement": "bottom"}
                        ),
                    ]),
                    html.Div([
                        html.Label("Custom weights (optional, comma-separated, in % or decimal)", style={"fontWeight": "bold"}),
                        dcc.Input(
                            id="custom-weights",
                            type="text",
                            placeholder="e.g., 25,25,25,25 or 0.25,0.25,0.25,0.25",
                            style={"width": "100%", "padding": "8px", "fontSize": "14px", "marginTop": "5px"}
                        ),
                        html.P("Leave empty to use Markowitz optimization", style={"fontSize": "12px", "color": "#666", "marginTop": "5px"}),
                    ], style={"marginTop": "15px"}),
                    html.Br(),
                    html.Button(
                        "Compute allocation",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "10px",
                            "padding": "12px 30px",
                            "fontSize": "16px",
                            "backgroundColor": "#0078d4",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer"
                        }
                    ),
                ]
            ),
            html.Hr(),
            html.Div(id="output-warning", style={"color": "red", "marginBottom": "20px", "fontWeight": "bold"}),
            dcc.Tabs(
                id="tabs",
                value="tab-allocation",
                children=[
                    dcc.Tab(
                        label="Allocation",
                        value="tab-allocation",
                        style={"padding": "20px"},
                        children=[
                            html.Div([
                                dcc.Graph(id="weights-pie", style={"display": "inline-block", "width": "48%"}),
                                html.Div(
                                    id="allocation-metrics",
                                    style={"display": "inline-block", "width": "48%", "verticalAlign": "top", "paddingLeft": "20px"}
                                ),
                            ]),
                            html.Br(),
                            html.Div(id="weights-table"),
                        ]
                    ),
                    dcc.Tab(
                        label="Graphs",
                        value="tab-graphs",
                        style={"padding": "20px"},
                        children=[
                            html.Div([
                                html.Label("Time window for graphs (days)", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="graph-window",
                                    min=30,
                                    max=1825,
                                    step=30,
                                    value=365,
                                    marks={30: "1 mo", 90: "3 mo", 180: "6 mo", 365: "1 yr", 730: "2 yr"},
                                    tooltip={"placement": "bottom"}
                                ),
                            ], style={"marginBottom": "30px"}),
                            html.Div([
                                dcc.Graph(id="portfolio-returns-graph"),
                                dcc.Graph(id="rolling-volatility-graph"),
                            ]),
                            html.Div(
                                id="portfolio-stats",
                                style={
                                    "padding": "20px",
                                    "backgroundColor": "#f9f9f9",
                                    "borderRadius": "10px",
                                    "marginTop": "20px"
                                }
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
    return app