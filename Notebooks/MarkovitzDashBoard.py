import sys
from pathlib import Path

# --------------------------------------------------
# Path setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # QuantWork/
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# --------------------------------------------------
# Imports
# --------------------------------------------------
from Models.Allocation import OptimalAllocation
from utils.TickerResolver import resolve_ticker

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = dash.Dash(__name__)
app.title = "Markowitz Portfolio Dashboard"

# --------------------------------------------------
# Layout
# --------------------------------------------------
app.layout = html.Div(
    style={
        "maxWidth": "1100px",
        "margin": "auto",
        "fontFamily": "Arial"
    },
    children=[

        html.H1("Markowitz Portfolio Allocation", style={"textAlign": "center"}),

        html.Hr(),

        # ---------- Controls ----------
        html.Div(
            style={
                "padding": "20px",
                "backgroundColor": "#f9f9f9",
                "borderRadius": "10px"
            },
            children=[

                html.Label("Assets (company names or tickers, comma-separated)"),
                dcc.Input(
                    id="tickers-input",
                    type="text",
                    value="Apple, Microsoft, Google, Amazon",
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                dcc.Checklist(
                    id="allow-short",
                    options=[{"label": " Allow short selling", "value": "short"}],
                    value=[]
                ),

                html.Br(),

                html.Label("Target annual return (optional)"),
                dcc.Slider(
                    id="target-return",
                    min=0,
                    max=0.25,
                    step=0.01,
                    value=None,
                    marks={i / 100: f"{i}%" for i in range(0, 26, 5)},
                    tooltip={"placement": "bottom"}
                ),

                html.Br(),

                html.Button(
                    "Compute allocation",
                    id="run-btn",
                    n_clicks=0,
                    style={"marginTop": "10px"}
                ),
            ]
        ),

        html.Hr(),

        # ---------- Outputs ----------
        html.Div(id="output-warning", style={"color": "red"}),

        dcc.Graph(id="weights-pie"),

        html.Div(id="weights-table")
    ]
)

# --------------------------------------------------
# Callback
# --------------------------------------------------
@app.callback(
    Output("weights-pie", "figure"),
    Output("weights-table", "children"),
    Output("output-warning", "children"),
    Input("run-btn", "n_clicks"),
    State("tickers-input", "value"),
    State("allow-short", "value"),
    State("target-return", "value")
)
def compute_allocation(n_clicks, tickers_input, allow_short, target_return):

    if not tickers_input:
        return {}, None, "Please enter at least one asset."

    raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]

    if len(raw_assets) < 1:
        return {}, None, "Please enter at least one asset."

    try:
        # Resolve tickers
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]

        target_ret = target_return if target_return and target_return > 0 else None

        weights = OptimalAllocation(
            tickers=resolved_tickers,
            method="Markovitz",
            allow_short=("short" in allow_short),
            target_return=target_ret
        )

        df = pd.DataFrame({
            "Asset": raw_assets,
            "Ticker": resolved_tickers,
            "Weight (%)": weights * 100
        })

        # Pie chart
        pie = px.pie(
            df,
            names="Ticker",
            values="Weight (%)",
            title="Portfolio Allocation"
        )

        # Table
        table = html.Table(
            style={
                "marginTop": "20px",
                "borderCollapse": "collapse",
                "width": "100%"
            },
            children=[
                html.Thead(html.Tr([
                    html.Th("Asset"),
                    html.Th("Ticker"),
                    html.Th("Weight (%)")
                ])),
                html.Tbody(
                    [
                        html.Tr([
                            html.Td(row["Asset"]),
                            html.Td(row["Ticker"]),
                            html.Td(f"{row['Weight (%)']:.2f}")
                        ])
                        for _, row in df.iterrows()
                    ] + [
                        html.Tr([
                            html.Td(html.B("Total")),
                            html.Td(""),
                            html.Td(html.B(f"{df['Weight (%)'].sum():.2f}"))
                        ])
                    ]
                )
            ]
        )

        return pie, table, ""

    except Exception as e:
        return {}, None, str(e)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
# --------------------------------------------------