import sys
from pathlib import Path

# Add src folder to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # QuantWork/
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Now imports work
from Models.Allocation import OptimalAllocation
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
# --------------------------------------------------
# App setup
# --------------------------------------------------

app = dash.Dash(__name__)
app.title = "Markowitz Portfolio Dashboard"


AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN",
    "ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA", "BNP.PA", "OR.PA", "SGO.PA", "VIE.PA", "SCHNE.PA"
]


# --------------------------------------------------
# Layout
# --------------------------------------------------

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"},
    children=[

        html.H1("Markowitz Portfolio Allocation", style={"textAlign": "center"}),

        html.Hr(),

        # ---------- Controls ----------
        html.Div([
            html.Label("Select assets"),
            dcc.Dropdown(
                id="tickers",
                options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
                value=["AAPL", "MSFT", "GOOG", "AMZN"],
                multi=True
            ),

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
                tooltip={"placement": "bottom", "always_visible": False}
            ),

            html.Br(),

            html.Button("Compute allocation", id="run-btn", n_clicks=1)

        ], style={"padding": "20px", "backgroundColor": "#f9f9f9", "borderRadius": "10px"}),

        html.Hr(),

        # ---------- Outputs ----------
        html.Div(id="output-warning", style={"color": "red"}),

        html.Div([
            dcc.Graph(id="weights-pie"),
            html.Div(id="weights-table")
        ])

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
    State("tickers", "value"),
    State("allow-short", "value"),
    State("target-return", "value")
)
def compute_allocation(n_clicks, selected_tickers, allow_short, target_return):
    # Validation
    if not selected_tickers or len(selected_tickers) < 2:
        return {}, None, "Please select at least two assets."

    try:
        # Convert slider None or 0 to None for OptimalAllocation
        target_ret = target_return if target_return and target_return > 0 else None

        # Compute optimal weights
        weights = OptimalAllocation(
            tickers=selected_tickers,
            method="Markovitz",
            allow_short=("short" in allow_short),
            target_return=target_ret
        )

        # Prepare DataFrame for pie and table
        df = pd.DataFrame({
            "Ticker": selected_tickers,
            "Weight (%)": weights * 100
        })

        # Pie chart
        pie = px.pie(
            df,
            names="Ticker",
            values="Weight (%)",
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        # Table
        table = html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Ticker"),
                    html.Th("Weight (%)")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(row["Ticker"]),
                        html.Td(f"{row['Weight (%)']:.2f}")
                    ]) for _, row in df.iterrows()
                ] + [
                    html.Tr([
                        html.Td("Total"),
                        html.Td(f"{df['Weight (%)'].sum():.2f}")
                    ])
                ])
            ],
            style={"marginTop": "20px", "borderCollapse": "collapse"}
        )

        return pie, table, ""

    except Exception as e:
        return {}, None, str(e)



# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port = 8050)
