from dashboard_app import create_dash_app
from dashboard_logic import compute_portfolio_metrics
from Models.Allocation import OptimalAllocation
from utils.TickerResolver import resolve_ticker
from dash import Input, Output, State, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = create_dash_app()

@app.callback(
    Output("window-display", "children"),
    Input("historical-window", "value")
)
def update_window_display(days):
    if days < 30:
        return f"({days} days)"
    elif days < 365:
        return f"({days / 30:.1f} months)"
    else:
        return f"({days / 365:.1f} years)"

@app.callback(
    Output("weights-pie", "figure"),
    Output("weights-table", "children"),
    Output("allocation-metrics", "children"),
    Output("output-warning", "children"),
    Input("run-btn", "n_clicks"),
    State("tickers-input", "value"),
    State("allow-short", "value"),
    State("target-return", "value"),
    State("historical-window", "value"),
    State("reference-ticker", "value")
)
def compute_allocation(n_clicks, tickers_input, allow_short, target_return, hist_window, ref_ticker):
    if not tickers_input:
        return {}, None, None, "Please enter at least one asset."
    raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(raw_assets) < 1:
        return {}, None, None, "Please enter at least one asset."
    try:
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        target_ret = target_return if target_return and target_return > 0 else None
        weights = OptimalAllocation(
            tickers=resolved_tickers,
            method="Markovitz",
            allow_short=("short" in allow_short),
            target_return=target_ret,
            period_days=int(hist_window)
        )
        df = pd.DataFrame({
            "Asset": raw_assets,
            "Ticker": resolved_tickers,
            "Weight (%)": weights * 100
        })
        pie = px.pie(
            df,
            names="Ticker",
            values="Weight (%)",
            title="Portfolio Allocation"
        )
        table = html.Table(
            style={
                "marginTop": "20px",
                "borderCollapse": "collapse",
                "width": "100%",
                "border": "1px solid #ddd"
            },
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th(label, style={"padding": "10px", "backgroundColor": "#f0f0f0", "border": "1px solid #ddd"}) 
                         for label in ["Asset", "Ticker", "Weight (%)"]],
                        style={"fontWeight": "bold"}
                    )
                ),
                html.Tbody(
                    [
                        html.Tr([
                            html.Td(row["Asset"], style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(row["Ticker"], style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(f"{row['Weight (%)']:.2f}", style={"padding": "10px", "border": "1px solid #ddd"})
                        ])
                        for _, row in df.iterrows()
                    ] + [
                        html.Tr([
                            html.Td(html.B("Total"), style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td("", style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(html.B(f"{df['Weight (%)'].sum():.2f}"), style={"padding": "10px", "border": "1px solid #ddd"})
                        ], style={"backgroundColor": "#f0f0f0"})
                    ]
                )
            ]
        )
        port_vol, beta = compute_portfolio_metrics(resolved_tickers, weights, hist_window, ref_ticker)
        beta_info = html.Div()
        if beta is not None:
            beta_info = html.Div([
                html.H4(f"Beta (vs {ref_ticker.strip()}): {beta:.3f}"),
                html.P("Higher beta means more volatile relative to the reference index")
            ])
        metrics = html.Div([
            html.H4(f"Portfolio Volatility (annual): {port_vol:.2%}"),
            html.P("Annualized volatility based on historical data"),
            html.Br(),
            beta_info
        ])
        return pie, table, metrics, ""
    except Exception as e:
        return {}, None, None, f"Error: {str(e)}"

# Add your other callbacks (graphs, stats, etc.) here as needed

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)