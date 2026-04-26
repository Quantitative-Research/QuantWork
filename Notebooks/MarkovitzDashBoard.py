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
from Models import PortfolioOptimizer as po
from MarketDataLoader.HistoricalPricesLoader import load_prices
from utils.TickerResolver import resolve_ticker

import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime

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
        "maxWidth": "1400px",
        "margin": "auto",
        "fontFamily": "Arial",
        "padding": "20px"
    },
    children=[

        html.H1("Markowitz Portfolio Allocation", style={"textAlign": "center"}),
        html.Hr(),

        # ---------- Controls Panel ----------
        html.Div(
            id="control-panel",
            style={
                "padding": "20px",
                "backgroundColor": "#f9f9f9",
                "borderRadius": "10px",
                "marginBottom": "20px"
            },
            children=[
                # Row 1: Assets & Short Selling
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

                # Row 2: Calibration window & Reference ticker
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
                            dcc.Input(
                                id="reference-ticker",
                                type="text",
                                value="^GSPC",
                                placeholder="e.g., ^GSPC (S&P 500), ^FCHI (CAC40)",
                                style={"width": "100%", "padding": "8px", "fontSize": "14px"}
                            ),
                        ]),
                    ]
                ),

                # Row 3: Target Return
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

        # ---------- Error/Warning Messages ----------
        html.Div(id="output-warning", style={"color": "red", "marginBottom": "20px", "fontWeight": "bold"}),

        # ---------- Tabs ----------
        dcc.Tabs(
            id="tabs",
            value="tab-allocation",
            children=[
                # Tab 1: Allocation
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

                # Tab 2: Graphs
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

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

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
        # Resolve tickers
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        target_ret = target_return if target_return and target_return > 0 else None

        # Compute optimal allocation
        weights = OptimalAllocation(
            tickers=resolved_tickers,
            method="Markovitz",
            allow_short=("short" in allow_short),
            target_return=target_ret,
            period_days=int(hist_window)
        )

        # Create allocation table
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

        # Allocation table
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

        # Calculate portfolio metrics (beta, volatility)
        try:
            prices = load_prices(resolved_tickers, period_days=int(hist_window))
            returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)
            
            # Calculate portfolio volatility
            port_vol = po.portfolio_volatility(returns, weights)
            
            # Try to compute beta if reference ticker is provided
            beta_info = html.Div()
            if ref_ticker and ref_ticker.strip():
                try:
                    ref_prices = load_prices([ref_ticker.strip()], period_days=int(hist_window))
                    ref_returns = po.daily_log_returns(ref_prices["Close"])
                    ref_ticker_col = ref_ticker.strip()
                    if ref_ticker_col in ref_returns.columns:
                        port_beta = po.portfolio_beta(returns, weights, ref_returns[ref_ticker_col])
                        beta_info = html.Div([
                            html.H4(f"Beta (vs {ref_ticker.strip()}): {port_beta:.3f}"),
                            html.P("Higher beta means more volatile relative to the reference index")
                        ])
                except Exception as e:
                    beta_info = html.Div([html.P(f"Beta calculation error: {str(e)}", style={"color": "orange"})])
            
            metrics = html.Div([
                html.H4(f"Portfolio Volatility (annual): {port_vol:.2%}"),
                html.P("Annualized volatility based on historical data"),
                html.Br(),
                beta_info
            ])

        except Exception as e:
            metrics = html.Div([html.P(f"Metrics calculation error: {str(e)}", style={"color": "orange"})])

        return pie, table, metrics, ""

    except Exception as e:
        return {}, None, None, f"Error: {str(e)}"


@app.callback(
    Output("portfolio-returns-graph", "figure"),
    Output("rolling-volatility-graph", "figure"),
    Output("portfolio-stats", "children"),
    Input("run-btn", "n_clicks"),
    State("tickers-input", "value"),
    State("allow-short", "value"),
    State("target-return", "value"),
    State("historical-window", "value"),
    State("graph-window", "value"),
    State("reference-ticker", "value")
)
def update_graphs(n_clicks, tickers_input, allow_short, target_return, hist_window, graph_window, ref_ticker):
    if not tickers_input or n_clicks == 0:
        empty_fig = go.Figure().add_annotation(text="Click 'Compute allocation' first")
        return empty_fig, empty_fig, html.Div("No data yet")

    try:
        raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        target_ret = target_return if target_return and target_return > 0 else None

        # Compute weights
        weights = OptimalAllocation(
            tickers=resolved_tickers,
            method="Markovitz",
            allow_short=("short" in allow_short),
            target_return=target_ret,
            period_days=int(hist_window)
        )

        # Load price data for graph window
        prices = load_prices(resolved_tickers, period_days=int(graph_window))
        returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)

        # Compute portfolio returns
        port_returns = po.portfolio_returns(returns, weights)
        port_cum_returns = (1 + port_returns).cumprod() - 1
        
        port_returns_df = pd.DataFrame({
            "Date": port_returns.index,
            "Portfolio": port_cum_returns.values
        })

        # Add reference benchmark if provided
        benchmark_label = "Reference"
        if ref_ticker and ref_ticker.strip():
            try:
                ref_ticker_clean = ref_ticker.strip()
                ref_prices = load_prices([ref_ticker_clean], period_days=int(graph_window))
                ref_returns = po.daily_log_returns(ref_prices["Close"])
                if ref_ticker_clean in ref_returns.columns:
                    ref_cum_returns = (1 + ref_returns[ref_ticker_clean]).cumprod() - 1
                    # Align dates
                    common_dates = port_returns_df["Date"].isin(ref_cum_returns.index)
                    port_returns_df.loc[common_dates, "Reference"] = ref_cum_returns.loc[port_returns_df.loc[common_dates, "Date"]].values
                    benchmark_label = f"Reference ({ref_ticker_clean})"
            except Exception as e:
                # If reference fails, just show portfolio
                pass

        # Graph 1: Cumulative returns
        returns_fig = go.Figure()
        returns_fig.add_trace(go.Scatter(
            x=port_returns_df["Date"],
            y=port_returns_df["Portfolio"],
            mode='lines',
            name='Portfolio',
            line=dict(color='#0078d4', width=2)
        ))
        
        if "Reference" in port_returns_df.columns:
            returns_fig.add_trace(go.Scatter(
                x=port_returns_df["Date"],
                y=port_returns_df["Reference"],
                mode='lines',
                name=benchmark_label,
                line=dict(color='#d60000', width=2, dash='dash')
            ))
        
        returns_fig.update_layout(
            title=f"Portfolio vs Benchmark Cumulative Returns (last {graph_window} days)",
            xaxis_title="",
            yaxis_title="Return",
            hovermode='x unified',
            template='plotly_white'
        )
        returns_fig.update_yaxes(tickformat=".2%")

        
        returns_fig.update_layout(
            title=f"Portfolio vs Benchmark Cumulative Returns (last {graph_window} days)",
            xaxis_title="",
            yaxis_title="Return",
            hovermode='x unified',
            template='plotly_white'
        )
        returns_fig.update_yaxes(tickformat=".2%")

        # Graph 2: Rolling volatility (30-day window)
        rolling_std = returns.multiply(weights).sum(axis=1).rolling(window=30).std() * np.sqrt(252)
        rolling_vol_df = pd.DataFrame({
            "Date": rolling_std.index,
            "Rolling Volatility (30d)": rolling_std.values
        }).dropna()

        vol_fig = px.line(
            rolling_vol_df,
            x="Date",
            y="Rolling Volatility (30d)",
            title=f"Portfolio 30-day Rolling Volatility (last {graph_window} days)",
            labels={"Rolling Volatility (30d)": "Volatility", "Date": ""}
        )
        vol_fig.update_yaxes(tickformat=".2%")

        # Calculate statistics
        annual_return = port_returns.mean() * 252
        annual_vol = po.portfolio_volatility(returns, weights)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        max_drawdown = (port_cum_returns.cummax() - port_cum_returns).max()

        stats = html.Div([
            html.H4("Portfolio Statistics", style={"marginBottom": "20px"}),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "20px"}, children=[
                html.Div([
                    html.P("Annual Return", style={"fontWeight": "bold"}),
                    html.H3(f"{annual_return:.2%}", style={"color": "#0078d4"})
                ], style={"padding": "15px", "backgroundColor": "#f0f8ff", "borderRadius": "8px"}),
                html.Div([
                    html.P("Annual Volatility", style={"fontWeight": "bold"}),
                    html.H3(f"{annual_vol:.2%}", style={"color": "#0078d4"})
                ], style={"padding": "15px", "backgroundColor": "#f0f8ff", "borderRadius": "8px"}),
                html.Div([
                    html.P("Sharpe Ratio", style={"fontWeight": "bold"}),
                    html.H3(f"{sharpe_ratio:.2f}", style={"color": "#0078d4"})
                ], style={"padding": "15px", "backgroundColor": "#f0f8ff", "borderRadius": "8px"}),
                html.Div([
                    html.P("Max Drawdown", style={"fontWeight": "bold"}),
                    html.H3(f"{max_drawdown:.2%}", style={"color": "#d60000"})
                ], style={"padding": "15px", "backgroundColor": "#fff0f0", "borderRadius": "8px"})
            ])
        ])

        return returns_fig, vol_fig, stats

    except Exception as e:
        error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
        return error_fig, error_fig, html.Div(f"Error computing graphs: {str(e)}", style={"color": "red"})


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
# --------------------------------------------------