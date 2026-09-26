import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html

from BlackLitterman.black_litterman import black_litterman_weights
from BlackLitterman.dashboard_app import create_dash_app
from BlackLitterman.dashboard_logic import compute_portfolio_metrics
from MarketDataLoader.HistoricalPricesLoader import load_prices
from Models import PortfolioOptimizer as po
from utils.TickerResolver import resolve_ticker

app = create_dash_app()


@app.callback(
    Output("window-display", "children"),
    Input("historical-window", "value"),
)
def update_window_display(days):
    if days < 30:
        return f"({days} days)"
    if days < 365:
        return f"({days / 30:.1f} months)"
    return f"({days / 365:.1f} years)"


@app.callback(
    Output("weights-pie", "figure"),
    Output("weights-table", "children"),
    Output("allocation-metrics", "children"),
    Output("output-warning", "children"),
    Input("run-btn", "n_clicks"),
    State("tickers-input", "value"),
    State("allow-short", "value"),
    State("historical-window", "value"),
    State("reference-ticker", "value"),
    State("bl-views", "value"),
    State("bl-confidence", "value"),
)
def compute_allocation(n_clicks, tickers_input, allow_short, hist_window, ref_ticker, bl_views, bl_confidence):
    if not tickers_input:
        return {}, None, None, "Please enter at least one asset."

    raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(raw_assets) < 1:
        return {}, None, None, "Please enter at least one asset."

    try:
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        prices = load_prices(resolved_tickers, period_days=int(hist_window))
        returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)

        if returns.empty:
            return {}, None, None, "No valid return data returned for these inputs."

        view_values = None
        confidence_values = None
        if bl_views and bl_views.strip():
            view_values = [float(v.strip()) for v in bl_views.split(",") if v.strip()]
            if len(view_values) != len(resolved_tickers):
                return {}, None, None, f"Error: You provided {len(view_values)} view returns but {len(resolved_tickers)} assets."

        if bl_confidence and bl_confidence.strip():
            confidence_values = [float(v.strip()) for v in bl_confidence.split(",") if v.strip()]
            if len(confidence_values) == 1:
                confidence_values = confidence_values * len(resolved_tickers)
            elif len(confidence_values) != len(resolved_tickers):
                return {}, None, None, f"Error: You provided {len(confidence_values)} confidence values but {len(resolved_tickers)} assets."

        weights = black_litterman_weights(
            returns=returns,
            market_weights=None,
            views=view_values,
            view_confidence=confidence_values,
            allow_short=("short" in allow_short),
        )

        df = pd.DataFrame({
            "Asset": raw_assets,
            "Ticker": resolved_tickers,
            "Weight (%)": weights * 100,
        })

        pie = px.pie(df, names="Ticker", values="Weight (%)", title="Black-Litterman Allocation")
        table = html.Table(
            style={
                "marginTop": "20px",
                "borderCollapse": "collapse",
                "width": "100%",
                "border": "1px solid #ddd",
            },
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th(label, style={"padding": "10px", "backgroundColor": "#f0f0f0", "border": "1px solid #ddd"}) for label in ["Asset", "Ticker", "Weight (%)"]],
                        style={"fontWeight": "bold"},
                    )
                ),
                html.Tbody(
                    [
                        html.Tr([
                            html.Td(row["Asset"], style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(row["Ticker"], style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(f"{row['Weight (%)']:.2f}", style={"padding": "10px", "border": "1px solid #ddd"}),
                        ])
                        for _, row in df.iterrows()
                    ] + [
                        html.Tr([
                            html.Td(html.B("Total"), style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td("", style={"padding": "10px", "border": "1px solid #ddd"}),
                            html.Td(html.B(f"{df['Weight (%)'].sum():.2f}"), style={"padding": "10px", "border": "1px solid #ddd"}),
                        ], style={"backgroundColor": "#f0f0f0"})
                    ]
                ),
            ],
        )

        port_vol, beta = compute_portfolio_metrics(resolved_tickers, weights, hist_window, ref_ticker)
        beta_info = html.Div()
        if beta is not None:
            beta_info = html.Div([
                html.H4(f"Beta (vs {ref_ticker.strip()}): {beta:.3f}"),
                html.P("Higher beta means more volatile relative to the reference index"),
            ])

        metrics = html.Div([
            html.H4(f"Portfolio Volatility (annual): {port_vol:.2%}"),
            html.P("Annualized volatility based on historical data"),
            html.Br(),
            beta_info,
        ])
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
    State("historical-window", "value"),
    State("graph-window", "value"),
    State("reference-ticker", "value"),
    State("bl-views", "value"),
    State("bl-confidence", "value"),
    prevent_initial_call=True,
)
def update_graphs(n_clicks, tickers_input, allow_short, hist_window, graph_window, ref_ticker, bl_views, bl_confidence):
    if not tickers_input:
        return go.Figure(), go.Figure(), html.Div("Please enter assets", style={"color": "orange"})

    try:
        raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if not raw_assets:
            return go.Figure(), go.Figure(), html.Div("No valid assets entered", style={"color": "orange"})

        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        prices = load_prices(resolved_tickers, period_days=int(hist_window))
        returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)

        view_values = None
        confidence_values = None
        if bl_views and bl_views.strip():
            view_values = [float(v.strip()) for v in bl_views.split(",") if v.strip()]
            if len(view_values) != len(resolved_tickers):
                return go.Figure(), go.Figure(), html.Div(f"Error: You provided {len(view_values)} view returns but {len(resolved_tickers)} assets.", style={"color": "red"})
        if bl_confidence and bl_confidence.strip():
            confidence_values = [float(v.strip()) for v in bl_confidence.split(",") if v.strip()]
            if len(confidence_values) == 1:
                confidence_values = confidence_values * len(resolved_tickers)
            elif len(confidence_values) != len(resolved_tickers):
                return go.Figure(), go.Figure(), html.Div(f"Error: You provided {len(confidence_values)} confidence values but {len(resolved_tickers)} assets.", style={"color": "red"})

        weights = black_litterman_weights(
            returns=returns,
            market_weights=None,
            views=view_values,
            view_confidence=confidence_values,
            allow_short=("short" in allow_short),
        )

        graph_prices = load_prices(resolved_tickers, period_days=int(graph_window))
        graph_returns = po.daily_log_returns(graph_prices["Close"]).dropna(axis=1)
        if graph_returns.empty or len(graph_returns) == 0:
            return go.Figure(), go.Figure(), html.Div("No returns data available", style={"color": "red"})

        weights_aligned = weights.reindex(graph_returns.columns, fill_value=0)
        portfolio_returns = (graph_returns * weights_aligned.values).sum(axis=1)
        portfolio_returns_cumulative = (1 + portfolio_returns).cumprod() - 1
        portfolio_returns_cumulative.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in portfolio_returns_cumulative.index])
        portfolio_returns.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in portfolio_returns.index])

        returns_fig = go.Figure()
        returns_fig.add_trace(go.Scatter(
            x=portfolio_returns_cumulative.index,
            y=portfolio_returns_cumulative.values,
            mode="lines",
            name="Portfolio Return",
            line={"color": "#0078d4", "width": 2},
        ))

        rolling_vol = portfolio_returns.rolling(window=20).std() * np.sqrt(252)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode="lines",
            name="Portfolio Volatility",
            line={"color": "#0078d4", "width": 2},
        ))

        if ref_ticker and ref_ticker.strip():
            try:
                ref_ticker_resolved = resolve_ticker(ref_ticker.strip())
                ref_prices = load_prices([ref_ticker_resolved], period_days=int(graph_window))
                ref_returns = po.daily_log_returns(ref_prices["Close"])
                ref_returns.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in ref_returns.index])

                if ref_ticker_resolved in ref_returns.columns:
                    common_idx = portfolio_returns_cumulative.index.intersection(ref_returns.index)
                    if len(common_idx) > 0:
                        ref_returns_common = ref_returns.loc[common_idx, ref_ticker_resolved]
                        ref_cumulative = (1 + ref_returns_common).cumprod() - 1
                        returns_fig.add_trace(go.Scatter(
                            x=ref_cumulative.index,
                            y=ref_cumulative.values,
                            mode="lines",
                            name=f"{ref_ticker_resolved} (Reference)",
                            line={"color": "#ff6b6b", "width": 2, "dash": "dash"},
                        ))

                        ref_rolling_vol = ref_returns_common.rolling(window=20).std() * np.sqrt(252)
                        vol_fig.add_trace(go.Scatter(
                            x=ref_rolling_vol.index,
                            y=ref_rolling_vol.values,
                            mode="lines",
                            name=f"{ref_ticker_resolved} Volatility",
                            line={"color": "#ff6b6b", "width": 2, "dash": "dash"},
                        ))
            except Exception:
                pass

        returns_fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
        )
        vol_fig.update_layout(
            title="Rolling Volatility Comparison (20-day, Annualized)",
            xaxis_title="Date",
            yaxis_title="Volatility",
            hovermode="x unified",
        )

        total_return = portfolio_returns_cumulative.iloc[-1]
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        max_dd = (portfolio_returns_cumulative.cummin() - portfolio_returns_cumulative).max()

        stats = html.Div([
            html.Div([
                html.Div([html.H4("Total Return"), html.P(f"{total_return:.2%}")], style={"flex": 1, "padding": "10px", "textAlign": "center"}),
                html.Div([html.H4("Volatility"), html.P(f"{volatility:.2%}")], style={"flex": 1, "padding": "10px", "textAlign": "center"}),
                html.Div([html.H4("Sharpe Ratio"), html.P(f"{sharpe:.2f}")], style={"flex": 1, "padding": "10px", "textAlign": "center"}),
                html.Div([html.H4("Max Drawdown"), html.P(f"{max_dd:.2%}")], style={"flex": 1, "padding": "10px", "textAlign": "center"}),
            ], style={"display": "flex", "justifyContent": "space-around", "border": "1px solid #ddd", "padding": "20px"})
        ])

        return returns_fig, vol_fig, stats
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False)
        return empty_fig, empty_fig, html.Div(f"Error: {str(e)}", style={"color": "red", "whiteSpace": "pre-wrap"})


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
