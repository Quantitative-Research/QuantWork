from usage.Markowitz.dashboard_app import create_dash_app
from usage.Markowitz.dashboard_logic import compute_portfolio_metrics
from src.Models.Allocation import OptimalAllocation
from src.utils.TickerResolver import resolve_ticker
from src.MarketDataLoader.HistoricalPricesLoader import load_prices
from src.Models import PortfolioOptimizer as po
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
    State("reference-ticker", "value"),
    State("custom-weights", "value")
)
def compute_allocation(n_clicks, tickers_input, allow_short, target_return, hist_window, ref_ticker, custom_weights):
    if not tickers_input:
        return {}, None, None, "Please enter at least one asset."
    raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(raw_assets) < 1:
        return {}, None, None, "Please enter at least one asset."
    try:
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        
        # Check if custom weights are provided
        if custom_weights and custom_weights.strip():
            # Parse custom weights
            try:
                weight_values = [float(w.strip()) for w in custom_weights.split(",")]
                if len(weight_values) != len(resolved_tickers):
                    return {}, None, None, f"Error: You provided {len(weight_values)} weights but {len(resolved_tickers)} assets."
                
                # Normalize weights (handle both % and decimal formats)
                total = sum(weight_values)
                if total == 0:
                    return {}, None, None, "Error: Sum of weights is zero."
                
                # If max weight > 1, assume they're percentages
                if max(weight_values) > 1:
                    weight_values = [w / 100 for w in weight_values]
                
                # Normalize to sum to 1
                total = sum(weight_values)
                weight_values = [w / total for w in weight_values]
                
                weights = pd.Series(weight_values, index=resolved_tickers)
                allocation_method = "Custom"
            except ValueError as e:
                return {}, None, None, f"Error parsing custom weights: {str(e)}"
        else:
            # Use Markowitz optimization
            target_ret = target_return if target_return and target_return > 0 else None
            weights = OptimalAllocation(
                tickers=resolved_tickers,
                method="Markowitz",
                allow_short=("short" in allow_short),
                target_return=target_ret,
                period_days=int(hist_window)
            )
            allocation_method = "Markowitz"
        
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
    State("reference-ticker", "value"),
    State("custom-weights", "value"),
    prevent_initial_call=True
)
def update_graphs(n_clicks, tickers_input, allow_short, target_return, hist_window, graph_window, ref_ticker, custom_weights):
    if not tickers_input:
        return go.Figure(), go.Figure(), html.Div("Please enter assets", style={"color": "orange"})
    
    try:
        raw_assets = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if not raw_assets:
            return go.Figure(), go.Figure(), html.Div("No valid assets entered", style={"color": "orange"})
        
        resolved_tickers = [resolve_ticker(a) for a in raw_assets]
        target_ret = target_return if target_return and target_return > 0 else None
        
        weights = OptimalAllocation(
            tickers=resolved_tickers,
            method="Markowitz",
            allow_short=("short" in allow_short),
            target_return=target_ret,
            period_days=int(hist_window)
        )
        
        prices = load_prices(resolved_tickers, period_days=int(graph_window))
        returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)
        
        if returns.empty or len(returns) == 0:
            return go.Figure(), go.Figure(), html.Div("No returns data available", style={"color": "red"})
        
        # Align weights with returns columns
        weights_aligned = weights.reindex(returns.columns, fill_value=0)
        
        # Portfolio returns
        portfolio_returns = (returns * weights_aligned.values).sum(axis=1)
        portfolio_returns_cumulative = (1 + portfolio_returns).cumprod() - 1
        
        # Normalize portfolio index to timezone-naive dates
        portfolio_returns_cumulative.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in portfolio_returns_cumulative.index])
        portfolio_returns.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in portfolio_returns.index])
        
        returns_fig = go.Figure()
        returns_fig.add_trace(go.Scatter(
            x=portfolio_returns_cumulative.index,
            y=portfolio_returns_cumulative.values,
            mode='lines',
            name='Portfolio Return',
            line=dict(color='#0078d4', width=2)
        ))
        
        # Rolling volatility
        rolling_vol = portfolio_returns.rolling(window=20).std() * np.sqrt(252)
        
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='Portfolio Volatility',
            line=dict(color='#0078d4', width=2)
        ))
        
        # Add reference ticker if provided
        if ref_ticker and ref_ticker.strip():
            try:
                print(f"DEBUG: ref_ticker = '{ref_ticker}'")
                ref_ticker_resolved = resolve_ticker(ref_ticker.strip())
                print(f"DEBUG: ref_ticker_resolved = '{ref_ticker_resolved}'")
                
                ref_prices = load_prices([ref_ticker_resolved], period_days=int(graph_window))
                print(f"DEBUG: ref_prices keys = {ref_prices.keys()}")
                print(f"DEBUG: ref_prices['Close'] columns = {ref_prices['Close'].columns.tolist()}")
                
                ref_returns = po.daily_log_returns(ref_prices["Close"])
                print(f"DEBUG: ref_returns shape = {ref_returns.shape}")
                print(f"DEBUG: ref_returns columns = {ref_returns.columns.tolist()}")
                
                if ref_ticker_resolved in ref_returns.columns:
                    # Align reference index to dates
                    ref_returns.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in ref_returns.index])
                    print(f"DEBUG: ref_returns index after alignment = {ref_returns.index[:5].tolist()}")
                    
                    # Align with portfolio dates
                    common_idx = portfolio_returns_cumulative.index.intersection(ref_returns.index)
                    print(f"DEBUG: common_idx length = {len(common_idx)}")
                    print(f"DEBUG: portfolio_returns_cumulative.index = {portfolio_returns_cumulative.index[:5].tolist()}")
                    print(f"DEBUG: ref_returns.index = {ref_returns.index[:5].tolist()}")
                    
                    if len(common_idx) > 0:
                        ref_returns_common = ref_returns.loc[common_idx, ref_ticker_resolved]
                        ref_cumulative = (1 + ref_returns_common).cumprod() - 1
                        
                        # Add reference to returns plot
                        returns_fig.add_trace(go.Scatter(
                            x=ref_cumulative.index,
                            y=ref_cumulative.values,
                            mode='lines',
                            name=f'{ref_ticker_resolved} (Reference)',
                            line=dict(color='#ff6b6b', width=2, dash='dash')
                        ))
                        print(f"DEBUG: Added reference returns trace")
                        
                        # Add reference rolling volatility
                        ref_rolling_vol = ref_returns_common.rolling(window=20).std() * np.sqrt(252)
                        vol_fig.add_trace(go.Scatter(
                            x=ref_rolling_vol.index,
                            y=ref_rolling_vol.values,
                            mode='lines',
                            name=f'{ref_ticker_resolved} Volatility',
                            line=dict(color='#ff6b6b', width=2, dash='dash')
                        ))
                        print(f"DEBUG: Added reference volatility trace")
                    else:
                        print(f"DEBUG: No common index found")
                else:
                    print(f"DEBUG: ref_ticker_resolved '{ref_ticker_resolved}' not in ref_returns.columns: {ref_returns.columns.tolist()}")
            except Exception as ref_e:
                print(f"DEBUG: Error processing reference ticker: {str(ref_e)}")
                import traceback
                traceback.print_exc()
        
        returns_fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified"
        )
        
        vol_fig.update_layout(
            title="Rolling Volatility Comparison (20-day, Annualized)",
            xaxis_title="Date",
            yaxis_title="Volatility",
            hovermode="x unified"
        )
        
        # Stats
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
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False)
        return empty_fig, empty_fig, html.Div(error_msg, style={"color": "red", "whiteSpace": "pre-wrap"})

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)