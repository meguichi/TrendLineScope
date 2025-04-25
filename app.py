import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.linear_model import RANSACRegressor

# ─── データ取得 & カラム正規化 ────────────────────────────────────
@st.cache_data
def get_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    # MultiIndex flatten + 小文字化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

# ─── RSI 計算 ───────────────────────────────────────────────────
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ─── アップ／ダウンバー別ボリュームプロファイル ───────────────────────
def compute_buy_sell_volume_profile(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    bins = pd.cut(df['close'], bins=n_bins)
    buy_vol  = np.where(df['close'] > df['open'], df['volume'], 0)
    sell_vol = np.where(df['close'] < df['open'], df['volume'], 0)
    profile = (
        df.assign(price_bin=bins, buy_vol=buy_vol, sell_vol=sell_vol)
          .groupby('price_bin')[['buy_vol','sell_vol']]
          .sum()
          .reset_index()
    )
    profile['bin_center'] = profile['price_bin'].apply(lambda iv: iv.mid)
    return profile

# ─── ピボット検出 & トレンドライン生成 ────────────────────────────
def detect_pivots(df: pd.DataFrame, order: int = 5):
    highs = argrelextrema(df['high'].values, np.greater, order=order)[0]
    lows  = argrelextrema(df['low'].values,  np.less,   order=order)[0]
    return highs, lows

def fit_trendlines(df, idxs, kind='resistance', residual_threshold=1.0, min_samples=2):
    lines = []
    y_full = df['high'].values if kind=='resistance' else df['low'].values
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            xi, xj = idxs[i], idxs[j]
            pts_x = np.array([xi, xj]).reshape(-1,1)
            pts_y = y_full[[xi, xj]]
            model = RANSACRegressor(
                residual_threshold=residual_threshold,
                min_samples=min_samples,
                random_state=0
            )
            model.fit(pts_x, pts_y)
            if model.inlier_mask_.mean() >= 0.5:
                slope     = model.estimator_.coef_[0]
                intercept = model.estimator_.intercept_
                lines.append((slope, intercept))
    return lines

# ─── Streamlit アプリ本体 ─────────────────────────────────────────
def main():
    st.title("🔮 TrendLineScope － 総合テクニカル＆需給ダッシュボード")

    # パラメータ入力
    ticker   = st.text_input("銘柄コード（例: 7203.T）", "7203.T")
    period   = st.selectbox("取得期間", ["1mo","3mo","6mo","1y","5y"], index=2)
    interval = st.selectbox("足種", ["1d","1h","30m","15m"], index=0)

    st.markdown("### 📐 ピボット＆ライン検出パラメータ")
    order   = st.slider("order (極値検出の前後本数)", 1, 20, 5)
    thresh  = st.slider("RANSAC 残差許容値", 0.1, 10.0, 1.0, 0.1)
    msamps  = st.slider("RANSAC 最小サンプル数", 2, 10, 2)
    top_n   = st.slider("表示ライン数", 1, 5, 2)

    st.markdown("### 🔧 テクニカルパラメータ")
    ma_s    = st.slider("短期MA（日）", 5, 50, 25, 5)
    ma_l    = st.slider("長期MA（日）", 20, 200, 75, 5)
    rsi_w   = st.slider("RSI 期間（日）", 5, 30, 14, 1)

    st.markdown("### ⚖️ 需給プロファイルパラメータ")
    n_bins  = st.slider("価格帯ビン数", 10, 100, 30, 5)

    if st.button("更新"):
        df = get_data(ticker, period, interval)

        # 移動平均 & RSI
        df[f"ma{ma_s}"] = df['close'].rolling(ma_s).mean()
        df[f"ma{ma_l}"] = df['close'].rolling(ma_l).mean()
        df['rsi']      = compute_rsi(df['close'], window=rsi_w)

        # 需給プロファイル
        prof      = compute_buy_sell_volume_profile(df, n_bins)
        top_buy   = prof.nlargest(3, 'buy_vol')
        top_sell  = prof.nlargest(3, 'sell_vol')

        # ピボット & ライン検出
        highs, lows = detect_pivots(df, order)
        res_lines   = fit_trendlines(df, highs, 'resistance', thresh, msamps)
        sup_lines   = fit_trendlines(df, lows,  'support',    thresh, msamps)

        # ─── プロット: 価格+MA+ライン+需給ゾーン ─────────────────────
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7,0.3], vertical_spacing=0.02)

        x0, x1 = df.index.min(), df.index.max()
        # 買い優勢ゾーン（水色帯）
        for _, row in top_buy.iterrows():
            y0, y1 = row['price_bin'].left, row['price_bin'].right
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor="skyblue", opacity=0.2,
                line_width=0, layer="below",
                row=1, col=1
            )
        # 売り優勢ゾーン（薄ピンク帯）
        for _, row in top_sell.iterrows():
            y0, y1 = row['price_bin'].left, row['price_bin'].right
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor="lightpink", opacity=0.2,
                line_width=0, layer="below",
                row=1, col=1
            )
        # 重要価格帯の水平ライン
        for _, row in pd.concat([top_buy, top_sell]).iterrows():
            mid    = row['bin_center']
            color  = "blue" if row['buy_vol'] > row['sell_vol'] else "red"
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x0, x1=x1, y0=mid, y1=mid,
                line=dict(color=color, width=2, dash="dot"),
                row=1, col=1
            )

        # ローソク足
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color="blue", decreasing_line_color="red",
            name="ローソク足"
        ), row=1, col=1)

        # 移動平均線
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"ma{ma_s}"],
            mode='lines', name=f"MA{ma_s}", line=dict(dash="dash")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"ma{ma_l}"],
            mode='lines', name=f"MA{ma_l}", line=dict(dash="dot")
        ), row=1, col=1)

        # サポート＆レジスタンスライン
        for slope, intercept in res_lines[:top_n]:
            yv = slope * np.arange(len(df)) + intercept
            fig.add_trace(go.Scatter(
                x=df.index, y=yv, mode='lines',
                line=dict(color="orange", dash="dash"),
                name="Resistance"
            ), row=1, col=1)
        for slope, intercept in sup_lines[:top_n]:
            yv = slope * np.arange(len(df)) + intercept
            fig.add_trace(go.Scatter(
                x=df.index, y=yv, mode='lines',
                line=dict(color="lime", dash="dot"),
                name="Support"
            ), row=1, col=1)

        # RSI（下段）
        fig.add_trace(go.Scatter(
            x=df.index, y=df['rsi'], mode='lines', name="RSI"
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="purple",
                      annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="purple",
                      annotation_text="Oversold", row=2, col=1)

        fig.update_layout(
            height=750,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)

        st.subheader("📈 価格＋サポレジライン＋MA＋RSI＋需給ゾーン")
        st.plotly_chart(fig, use_container_width=True)

        # 重要価格帯テーブル
        st.subheader("🔍 重要価格帯（需給プロファイル上位）")
        display = pd.DataFrame({
            "Type": ["Buy_zone"]*3 + ["Sell_zone"]*3,
            "Price_mid": list(top_buy['bin_center']) + list(top_sell['bin_center']),
            "Buy_vol": list(top_buy['buy_vol']) + [None]*3,
            "Sell_vol": [None]*3 + list(top_sell['sell_vol'])
        })
        st.table(display)

if __name__ == "__main__":
    main()