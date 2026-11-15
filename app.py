# -*- coding: utf-8 -*-
"""
A股智能K线分析与趋势实验室（Python版）
运行：
    pip install -r requirements.txt
    streamlit run app.py

说明：
- 使用 AkShare 获取公开历史行情；需要联网。
- 预测仅用于学习机器学习与技术分析，不是投资建议。
- 不提供自动交易或真实买卖指令。
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="A股智能K线分析", page_icon="📈", layout="wide")

st.title("📈 A股智能K线分析与趋势实验室")
st.caption("历史行情可视化 · 技术指标 · 机器学习趋势实验（非投资建议）")

@st.cache_data(ttl=900, show_spinner=False)
def load_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError("没有获取到行情数据，请检查股票代码或稍后重试。")
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "change_pct", "涨跌额": "change",
        "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "close", "high", "low"]).sort_values("date").reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for n in [5, 10, 20, 60]:
        x[f"MA{n}"] = x["close"].rolling(n).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI14"] = 100 - 100 / (1 + rs)
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["DIF"] = ema12 - ema26
    x["DEA"] = x["DIF"].ewm(span=9, adjust=False).mean()
    x["MACD柱"] = (x["DIF"] - x["DEA"]) * 2
    x["收益率"] = x["close"].pct_change()
    x["波动率20"] = x["收益率"].rolling(20).std()
    return x

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    close = df["close"]
    for lag in [1, 2, 3, 5, 10]:
        z[f"return_{lag}"] = close.pct_change(lag)
    for n in [5, 10, 20, 60]:
        ma = close.rolling(n).mean()
        z[f"price_ma_{n}"] = close / ma - 1
    z["rsi"] = df["RSI14"] / 100
    z["macd"] = df["DIF"] / close
    z["macd_signal"] = df["DEA"] / close
    z["volatility"] = df["波动率20"]
    z["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return z.replace([np.inf, -np.inf], np.nan)

def forecast(df: pd.DataFrame, horizon: int = 5):
    """用随机森林回归预测未来 horizon 个交易日累计收益率。"""
    feat = make_features(df)
    target = df["close"].shift(-horizon) / df["close"] - 1
    data = feat.copy()
    data["target"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 100:
        raise ValueError("有效历史数据不足，至少需要约120个交易日。")

    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    model = RandomForestRegressor(
        n_estimators=250, max_depth=5, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    model.fit(train.drop(columns="target"), train["target"])

    if len(test):
        test_pred = model.predict(test.drop(columns="target"))
        mae = mean_absolute_error(test["target"], test_pred)
    else:
        mae = np.nan

    latest_features = feat.iloc[[-1]].copy()
    if latest_features.isna().any(axis=None):
        raise ValueError("最新指标尚未完整，请增加历史数据范围。")
    predicted_return = float(model.predict(latest_features)[0])

    # 用历史测试集误差构造宽松的参考区间，不代表置信区间。
    if len(test):
        residual = np.abs(test["target"].to_numpy() - test_pred)
        error_band = float(np.nanquantile(residual, 0.8))
    else:
        error_band = float(mae) if np.isfinite(mae) else 0.03

    importance = pd.Series(
        model.feature_importances_, index=feat.columns
    ).sort_values(ascending=False)
    return predicted_return, error_band, mae, importance

with st.sidebar:
    st.header("查询设置")
    symbol = st.text_input("A股代码", value="600519", max_chars=6).strip()
    period_years = st.selectbox("历史数据范围", [1, 2, 3, 5], index=2)
    show_ma = st.multiselect("显示均线", ["MA5", "MA10", "MA20", "MA60"],
                             default=["MA5", "MA20", "MA60"])
    horizon = st.selectbox("预测跨度（交易日）", [3, 5, 10], index=1)
    refresh = st.button("🔄 重新获取行情", use_container_width=True)

if not (symbol.isdigit() and len(symbol) == 6):
    st.error("请输入6位A股股票代码，例如 600519、000001 或 300750。")
    st.stop()

if refresh:
    load_stock.clear()

end = pd.Timestamp.today()
start = end - pd.Timedelta(days=365 * period_years)
start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

try:
    with st.spinner("正在获取行情数据并计算指标……"):
        raw = load_stock(symbol, start_s, end_s)
        data = add_indicators(raw)
except Exception as exc:
    st.error(f"行情获取失败：{exc}")
    st.info("请确认网络可用、股票代码正确，并稍后重试。部分数据接口可能会临时不可用。")
    st.stop()

latest = data.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新收盘价", f"{latest['close']:.2f}")
c2.metric("当日涨跌幅", f"{float(latest.get('change_pct', 0)):.2f}%"
          if pd.notna(latest.get("change_pct", np.nan)) else "暂无")
c3.metric("RSI(14)", f"{latest['RSI14']:.1f}" if pd.notna(latest["RSI14"]) else "计算中")
c4.metric("数据日期", latest["date"].strftime("%Y-%m-%d"))

st.subheader(f"{symbol} · 日K线")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, row_heights=[0.72, 0.28])
fig.add_trace(go.Candlestick(
    x=data["date"], open=data["open"], high=data["high"],
    low=data["low"], close=data["close"], name="日K"
), row=1, col=1)
for ma in show_ma:
    fig.add_trace(go.Scatter(x=data["date"], y=data[ma],
                             mode="lines", name=ma), row=1, col=1)
fig.add_trace(go.Bar(x=data["date"], y=data["volume"], name="成交量",
                     marker_opacity=0.55), row=2, col=1)
fig.update_layout(height=720, xaxis_rangeslider_visible=False,
                  margin=dict(l=20, r=20, t=30, b=20),
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("MACD")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Bar(x=data["date"], y=data["MACD柱"], name="MACD柱"))
    macd_fig.add_trace(go.Scatter(x=data["date"], y=data["DIF"], name="DIF"))
    macd_fig.add_trace(go.Scatter(x=data["date"], y=data["DEA"], name="DEA"))
    macd_fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(macd_fig, use_container_width=True)
with right:
    st.subheader("RSI(14)")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=data["date"], y=data["RSI14"], name="RSI"))
    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")
    rsi_fig.update_layout(height=320, yaxis_range=[0, 100],
                          margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(rsi_fig, use_container_width=True)

st.subheader(f"🤖 机器学习趋势实验：未来{horizon}个交易日")
try:
    pred, band, mae, importance = forecast(data, horizon)
    p1, p2, p3 = st.columns(3)
    p1.metric("模型预测累计变化", f"{pred * 100:+.2f}%")
    p2.metric("历史测试集 MAE", f"{mae * 100:.2f}%" if np.isfinite(mae) else "暂无")
    p3.metric("历史误差参考带宽", f"±{band * 100:.2f}%")
    if pred > 0.01:
        st.info("模型倾向：偏强（仅为统计模型输出）")
    elif pred < -0.01:
        st.info("模型倾向：偏弱（仅为统计模型输出）")
    else:
        st.info("模型倾向：震荡或方向不明显")
    st.caption(
        f"模型参考区间：{(pred-band)*100:+.2f}% 至 {(pred+band)*100:+.2f}%。"
        "该区间由历史测试误差构造，不是可靠的概率置信区间。"
    )
    st.write("模型特征重要性（仅表示模型使用程度，不代表因果关系）")
    imp_df = importance.head(10).rename("重要性").reset_index()
    imp_df.columns = ["特征", "重要性"]
    st.bar_chart(imp_df.set_index("特征"))
except Exception as exc:
    st.warning(f"暂时无法生成预测：{exc}")

with st.expander("查看最近行情数据"):
    st.dataframe(data.tail(30).sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)

st.caption("数据来源：AkShare公开行情接口。数据可能延迟、缺失或发生调整。"
           "本程序仅供编程学习和历史数据研究，请勿据此进行投资决策。")
