import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AKSHARE_TOKEN = "b83c781fd62088f1597f5dd244c61e671fff5165042c6cfc5755fb9b"
ak.set_token(AKSHARE_TOKEN)

st.set_page_config(
    page_title="A股智能分析与预测平台",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class ModelFeedback:
    reward: int
    penalty: int
    n_estimators: int


class AdaptiveRandomForest:
    """RandomForest 模型，加入奖励与惩罚后的自适应调参机制。"""

    def __init__(self, n_estimators: int = 120, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.reward_total = 0
        self.penalty_total = 0
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_X = X
        self._train_y = y
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model.fit(X, y)

    def evaluate_and_update(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[np.ndarray, ModelFeedback]:
        if self._train_X is None or self._train_y is None:
            raise ValueError("Model must be fitted before evaluation.")

        preds = self.model.predict(X_test)
        pred_direction = np.sign(preds)
        true_direction = np.sign(y_test)
        reward = int(np.sum(pred_direction == true_direction))
        penalty = int(np.sum(pred_direction != true_direction))

        self.reward_total += reward
        self.penalty_total += penalty

        if penalty > reward:
            self.n_estimators = max(50, self.n_estimators - 10)
        else:
            self.n_estimators = min(220, self.n_estimators + 5)

        combined_X = np.vstack([self._train_X, X_test])
        combined_y = np.hstack([self._train_y, y_test])
        self._train_X = combined_X
        self._train_y = combined_y

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model.fit(combined_X, combined_y)

        feedback = ModelFeedback(
            reward=self.reward_total,
            penalty=self.penalty_total,
            n_estimators=self.n_estimators,
        )
        return preds, feedback

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def format_stock_code(code: str) -> str:
    code = code.strip()
    if not code:
        return code
    if code.startswith("6"):
        return f"sh{code}"
    return f"sz{code}"


@st.cache_data(ttl=60)
def fetch_realtime_board() -> pd.DataFrame:
    return ak.stock_zh_a_spot_em()


def load_realtime_stock(code: str) -> Optional[pd.Series]:
    board = fetch_realtime_board()
    stock = board.loc[board["代码"] == code]
    if stock.empty:
        return None
    return stock.iloc[0]


def load_history_data(
    formatted_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=formatted_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    df = df.rename(columns={"日期": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def compute_factors(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    if df.empty:
        return df

    lookback = 20 if horizon == "短线" else 60
    vol_window = 10 if horizon == "短线" else 30

    factors = df.copy()
    factors["pct_change"] = factors["收盘"].pct_change()
    factors["momentum"] = factors["收盘"].pct_change(periods=lookback)
    factors["volatility"] = (
        factors["pct_change"].rolling(vol_window).std().replace(0, np.nan)
    )
    factors["volume_ratio"] = (
        factors["成交量"] / factors["成交量"].rolling(vol_window).mean()
    )
    if "换手率" in factors.columns:
        factors["turnover"] = factors["换手率"].rolling(vol_window).mean()
    else:
        factors["turnover"] = np.nan
    factors["quality_factor"] = (
        factors["收盘"].ewm(span=lookback, adjust=False).mean() / factors["收盘"]
    )
    factors["value_factor"] = (
        factors["收盘"].rolling(lookback).mean() / factors["收盘"].rolling(lookback).max()
    )
    factors["target_return"] = factors["pct_change"].shift(-1)
    factors = factors.dropna(subset=[
        "momentum",
        "volatility",
        "volume_ratio",
        "quality_factor",
        "value_factor",
        "target_return",
    ])
    return factors


def multi_factor_score(factors: pd.DataFrame) -> pd.DataFrame:
    score_df = factors[[
        "date",
        "momentum",
        "volatility",
        "volume_ratio",
        "quality_factor",
        "value_factor",
    ]].copy()
    for col in ["momentum", "volatility", "volume_ratio", "quality_factor", "value_factor"]:
        col_mean = score_df[col].mean()
        col_std = score_df[col].std(ddof=0) or 1
        score_df[f"z_{col}"] = (score_df[col] - col_mean) / col_std
    z_cols = [c for c in score_df.columns if c.startswith("z_")]
    score_df["multi_factor_score"] = score_df[z_cols].mean(axis=1)
    return score_df


def train_predict_model(
    factor_df: pd.DataFrame,
) -> Tuple[AdaptiveRandomForest, StandardScaler, pd.Series, pd.Series, ModelFeedback]:
    feature_cols = [
        "momentum",
        "volatility",
        "volume_ratio",
        "quality_factor",
        "value_factor",
    ]
    X = factor_df[feature_cols].values
    y = factor_df["target_return"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = AdaptiveRandomForest()
    model.fit(X_train_scaled, y_train)
    preds, feedback = model.evaluate_and_update(X_test_scaled, y_test)

    pred_series = pd.Series(preds, index=factor_df.index[-len(preds) :])
    y_test_series = pd.Series(y_test, index=factor_df.index[-len(y_test) :])

    return model, scaler, pred_series, y_test_series, feedback


def build_prediction_chart(
    factor_df: pd.DataFrame,
    predictions: pd.Series,
    y_true: pd.Series,
) -> go.Figure:
    chart_df = pd.DataFrame({
        "date": factor_df.loc[predictions.index, "date"],
        "predicted_return": predictions.values,
        "actual_return": y_true.loc[predictions.index].values,
    })
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["predicted_return"],
            mode="lines+markers",
            name="预测收益率",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["actual_return"],
            mode="lines+markers",
            name="实际收益率",
        )
    )
    fig.update_layout(
        title="模型预测 vs 实际收益率",
        xaxis_title="日期",
        yaxis_title="收益率",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_investor_advice(prediction: float, horizon: str) -> str:
    threshold = 0.005 if horizon == "短线" else 0.02
    if prediction >= threshold:
        return "建议：买入。模型预计收益率较高。"
    if prediction <= -threshold:
        return "建议：卖出。模型提示潜在风险。"
    return "建议：持有。模型预测收益率波动有限。"


def build_stock_pool(
    price_limit: float,
    horizon: str,
    base_model: AdaptiveRandomForest,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    board = fetch_realtime_board()
    board = board[pd.to_numeric(board["最新价"], errors="coerce") <= price_limit]
    board = board.dropna(subset=["代码"])[:20]

    recommendations: List[Dict[str, float]] = []

    for _, row in board.iterrows():
        code = row["代码"]
        formatted = format_stock_code(code)
        try:
            hist = load_history_data(
                formatted,
                start_date=(dt.datetime.today() - dt.timedelta(days=365 * 2)).strftime("%Y%m%d"),
                end_date=dt.datetime.today().strftime("%Y%m%d"),
            )
            factor_df = compute_factors(hist, horizon)
            if factor_df.empty:
                continue
            latest_features = factor_df.iloc[-1][feature_cols].values.reshape(1, -1)
            latest_scaled = scaler.transform(latest_features)
            predicted_return = float(base_model.predict(latest_scaled)[0])
            mf_score = float(factor_df.iloc[-1]["momentum"] + factor_df.iloc[-1]["value_factor"])
            recommendations.append(
                {
                    "代码": code,
                    "名称": row.get("名称", "-"),
                    "最新价": row.get("最新价", np.nan),
                    "预测收益": predicted_return,
                    "多因子得分": mf_score,
                }
            )
        except Exception:
            continue

    if not recommendations:
        return pd.DataFrame()

    rec_df = pd.DataFrame(recommendations)
    rec_df["综合排名分"] = (
        rec_df["预测收益"].rank(ascending=False) + rec_df["多因子得分"].rank(ascending=False)
    )
    rec_df = rec_df.sort_values("综合排名分")
    return rec_df


def main() -> None:
    st.title("A股智能多因子预测系统")
    st.markdown(
        "本平台整合实时行情、A股特色因子、机器学习预测与自适应调参机制，帮助投资者做出更聪明的决策。"
    )

    with st.sidebar:
        st.header("策略设定")
        stock_code = st.text_input("请输入6位股票代码", value="600519")
        horizon = st.selectbox("投资周期", options=["短线", "长线"], index=0)
        price_limit = st.number_input("股票池价格上限", value=120.0, min_value=1.0, step=1.0)
        start_date = st.date_input(
            "回测开始日期",
            value=dt.date.today() - dt.timedelta(days=365 * 3),
        )
        end_date = st.date_input("回测结束日期", value=dt.date.today())
        analyze_button = st.button("开始分析")

    if not analyze_button:
        st.info("请在左侧输入参数并点击“开始分析”。")
        return

    st.subheader("实时行情监控")
    realtime = load_realtime_stock(stock_code)
    if realtime is None:
        st.warning("未获取到实时行情，请确认股票代码是否正确。")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("最新价", f"{realtime['最新价']}", delta=f"{realtime['涨跌额']}" if '涨跌额' in realtime else None)
        col2.metric("涨跌幅", f"{realtime['涨跌幅']}%")
        col3.metric("换手率", f"{realtime.get('换手率', 'N/A')}%")

    st.subheader("多因子策略分析")
    formatted_code = format_stock_code(stock_code)
    try:
        history_df = load_history_data(
            formatted_code,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
    except Exception as exc:
        st.error(f"历史数据获取失败：{exc}")
        return

    factor_df = compute_factors(history_df, horizon)
    if factor_df.empty:
        st.warning("因子数据不足，无法生成策略。")
        return

    score_df = multi_factor_score(factor_df)
    st.dataframe(
        score_df.tail(10)[
            [
                "date",
                "multi_factor_score",
                "z_momentum",
                "z_volatility",
                "z_volume_ratio",
                "z_quality_factor",
                "z_value_factor",
            ]
        ].set_index("date"),
        use_container_width=True,
    )

    try:
        model, scaler, preds, y_true, feedback = train_predict_model(factor_df)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("模型预测表现")
    chart = build_prediction_chart(factor_df, preds, y_true)
    st.plotly_chart(chart, use_container_width=True)

    latest_prediction = float(preds.iloc[-1])
    st.metric("最新预测收益率", f"{latest_prediction:.2%}")
    st.write(
        f"累计奖励：{feedback.reward}，累计惩罚：{feedback.penalty}，当前树数量：{feedback.n_estimators}"
    )
    st.success(generate_investor_advice(latest_prediction, horizon))

    st.subheader("股票池推荐")
    feature_cols = ["momentum", "volatility", "volume_ratio", "quality_factor", "value_factor"]
    pool_df = build_stock_pool(price_limit, horizon, model, scaler, feature_cols)
    if pool_df.empty:
        st.warning("未能根据条件生成股票池推荐。")
    else:
        st.dataframe(
            pool_df[["代码", "名称", "最新价", "预测收益", "多因子得分"]].head(10),
            use_container_width=True,
        )

    st.caption(
        "*本工具仅用于学习与策略研究，不构成任何投资建议。数据来源：AkShare。"
    )


if __name__ == "__main__":
    main()
