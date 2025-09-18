"""Utility module describing accessible A-share market data APIs and providing
helper functions to fetch data from public and commercial providers.

The module centralises API metadata so the quant research platform can
programmatically discover which services to integrate. It also includes example
functions that demonstrate how to request data from a public REST endpoint and a
professional data provider (Tushare Pro).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import importlib
import json
from typing import Dict, Iterable, List, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


@dataclasses.dataclass(frozen=True)
class AshareAPI:
    """Metadata describing an A-share data service."""

    name: str
    provider: str
    base_url: str
    data_types: Iterable[str]
    authentication: str
    notes: str

    def to_dict(self) -> Dict[str, str]:
        """Return a dictionary representation convenient for serialization."""

        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "data_types": ", ".join(self.data_types),
            "authentication": self.authentication,
            "notes": self.notes,
        }


def list_available_apis() -> List[AshareAPI]:
    """Return curated A-share data API options.

    The list mixes 免费公共数据源 and 付费专业数据供应商, covering 行情、资金面、基本面
    以及衍生指标, so users can decide what to integrate based on需求与预算.
    """

    return [
        AshareAPI(
            name="Tushare Pro",
            provider="Tushare",
            base_url="https://api.tushare.pro",
            data_types=(
                "日/分钟行情",
                "财务报表",
                "资金流向",
                "指数及期货",
                "宏观经济",
            ),
            authentication="Token（需注册并订阅相应积分套餐）",
            notes=(
                "官方 Python SDK，可用于策略回测和实时查询；数据覆盖面广，"
                "支持字段筛选与分页请求。"
            ),
        ),
        AshareAPI(
            name="聚宽数据服务 (JoinQuant)",
            provider="聚宽 (JoinQuant)",
            base_url="https://dataapi.joinquant.com",
            data_types=(
                "日/分钟级行情",
                "基本面与财务",
                "资金流、龙虎榜",
                "期权期货",
                "多空情绪指标",
            ),
            authentication="账号 + Token（需订阅付费套餐）",
            notes=(
                "提供回测与实盘环境；Python SDK 支持直接在策略中调用，"
                "适合专业量化团队。"
            ),
        ),
        AshareAPI(
            name="米筐数据 (Ricequant RQData)",
            provider="米筐 (Ricequant)",
            base_url="https://rqdata.ricequant.com",
            data_types=(
                "多频行情",
                "财务/因子",
                "期权期货",
                "指数成分",
            ),
            authentication="账号 + License（付费）",
            notes=(
                "与 rqalpha 框架无缝对接，支持大规模历史数据下载，"
                "可用于高频回测。"
            ),
        ),
        AshareAPI(
            name="东方财富行情接口",
            provider="东方财富网",
            base_url="https://push2.eastmoney.com",
            data_types=(
                "实时/延迟行情",
                "K 线数据",
                "板块热点",
            ),
            authentication="无需认证（公开接口，注意访问频次限制）",
            notes=(
                "可用于快速获取公开行情，适合原型验证；需自行处理字段含义"
                "和限频策略。"
            ),
        ),
        AshareAPI(
            name="新浪行情 API",
            provider="新浪财经",
            base_url="https://hq.sinajs.cn",
            data_types=("实时快照", "分笔成交"),
            authentication="无需认证（公开接口，频率有限）",
            notes="轻量级行情获取方式，适合做实时监控或验证。",
        ),
    ]


def fetch_recent_quotes_from_eastmoney(
    secid: str,
    count: int = 10,
    klt: int = 101,
    fields1: str = "f1,f2,f3,f4,f5,f6",
    fields2: str = "f51,f52,f53,f54,f55,f56,f57",
    timeout: int = 10,
) -> Dict[str, object]:
    """Fetch recent K-line data from the Eastmoney public API.

    Args:
        secid: 市场+代码，例如上证主板 1.600519，深证 0.000001。
        count: 请求的K线数量。
        klt: 周期代码（101=日线, 5=5分钟等）。
        fields1: 返回的指标字段集合1。
        fields2: 返回的指标字段集合2。
        timeout: HTTP 请求超时时间（秒）。

    Returns:
        API 返回的 JSON 解析结果。

    Raises:
        urllib.error.URLError: 请求失败时抛出。
        ValueError: 返回内容不是合法 JSON 时抛出。
    """

    params = {
        "secid": secid,
        "fields1": fields1,
        "fields2": fields2,
        "klt": klt,
        "fqt": 1,  # 前复权
        "end": _dt.datetime.now().strftime("%Y%m%d"),
        "lmt": count,
    }
    query = urlparse.urlencode(params)
    url = f"https://push2.eastmoney.com/api/qt/stock/kline/get?{query}"

    with urlrequest.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def fetch_daily_price_from_tushare(
    ts_code: str,
    start_date: str,
    end_date: str,
    token: Optional[str] = None,
):
    """Fetch daily price data using the Tushare Pro Python SDK.

    Args:
        ts_code: Tushare 证券代码，如 "600519.SH"。
        start_date: 起始日期，格式 YYYYMMDD。
        end_date: 结束日期，格式 YYYYMMDD。
        token: 可选的 Tushare token（若已在环境变量 TUSHARE_TOKEN 中配置，可省略）。

    Returns:
        包含行情数据的 pandas.DataFrame。

    Raises:
        RuntimeError: 当 Tushare SDK 未安装或 token 缺失时抛出。
    """

    if importlib.util.find_spec("tushare") is None:
        raise RuntimeError(
            "未安装 tushare，请先执行 'pip install tushare' 后再调用该函数。"
        )

    tushare = importlib.import_module("tushare")
    pro = tushare.pro_api(token)
    data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return data


def _demo() -> None:
    """Print available APIs and try to fetch sample public data."""

    print("可用的 A 股数据 API：")
    for api in list_available_apis():
        print("- {name} ({provider}) => {base_url}".format(**api.to_dict()))

    try:
        sample = fetch_recent_quotes_from_eastmoney("1.600519", count=2)
        klines = sample.get("data", {}).get("klines", [])[:2]
        print("\n东方财富日线样例：", klines)
    except (urlerror.URLError, ValueError) as exc:
        print("无法访问东方财富行情接口：", exc)


if __name__ == "__main__":
    _demo()
