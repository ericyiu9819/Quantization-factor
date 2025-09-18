# Quantization Factor Toolkit

该项目汇总了可用于 A 股量化分析的关键组件和工具。当前版本包含：

- `src/a_share_data_apis.py`：整理了主流可用的 A 股数据获取 API，并提供东方财富公开接口与 Tushare Pro 的示例调用方法，可直接集成到量化研究代码中。

## 使用说明

1. 在命令行运行 `python -m src.a_share_data_apis` 可查看可用数据源列表，并尝试调用东方财富公开 API 获取样例 K 线数据。
2. 若需要访问 Tushare Pro 数据，请先安装官方 SDK 并配置个人 Token：

   ```bash
   pip install tushare
   export TUSHARE_TOKEN="你的token"
   ```

   然后在代码中调用 `fetch_daily_price_from_tushare` 即可获取指定区间的日行情。

3. 对于其他商业数据供应商（聚宽、米筐等），请根据其官方文档使用账号与授权信息发起请求。

## 免责声明

请在遵循各数据供应商的使用条款及法律法规的前提下访问和使用市场数据。
