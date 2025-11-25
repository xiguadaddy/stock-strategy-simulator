# Stock Strategy Simulator & Optimizer / 股票策略模拟与优化器

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🇬🇧 English

### Overview
This project is a comprehensive stock trading strategy simulator designed to help users visualize and optimize their investment strategies. It uses Monte Carlo simulations to project future price paths and tests various trading strategies against these scenarios.

### Features
- **Interactive UI**: Built with Streamlit for a seamless user experience.
- **Multi-Stock Support**: Quickly select popular tech stocks (TSLA, NVDA, GOOGL, etc.) or enter any custom ticker.
- **Advanced Prediction Engines**:
    - **Linear Trend**: Regression-based prediction.
    - **CAGR**: Historical growth projection.
    - **Analyst Consensus**: Wall Street target prices.
    - **AI Technical Agent**: Smart prediction based on RSI, MACD, and momentum.
- **Strategy Simulation**:
    - **Buy & Hold**: Baseline comparison.
    - **Grid Trading**: Profiting from volatility.
    - **Rebalancing**: Fixed asset allocation.
    - **Custom Threshold**: Define your own dynamic buy/sell rules (e.g., "Buy on 5% dip, Sell on 5% rally").
- **Internationalization**: Full support for English and Chinese interfaces.

### Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   To make the server accessible from any IP address (useful for remote access):
   ```bash
   python -m streamlit run app.py --server.address 0.0.0.0
   ```
   
   Or use the VS Code Task: `Run Streamlit App`

3. **Access**:
   Open your browser at `http://localhost:8501` (or your machine's IP address).

---

<a name="chinese"></a>
## 🇨🇳 中文 (Chinese)

### 项目概述
本项目是一个功能全面的股票交易策略模拟器，旨在帮助用户可视化并优化其投资策略。它利用蒙特卡洛模拟来预测未来的股价走势，并在这些情景下测试各种交易策略的表现。

### 主要功能
- **交互式界面**: 基于 Streamlit 构建，操作流畅。
- **多股票支持**: 快速选择热门科技股 (TSLA, NVDA, GOOGL 等) 或输入任意自定义代码。
- **高级预测引擎**:
    - **线性趋势**: 基于回归分析的预测。
    - **历史增长率 (CAGR)**: 基于历史动量的预测。
    - **分析师一致预期**: 华尔街目标价。
    - **AI 技术分析代理**: 基于 RSI、MACD 和动量的智能预测。
- **策略模拟**:
    - **买入持有**: 基准策略对比。
    * **网格交易**: 利用震荡行情获利。
    - **动态平衡**: 固定资产配置比例。
    - **自定义阈值策略**: 定义您自己的动态买卖规则（例如：“跌5%买入，涨5%卖出”）。
- **国际化支持**: 完美支持中文和英文界面切换。

### 安装与使用

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **运行应用**:
   若要使服务器对所有 IP 地址开放（便于远程访问）：
   ```bash
   python -m streamlit run app.py --server.address 0.0.0.0
   ```
   
   或者使用 VS Code 任务: `Run Streamlit App`

3. **访问**:
   在浏览器中打开 `http://localhost:8501` (或您机器的 IP 地址)。
