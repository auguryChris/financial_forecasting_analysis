## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/auguryChris/financial_forecasting_analysis/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/auguryChris/financial_forecasting_analysis/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.



Table of Contents
1. Introduction
1.1 Background
1.2 Audience & Motivation
2. Data Acquisition
2.1 Data Sources
3. Methodology
3.1 Data Cleaning / Splitting
3.2 Feature Engineering
3.3 Scaling
3.4 Forecasting Models
4. Results & Discussion
4.1 Forecasting Numerical Results
4.2 Back Testing Results
4.3 Discussion & Failure Analysis
5. Conclusion & Future Direction
    
## 1. Introduction
### 1.1.Background
Financial time series analysis and forecasting have had several approaches over time. Many scholars and teams of professionals have devoted their life’s work to trying to forecast stock prices accurately. The traditional models follow a stochastic probabilistic approach, while more recent models are based on machine learning methods. Currently, the most widely used approaches to forecasting can be roughly divided into two groups: statistical approaches and artificial intelligence approaches. 

The traditional statistical models arose in the beginning of the last century with Yule and Walker, who proposed the autoregressive model (AR)  describing linear relations between past and future values. After that, Whittle showed better results by also considering moving averages, extending the AR model into ARMA. Then, Box and Jenkins proposed a method to estimate ARMA coefficients and they also popularized model variations including seasonality and integrality (ARIMA and SARIMA). More advances on explaining and predicting time series are Engle’s results for modelling time series volatility and Hamilton’s cointegration theory. These approaches were very successful to explain time series based on a priori hypothesis, generating a single model that can be applied to circumstantial situations. However, although statistical approaches improve the accuracy of forecasting stock prices to some extent, the assumption of linearity of stock prices cannot be met according to some recent research, and hence it limits the accuracy. 

More recent models try to focus on learning the behavior of a series from its data, without prior explicit assumptions, such as linearity or stationarity. This approach makes it possible to use machine learning and signal processing techniques with less financial market singularities, for instance market’s liquidity, volatility or efficiency. Therefore, a variety of Machine Learning (ML) and Artificial Intelligence (AI) approaches have been proposed to capture the nonlinearity and nonstationarity of stock prices in the last decades. These models are a hybrid combination of neural network models, traditional models and signal processing. This last element is responsible for decomposing or filtering a time series prior to the model fitting. 
    
### 1.2 Audience Motivation
We conducted an empirical survey of these distinct approaches to forecasting stock prices on a daily scale. Specifically we forecasted using: Auto Regression (AR), ARIMA, SARIMA, Linear Regression, Random Forests, XGBoosted Decision Trees, and various Deep Neural Network LSTM ensembles. We also explored training these models using 363 features that leverage fundamental and technical information. The motivation for this effort is to be able to articulate the nuanced trade-offs and impact that decisions about feature engineering, signal processing, model complexity, model interpretability have on numerical accuracy and the resulting forecast's potential to create value.
    
## 2.Data Acquisition
### 2.1 Data Sources
For our initial dataset, we use yfinance for daily pricing data, and Nasdaq (formerly Quandl) for fundamentals data. We obtain a dataframe with one time series for each of the features, for each of the companies we are working with. The daily pricing data includes 5 columns containing pricing information for each company: Open, High, Low, Close, Volume. This is standard ticker (stock) data that is easily accessible through the yfinance library. The fundamental data is a proprietary dataset that is available through https://data.nasdaq.com/databases/SF1/data, containing 20 years of data with 150 fundamental indicators for over 14,000 companies. For our initial dataset, we gathered 104 fundamental indicators along with the 5 standard price indicators.  We take the fundamental and yfinance data and join the data based on the current S&P 500 companies to reduce the size. 


## 3. Methodology
### 3.1 Data Cleaning & Splitting
**Splitting the data**
The test-train split for modeling was 80-10-10, representing 80% (4336 days) training data, 10% (542 days) validation data, and 10% (542 days) final holdout test data. The decision to leave cross validation out was due to the relatively low number of data instances compared to using higher frequency data. If using higher frequency data with a similar target, we would be able to benefit from a windowed time series cross validation and achieve more robust results. Since we only tested results on the last 20% of the data, our models may perform worse if there is a regime change. In other words, if the distribution of the features drastically changes during a new time period, our current models would not be robust to those new distributions.
    
### 3.2 Feature Engineering
***Feature Engineering to Create Technical Features***
    
In order to create additional features on top of our core pricing and fundamental data, we used a Python technical analysis library, https://github.com/bukosabino/ta, to help engineer our remaining features. We used rolling windows of 6, 18, 24, 30, 50, 100, and 200 days for each technical indicator in our list. In total, we created an additional 254 technical features, which amounted to 363 combined price features that reflected volatility, momentum, volume, and trend to use for modeling. We trained each model on the same length of data instances, which is 5419 trading days in total. 
    
***Technical Feature Descriptions***
Relative Strength Index (RSI) momentum indicator
    
Relative Strength Index is a momentum indicator that tracks the magnitude of recent price changes. The recency component, as well as the ability to include magnitude, means the indicator can evaluate overbought/oversold conditions quickly, despite being a lagging indicator. The indicator is an oscillator (ranges between two values) and often has cutoffs at 80% (overbought) and 20% (oversold), though you may find these as a 70/30 split as well. Our combination list iterated over the lookback period length.
(More information: https://www.investopedia.com/terms/r/rsi.asp )
    
***MACD trend-following momentum indicator***
    
MACD (Moving Average Convergence Divergence) is a lagging trend-following momentum indicator. This indicator leverages two different moving averages for a ticker’s price, where one is slower and one is faster, and the difference is taken between the two to determine the MACD line. There is also a signal line, typically faster than the faster of the moving averages, which acts as an indicator for buys and sells. When the MACD line passes above the signal line, the security is considered a Buy. Our combination list iterated over the short and long lookback period lengths,  where the faster is less than the slower length. Our feature metric is the difference between these two, for some normalization around 0.
(More information: https://www.investopedia.com/terms/m/macd.asp )
    
***Bollinger Bands (BB) volatility indicator***
    
Bollinger Bands are a volatility indicator that leverages standard deviations of volatility to identify two bands - one above and one below the SMA for price movement - for what would be reasonable volatility ranges. This is a way to identify overbought/oversold conditions (where above the higher band is overbought, below the lower band is oversold). In more volatile markets, bands widen; in less volatile markets bands contract. Our combination list iterated over the lookback period length, a traditional, uniform 2 standard deviations were used throughout all iterations, and the indicator returned both band values, as well as a binary 0/1 for both high and low bands as to whether the price movement had crossed it, providing a normalized value across securities.
(More information: https://www.investopedia.com/terms/b/bollingerbands.asp )
    
***On-Balance Volume (OBV) volume indicator***
    
On-Balance Volume is a technical momentum indicator that uses volume to predict stock price changes. Volume can be a strong indicator of momentum, in that significant volume flows and changes can mean a material price change with “stickiness”. If volume increases significantly with no immediate stock price chance, that means that the change is likely about to happen. Based on today’s closing price in relation to yesterday’s closing price, we can also assign direction to this indicator (bearish/bullish). OBV does not require a window, as it is a running summation of volume.
(More information: https://www.investopedia.com/terms/o/onbalancevolume.asp )
    
***Average True Range (ATR) volatility indicator***
    
Average True Range is a volatility indicator that decomposes the price range movement for a lookback time period. The ATR is a moving average of the true ranges, defined as the maximum of three different range-related values. Our combination list iterated over the lookback period length of this moving average window.
(More information: https://www.investopedia.com/terms/a/atr.asp )
    
***Stochastic Oscillator (SR) momentum indicator***
    
A stochastic oscillator is a momentum indicator that compares closing price to a factor of high and low price ranges over a lookback window. The indicator is an oscillator, indicating it is range-bound, bouncing between a range of 0-100. It specifically indicates overbought/oversold conditions. Our combination list iterated over the lookback period length.
(More information: https://www.investopedia.com/terms/s/stochasticoscillator.asp )
    
***Volume-Price Trend (VPT) volume indicator***
    
Volume-Price Trend is a volume indicator that helps identify price direction and the magnitude of that movement. It is specifically used to determine supply and demand of a security. Like On-Balance volume, the VPT consists of a cumulative volume trend line, though leveraging a multiple of the percentage change as opposed to flat volume count, as well as a smoothed moving average of that line. VPT does not require a window.
(More information: https://www.investopedia.com/terms/v/vptindicator.asp )

***Accumulation/Distribution Index (ADI) volume indicator***
    
Similarly to Volume-Price Trend, ADI is a volume indicator that uses a money flow multiplier and volume to determine supply and demand of a security. The money flow multiplier is a percentage derived using differences in high, low, and close prices for the period. ADI does not require a window.
(More information: https://www.investopedia.com/terms/a/accumulationdistribution.asp )

***Chaikin Money Flow (CMF) volume indicator***
    
Chaikin Money Flow is a volume indicator that is an oscillator. It operates under the assumption that values above zero indicate strength, and values below zero indicate weakness in a security’s price. The CMF is similar to ADI, except that it leverages a lookback window to create a moving average and generates a signal line with this. Our combination list iterated over the lookback period length of the moving average window.
(More information: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf )
    
***Ichimoku Kinko Hyo trend and momentum indicator***
    
Ichimoku is a trend and momentum indicator that is used to determine support and resistance levels. There are five parts to the Ichimoku Kinko Hyo, each determining something important about the stock’s movements. These lines create a combination of moving averages of highs, lows, closes, divergences, and convergences that generate insight to lines of support and resistance, as well as crossovers. Our combination list iterated over the three possible window lookback period lengths,  where the faster is less than middle, is less than the slower length. Our feature metrics related to Ichimoku are the values of the conversion lines, values of the base lines, and the differences between those two for some normalization around 0.
(More information: https://www.investopedia.com/terms/i/ichimokuchart.asp )
    
***Donchian Channel volatility indicator***
    
The Donchian Channel is a volatility indicator that comprises three lines, where one is a moving average and the other two are bands. The upper band is the highest price of a security over a lookback period, and the lower band is the lowest price of a security during that same period. A breakthrough in either direction indicates strong volatility and momentum in those directions. Our combination list iterated over the lookback period length.
(More information: https://www.investopedia.com/terms/d/donchianchannels.asp )
    
***Simple Moving Average (SMA)***
    
A simple moving average is a trend indicator, and is just the arithmetic mean of the price over a particular lookback period.
(More information: https://www.investopedia.com/terms/m/movingaverage.asp )
    
***Exponential Moving Average (EMA) trend indicator***
    
An exponential moving average is a trend indicator, and is a weighted average of the price over a particular lookback period that takes recency into account, and is thus more responsive to new information in pricing changes. 
(More information: https://www.investopedia.com/terms/m/movingaverage.asp )
    
***Fundamentals (General)***
   
Fundamentals are a category of metrics that are not necessarily indicators for pricing movements, but can have an influence on price. Fundamentals are taken from the company’s public reporting documents, including their income statements, cash flow statements, and balance sheets. There are metrics in our raw dataset that are nominal for each company, but some ratios, such as P/E, P/B, etc.,  that are also normalized in some capacity to compare to other companies, despite significant changes in the value of a dollar over the 20 year period. 

The inclusion of fundamentals alongside technical indicators is the hope that aspects of the company’s underlying workings and fundamental value may lend some insight into its pricing and movements. The difficulty here is that these reports come out quarterly, and some stats are updated yearly, while technical indicators are different daily. So, this slower updating of information and subsequent feed-through to price may lead this subset to have less feature importance against a target with daily movement.
    
***Signal Processing: Decomposition Features***
As mentioned, in the last decade there has been a focus on a hybrid combination of neural network models and signal processing to capture the nonlinearity and nonstationarity of stock prices. Here the original time series is decomposed into several components. We leveraged the Complete Ensemble Empirical Mode Decomposition with Adaptive Noise ( CEEMDAN) to decompose our nonlinear, nonstationary sequences into intrinsic mode functions (IMFs) in the spirit of the Fourier series. In contrast to the Fourier series, they are not simply sine or cosine functions, but rather functions that represent the characteristics of the local oscillation frequency of the original data.


<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~augurychris/1.embed"></iframe>
