# [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)
In this project, the objective is to build models that will predict short-term volatility for hundreds of stocks across different sectors. In the given dataset from the Kaggle Competition, the available columns are as shown below. 

|Column| Description | Data Type|
|---|---|---|
| "time_id"| ID code for the time bucket| Integer|
| "seconds_in_bucket"| Number of seconds from the start of the bucket, always starting from 0| Integer|
| "bid_price1"| Normalized price of the most competitive buy level| Float|
| "ask_price1"|Normalized price of the most competitive sell level | Float|
| "bid_price2"|Normalized price of the second most competitive buy level| Float|
| "ask_price2"|Normalized price of the second most competitive sell level |Float|
| "bid_size1"|The number of shares on the most competitve buy level | Integer|
| "ask_size1"|The number of shares on the most competitve sell level | Integer|
| "bid_size2"|The number of shares on the second most competitve buy level | Integer|
| "ask_size2"|The number of shares on the most competitve sell level |Integer|
| "stock_id"| ID code for the stock | Category|

There are 11 columns and 167,253,289 rows.

### Data Exploration adn Feature Engineering
Based on the available columns, Weighted Averaged Price (WAP) was calculated for instaneous stock valuation which . 

WAP equation is as shown below:

WAP = ![equation](https://latex.codecogs.com/svg.image?\frac{BidPrice_{1}&space;*&space;AskSize_{1}&space;&plus;&space;AskPrice_{1}&space;*&space;BidSize_{1}}{BidSize_{1}&space;&plus;&space;AskSize_{1}})

In order to compare price differences across different stocks, log transformation is applied on WAP in "log_return" column. NaN values are then replaced with "0".

Due to the large amount of dataset, the notebooks are hosted in Google Cloud Platform Vertex AI. Links are in the title of the models.

### Models

[Deep Learning Models](https://1843eabbf1f5bf19-dot-asia-southeast1.notebooks.googleusercontent.com/doc/workspaces/auto-f/tree/imported/optiver-dl-c56403b5-6407-4ee5-8145-a7fa8c6021ed.ipynb)

For Deep Learning Models, Simple Recurrent Neural Networks (RNN) and Long Short Term Memory (LSTM) were trained and tested with various different parameters such as batch size, cell units, epochs, interval and optimizer.

It would appear that LSTM is a better performing model in this case according to the scores below. The parameters that give the best results are batch size of 10, cell units of 100, epochs of 100, interval of 100 and the optimizer is adam. It gives the best R2 score of -1.757 and a good Root Mean Squared Percentage Error (RMSPE) value of 2.8.

[Moving Average](https://5356215ab0a9d461-dot-asia-southeast1.notebooks.googleusercontent.com/doc/tree/imported/optiver-MA-afb0616f-4b5e-4656-bf58-fab142aa4d7e.ipynb)

For the Moving Average model, the predicted value is the average of previous 100 values. The result of the model is pretty good with a R2 score of -1.254 and RMSPE value of 0.873.

[Linear Regression Model](https://7f4f0e86daf2d949-dot-asia-southeast1.notebooks.googleusercontent.com/doc/tree/imported/optiver-linearregression-11686072-750c-42d0-8bf9-69bd29ad9b1f.ipynb)

Simple Linear Regression model was also compared with Ridge Regression cross-validation and Lasso cross-validation. All 3 models produced the same values. However, when the 3 models were tested with different test size of 0.2 and 0.3, a test size of 0.3 performed slightly better with a R2 score of -1.7466182328 and RMSPE of 1.0000087618.

Based on the different models above, below is a summary of R2 scores and RMSPE. From the table, it appears out of all the models tested, Moving Average is the best one so far with lower R2 Score and RMSPE value. 

|Models| R2 Score | RMSPE|
|---|---|---|
|Simple RNN| -193.676| 19.711|
|LSTM|  -1.757|  2.8|
|Moving Average| -1.254| 0.873|
|Linear Regression| -1.7466182328| 1.0000087618|

### Key Findings and Insights/ Future Works
More could be done to fine tune the parameters in each model. 

For the Moving Average moel, further fine tuning of the number of samples in the moving average could potentially improve the model. Perhaps a graph could be plotted with number of samples against R2 scores and RMSPE.

Whereas for the Linear Regression model, perhaps the test size or the way to train and test the model could be further investigated to improve it.

Deep Learning Models do not seem to be a good fit for this case. 

If possible, additional features could be added such as a rated public sentiment of the particular stock since stock prices could be affected by public opinions.

## Reference
[Introduction to Financial Concepts and Data](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data)
