# Portfolio
Portfolio of personal projects for self-learning and hobby. These projects are written in Python (Jupyter Notebook).

Please feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/wei-en-chua/).

## Projects
#### [Abalone](https://github.com/cweien3008/portfolio/tree/main/Abalone)
The objective is to focus on clustering the data set to find similarity so that it can be used to predict the age of the abalone.

#### [Ames Housing Sales](https://github.com/cweien3008/portfolio/tree/main/Ames%20Housing%20Sales)
Objective of this project is to analyse the dataset and to use Linear Regression models to predict Sale Price. Through this exercise, we could potentially see which feature has more impact than the others.

#### [CO2 Level](https://github.com/cweien3008/portfolio/tree/main/CO2%20level%20(ppm))
The objective is to build a model that correctly predicts CO2 level (ppm) in Mauna Loa, Hawaii. We will be using both Deep Learning and SARIMA models for comparison. Once the model is ready, stakeholders could potentially use it to predict future CO2 level and prepare their residents for it.

#### [Virgin America Airline](https://github.com/cweien3008/portfolio/tree/main/Virgin%20America%20Airline)
The main objective is to analyse tweets mentioning Virgin America Airline using Long-Short Term Memory of Recurrent Neural Networks. This will help to train the model to understand what the tweets are about and perhaps could be used for text generation in the future to answer those tweets.

### Using Kaggle Datasets
#### [Titanic](https://github.com/cweien3008/portfolio/tree/main/Titanic) 
Objective of this analysis is to predict the survival of passengers from the sinking Titanic based on the features given in the dataset.

#### [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)
In this project, the objective is to build models that will predict short-term volatility for hundreds of stocks across different sectors. In the given dataset, the available columns are: "time_id", "seconds_in_bucket", "bid_price1", "ask_price1", "bid_price2", "ask_price2", "bid_size1", "ask_size1", "bid_size2", "ask_size2", "stock_id". Based on the available columns, Weighted Averaged Price (WAP) was calculated for stock valuation and future prediction. 

WAP equation is as shown below:

WAP = ![equation](https://latex.codecogs.com/svg.image?\frac{BidPrice_{1}&space;*&space;AskSize_{1}&space;&plus;&space;AskPrice_{1}&space;*&space;BidSize_{1}}{BidSize_{1}&space;&plus;&space;AskSize_{1}})

Due to the large amount of dataset, the notebooks are hosted in Google Cloud Vertex AI.

[Deep Learning Models](https://1843eabbf1f5bf19-dot-asia-southeast1.notebooks.googleusercontent.com/doc/workspaces/auto-f/tree/imported/optiver-dl-c56403b5-6407-4ee5-8145-a7fa8c6021ed.ipynb)

For Deep Learning Models, Simple Recurrent Neural Networks (RNN) and Long Short Term Memory (LSTM) were trained and tested with various different parameters.

[Moving Average](https://5356215ab0a9d461-dot-asia-southeast1.notebooks.googleusercontent.com/doc/tree/imported/optiver-MA-afb0616f-4b5e-4656-bf58-fab142aa4d7e.ipynb)

For Moving Average model, the predicted value is the average of previous 100 values. 

[Linear Regression Model](https://7f4f0e86daf2d949-dot-asia-southeast1.notebooks.googleusercontent.com/doc/tree/imported/optiver-linearregression-11686072-750c-42d0-8bf9-69bd29ad9b1f.ipynb)

Linear Regression model was also compared with Ridge Regression cross-validation and Lasso cross-validation. All 3 models produced the same values. 

Based on the different models above, below is a summary of R2 score and Root Mean Squared Percentage Error (RMSPE). From the table, it appears out of all the models tested, Moving Average is the best one so far with lower R2 Score and RMSPE value. 

|Models| R2 Score | RMSPE|
|---|---|---|
|Simple RNN| -193.676| 19.711|
|LSTM|  -4.037|  2.036|
|Moving Average| -1.254| 0.873|
|Linear Regression| -1.7466182328| 1.0000087618|


#### NFL
### [Advent of Code 2015](https://adventofcode.com/2015)
