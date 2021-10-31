# CO2 Level (ppm) in Mauna Loa, Hawaii
The objective is to build a model that correctly predicts CO2 level (ppm) in Mauna Loa, Hawaii. We will be using both Deep Learning and SARIMA models for comparison. Once the model is ready, stakeholders could potentially use it to predict future CO2 level and prepare their residents for it.

The data set contains monthly CO2 level (ppm) in Manua Loa, Hawaii from 1965 to 1980. With reference to Figure 1, it shows that the data set contains 192 rows and 2 columns, one column being the month and the other column being the CO2 level.
 
![Figure 1 Dataframe](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%201.png)\
Figure 1: Dataframe

By running .info() command, we can observe the type of data and number of null values in the dataframe. As shown in Figure 2, both columns are string values and there are no null values.
 
 ![Figure 2 Information of the data frame](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%202.png)\
Figure 2: Information of the data frame

### Data Exploration and Data Cleaning
Since both columns are strings, we need to convert them. First, we convert ‘CO2 (ppm) mauna loa, 1965-1980’ to float. Then, convert ‘Month’ to datetime. The resulting dataframe is shown as Figure 3.

![Figure 3 Resulting Dataframe](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%203.png)\
Figure 3: Resulting Dataframe					

<img src="https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%204.png" width="250">
Figure 4: Time VS CO2 Graph

We then plot the CO2 level against time in Figure 4. As can be seen from Figure 4, the data is not stationary as mean and variance changes over time. From the graph, we can observe that the trend is increasing with seasonality and residue components.

To confirm this hypothesis, we will be performing Augmented Dickey-Fuller Test (ADF). The results are as follows in Figure 5 which shows that the data is non-stationary with a p-value of 0.996.
 
![Figure 5 ADF test](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%205.png)\
Figure 5: ADF test

<img src="https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%206-1.png" width="250">
<img src="https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%206-2.png" width="250">
<img src="https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%206-3.png" width="250">
Figure 6: Decomposing the data

We also decomposed the data into 3 different graphs: Trend, Seasonality and Residue as shown in Figure 6. 

### Models
#### Simple RNN Model
For the first model, we try to fit the dataset into a simple RNN model. Before that, we split the dataset into train and test sets. We set the length of held-out terminal sequence to be 100 and the sequence length to be 12. With that, we will have the following training and test sets:
Training input shape: (80, 12, 1)
Training output shape: (80,)
Test input shape: (12,)
Test output shape: (88,)

We then fit the train set into the simple RNN model with batch size of 5, cell units of 100 and epochs of 5000. We will have a model as shown below in Figure 7. The Simple RNN model is then used to predict using the test data as shown in Figure 8.

![Figure 7 Simple RNN Model Summary](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%207.png)\
Figure 7: Simple RNN Model Summary			 

![Figure 8 Test Data and RNN Predictions](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%208.png)\
Figure 8: Test Data and RNN Predictions

#### LSTM Model
With the same train and test sets, we use them to fit into a LSTM model with batch size of 5, cell units of 100 and epochs or 3000. It will generate a model summary as shown in Figure 9. We then use the LSTM model to predict for the test set as shown in Figure 10.
           
![Figure 9 LSTM Model Summary](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%209.png)\
Figure 9: LSTM Model Summary  			

![Figure 10 Test Data and LSTM Predictions](https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%2010.png)\
Figure 10: Test Data and LSTM Predictions

#### SARIMA
For the third model, we used SARIMA model with an order of (0,1,0) and seasonality of (0,1,1,12) to fit the dataframe. Figure 11 was generated from the model.
 
<img src="https://github.com/cweien3008/portfolio/blob/main/CO2%20level%20(ppm)/Figures/Picture%2011.png" width="250">
Figure 11: Test Data and SARIMA Predictions

As can be seen from the various graphs above, SARIMA is the best model out of the 3 because the predictions are the closest to the actual values and trend. Perhaps we do not have sufficient data to build a better Deep Learning models, maybe that is why they are not very accurate. 

### Key Findings and Insights
Initially, we were trying with 500 epochs and the predictions values were around 200 which are very far away from the actual values for both Simple RNN and LSTM models. However, once we increased the epochs to 5000 or 3000, the predicted values are slightly closer to the actual values. Therefore, it shows that we need more epochs for the model to improve. However, I do think that with limited amount of data, there is a limit on how much the model can improve. 

### Future Works
We could gather more data for the models, or increase the number of epochs, or try out different hyperparameters (i.e. batch size, cell units, sequence length, number of train and test sets, etc). As we continue to tune the hyperparameters, the models get better as mentioned in the previous section.
Perhaps instead of monthly CO2 level data, we could try the model with daily CO2 data for accurate predictions and this would give us more data points to improve the model.
