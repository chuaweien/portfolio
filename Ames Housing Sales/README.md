# Ames Housing Sales
In the Ames Housing Sales data, there are total of 1,379 rows and 80 columns. Sale Price will be the target variable we are predicting, and the rest of the variables are features for the model.

### Data Cleaning and Feature Engineering
1. Check for any null values in the columns. If there is, either fill in the missing values with median, or drop the feature from the model if there are too many null values.

But this dataset doesn't have any null values. So, we move on to the next step.

2. Drop the features with all unique values because they do not add value to the model.

This dataset does not have any feature with all unique values.

3. Check for string categorical variables and use One-hot encoding to convert all string categorical variables to binary dummies.

There are 43 string categorical variables, and it needs to be converted into binary dummies. We will be using One-hot encoding method to convert them and adding 215 more columns. After converting, we now have 1,379 rows and 252 columns.

4. Check the skew values of columns that contain float types. If the absolute skew value is above 0.75, log transformation needs to be performed to convert the distribution into normal distribution.

After the skew check, there are 19 columns which have an absolute skew value of above 0.75. One example is shown below (Figure 1) for ‘1stFlrSF’. It shows the before and after log transformation. This shall be done to all the 19 columns.

![Figure 1: Before and After Comparison of Log Transformation for 1stFlrSF](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%201.png)
<br>Figure 1: Before and After Comparison of Log Transformation for 1stFlrSF

### Linear Regression Models
#### Simple Linear Regression
Using Simple Linear Regression as baseline, first we did a StandardScaler to standardise the scales of all variables. We then plot a graph of Actual Sale Price vs Predicted Sale Price as can be seen below in Figure 2. There were a few outliers which cause the x-axis of the graph to be too huge. So, we ignore the outliers and restrict the x-axis scale as shown in Figure 3. We can observed there is linear regression between the Actual Sale Price and the Predicted Sale Price.
![Figure 2 Actual Sale Price vs Predicted Sale Price for Linear Regression](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%202.png)
<br>Figure 2 Actual Sale Price vs Predicted Sale Price for Linear Regression

![Figure 3 Actual Sale Price vs Predicted Sale Price for Linear Regression (with smaller x-axis)](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%203.png)
<br>Figure 3 Actual Sale Price vs Predicted Sale Price for Linear Regression (with smaller x-axis)

#### Polynomial Transformation
We assumed a polynomial degree of 2 and plotted a graph as shown in Figure 4. Due to the polynomial transformation, the outliers are not skewing the graph axes.
![Figure 4 Actual Sale Price vs Predicted Sale Price for Polynomial Feature](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%204.png)
<br>Figure 4 Actual Sale Price vs Predicted Sale Price for Polynomial Feature

#### Regularization Regression
##### Ridge CV
The alphas range used was [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80], and the alpha used was 10.0. The graph generated can be seen in Figure 5.
<br>![Figure 5 Actual Sale Price vs Predicted Sale Price for Ridge Regression](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%205.png)
<br>Figure 5 Actual Sale Price vs Predicted Sale Price for Ridge Regression

##### Lasso CV
The alphas range used was np.array([1e-5, 5e-5, 0.0001, 0.0005]), and the alpha used was 0.0005. The graph generated can be seen in Figure 6.
<br>![Figure 6 Actual Sale Price vs Predicted Sale Price for Lasso Regression](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%206.png)
<br>Figure 6 Actual Sale Price vs Predicted Sale Price for Lasso Regression

![Figure 7: RSME table of models](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%207.png)
<br>Figure 7: RSME table of models

Then we calculate the Root Squared Mean Error of each model and compare them as shown in the table above (Figure 7). Ridge has the least Root Squared Mean Error which shows that it is the best model in this case. Also, as can be seen from the 4 graphs, Ridge has the least outliers from the linear trend.

### Key Findings & Insights

![Figure 8: Features and their coefficients (sorted from highest to lowest)](https://github.com/cweien3008/portfolio/blob/main/Ames%20Housing%20Sales/Picture%208.png)
<br>Figure 8: Features and their coefficients (sorted from highest to lowest)

Using the Ridge CV model, out of 251 features, 10 of them have zero coefficients and 241 of them have non-zero coefficients. After sorting the values in ascending order, from Figure 8, Neighborhood_14, GrLiveArea, RoofMatl_6, 1stFlrSF and Neighborhood_21 are the top 5 features that have the most positive impact on the Sale Price. Whereas, BsmtQual_3, KitchenQual_2, KitchenQual_1, BmstQual_1 and PoolQC_1 are the top 5 features with the most negative impact on Sale Price.

### Future Works
For future works, we could revisit the model and tune some of the parameters for a better model. Perhaps, we could also remove some of the outliers and compare the results to see if the model improves.  
Alternatively, we could also gather more data for analysis. With more data, we would be able to get better insights and model better.
