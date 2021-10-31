# Abalone
The objective is to focus on clustering the data set to find similarity so that it can be used to predict the age of the abalone.

This set of data was taken from UCI Machine Learning Repository with the link [here](https://archive.ics.uci.edu/ml/datasets/Abalone). This data studies the physical attributes of abalones to predict its age. The physical attributes included in this data are as shown below with data type, measurement units and description:

Name | Data Type | Meas. Unit | Description
---|---|---|---|
Sex|	nominal	| |	M, F, and I (infant)
Length|	continuous|	mm|	Longest shell measurement
Diameter|	continuous|	mm|	perpendicular to length
Height|	continuous|	mm|	with meat in shell
Whole weight|	continuous|	grams|	whole abalone
Shucked weight|	continuous|	grams|	weight of meat
Viscera weight|	continuous|	grams|	gut weight (after bleeding)
Shell weight|	continuous|	grams|	after being dried
Rings|	integer	|	|+1.5 gives the age in years

There are 4177 rows and 9 columns. With this information, we can cluster them into different classes. Figure 1 shows a sample of the first 5 rows of the data.

![Figure 1: First 5 rows of Abalone data](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%201.png)\
Figure 1: First 5 rows of Abalone data

### Data Exploration and Feature Engineering
1.	Check if there are any null values in the dataset.
The author mentioned that null values were already removed as shown in Figure 2. 

![Figure 2 df.info()](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%202.png)\
Figure 2: df.info()

2.	Drop any features with all unique values because they do not add value to the model.
This data set does not have any features with all unique values as seen in Figure 3. Therefore, we do not need to drop any column for now.

![Figure 3 Number of unique values for each column](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%203.png)\
Figure 3: Number of unique values for each column

3.	Check for string categorical variables and use One-hot encoding to convert all string categorical variables to binary dummies.
With reference to Figure 2, ‘Sex’ is a string type so we would need to convert using one-hot encoding. Also, even though ‘Rings’ is an integer type, but we need to do one-hot encoding so that the model does not take the ordinal encoding.

4.	Check for skewness of the columns. With the skew limit as 0.75, any features will more than an absolute value of 0.75 will be log transformed. 
As shown in Figure 4, ‘Height’ is skewed and will need log transformation. 

![Figure 4 Skew Values for float columns](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%204.png)\
Figure 4: Skew Values for float columns

![Figure 5 Histogram of Height Before Log Transformation](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%205.png)\
Figure 5: Histogram of Height Before Log Transformation

![Figure 6 Histogram of Height After Log Transformation](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%206.png)\
Figure 6: Histogram of Height After Log Transformation

Even after Log Transformation, as shown in Figure 5 and 6, ‘Height’ is still quite skewed with a skew value of 1.099602282943724. Therefore, we try another method called square root transformation. After square root transformation, the skew value become -0.29564050925436225 which is within our threshold of 0.75. Below also shows the Normal Test Results that was performed. 

![Figure 7 Histogram of Height after Sqrt Transformation](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%207.png)\
Figure 7: Histogram of Height after Sqrt Transformation

5.	Check for correlation between each feature. 
A correlation heat map was performed as shown in Figure 8. The heat map shows that the following features have very high correlation with each other.
 
![](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%208.png)\
![Figure 8 Correlation Heat Map](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%208-1.png)\
Figure 8: Correlation Heat Map

Since these features have high correlation and to keep the visualisation of models simple, we will only focus on ‘Sqrt_Height’ and ‘Rings’ for the following unsupervised models. 


### Unsupervised Models
#### Mean Shift
To find the best bandwidth for this model, we ran a simulation for a range of bandwidths [0.1, 0.2, 0.3, …, 2.0]. A graph was plotted based on the homogeneity and completeness scores of these bandwidths as shown in Figure 9. From the graph, the best bandwidth is 0.9. After fitting the model with a bandwidth of 0.9, Figure 10 shows the clustering result for ‘Sqrt_Height’ vs ‘Rings’ scatter plot.

![Figure 9 Scores for various Bandwidths](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%209.png)\
Figure 9: Scores for various Bandwidths 			      

![Figure 10 Clustering effect from Mean Shift](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%2010.png)\
Figure 10: Clustering effect from Mean Shift


#### Hierarchical Agglomerative Clustering (Ward)
For Agglomerative Clustering, we need to pick the optimal number of clusters. A simulation was performed for a range of clusters [1, 2, 3, …, 20] and the scores were plotted in Figure 11. For the optimal homogeneity and completeness scores, there should be 15 clusters. Agglomerative Clustering was performed again with 15 clusters and the clustering effect is shown in Figure 12. 

![Figure 11 Scores for various clusters](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%2011.png)\
Figure 11: Scores for various clusters	

![Figure 12 Clustering effect from Agglomerative Clustering](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%2012.png)\
Figure 12: Clustering effect from Agglomerative Clustering

#### DBSCAN
For DBSCAN, we need to pick the optimal number for epsilon (the radius of the local neighbourhood) and the density threshold for each neighbourhood. After running a simulation of a range of epsilon [0.1, 0.2, 0.3, … 2.9] and a range of density threshold [0.5, 1, 1.5, …, 5] and compare the scores of homogeneity and completeness, epsilon should be 0.9 and a density threshold of 2. Below shows the clustering effect from DBSCAN in Figure 13.
 
![Figure 13 Clustering effect for DBSCAN](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%2013.png)\
Figure 13: Clustering effect for DBSCAN

Homogeneity score measures if all the data points in each cluster are members of a single class. Completeness score measures if all the members of a single class are elements of the same cluster. Since these are unsupervised learning models, we shall compare homogeneity and completeness scores instead. Comparing the homogeneity and completeness scores of all 3 models, as seen in Figure 14, DBSCAN is the best model in this case with the best scores. 
 
![Figure 14 Overall homogeneity and completeness scores](https://github.com/cweien3008/portfolio/blob/main/Abalone/Figures/Picture%2014.png)\
Figure 14: Overall homogeneity and completeness scores

### Key Findings and Insights
One of the main findings is that many features are highly correlated to ‘Length’ and ‘Diameter’ so it may not value add to have ‘Length’ and ‘Diameter’ fit to the model. Log transformation does not work for ‘Height’ in this case so square root transformation is a better alternative. 
The points for these data set are very close to one another, so the outlier detection was not very good. The models mistaken outliers as clusters. 

### Future Works
Perhaps we could include the place the abalones are found and water condition as it could affect the physical attributes of the abalone. Perhaps we could also perform dimensionality reduction as some features seem to be related to one another. We could process the different columns into one to make the model less complex to reduce variance. 
Other than that, we could include the labels generated from these models into supervised learning so that it can more accurately predict the abalone’s age.



