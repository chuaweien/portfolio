# Titanic
The main objective of this analysis is to predict the survival of passengers from the sinking Titanic. The benefit of doing this is so that we know how many survived and how many did not survive the Titanic.

This dataset is obtained from the Titanic Machine Learning competition in Kaggle.
It consists of 891 rows and 12 columns. Column names, their descriptions and data types are shown below:

| Column | Description | Data Types |
| ------------- | ------------- | ------------- |
| 'PassengerId' | Identification of passenger | Int64 |
|'Survived' | Represents whether the passenger survived or not| Int64 <br> i.e. 1 or 0|
|'Pclass' | Ticket Class <br>i.e. 1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class | Int64|
|'Name' | Name of the passenger | Object, string|
|'Sex'	| Sex of the passenger	| Object, string|
|'Age' | Age in years of the passenger. If age is less than 1, it will be in float.| Float64|
|'SibSp' | Number of siblings/ spouses aboard. <br> Siblings = brother, sister, stepbrother, stepsister <br> Spouses = husband, wife (mistresses and fiancés are ignored)| Int64|
|'Parch' | Number of parents/ children aboard. <br> Parent = mother, father <br> Child = daughter, son, stepdaughter, stepson | Int64|
|'Ticket'|	Ticket number|	Object, string|
|'Fare'	|Passenger Fare paid	|Float64|
|'Cabin'|	Cabin number	|Object, string|
|'Embarked'| Port of Embarkation <br> i.e. C = Cherbourg, Q = Queenstown, S = Southampton|	Object, string|

With this information, we shall build a model to predict whether a passenger survive the sinking Titanic or not. ‘Survived’ will be the target and the rest of the columns are features.

### Data Exploration and Feature Engineering
1.	Check if there are any null values in the dataset.
 
![Table 1 Table of null values](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%2012.png)<br>
Table 1 Table of null values

‘Age’, ‘Cabin’ and ‘Embarked’ have null values as shown in Figure 1. We shall drop ‘Cabin’ columns because it has too many null values and it is difficult to assume what values to input for ‘Cabin’. Null values in ‘Age’ and ‘Embarked’ can be filled in with mean and mode values respectively. 
2.	Drop any features with unique values because they do not add value to the model.
 
![Table 2 Table of unique values](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%2010.png)<br>
Table 2 Table of unique values

‘PassengerId’ has 891 unique values so it should be removed from the model as it does not value add to the model. ‘Ticket’ and ‘Fare’ are believed to be generated randomly so they should be removed from the model too.

3.	Add new features to add more insights to the data.
‘family_size’ column could be added to show insight to which passenger is alone on the ship. We could derive the family size based on ‘Parch’ and ‘SibSp’. If the passenger is alone, ‘Parch’ and ‘SibSp’ are zero. As can been seen from Figure 1, those who are alone or have a family bigger than 7 have a higher chance of not surviving.
We could also add a ‘Salutation’ column derived from ‘Name’. From Figure 2, those who are ‘Mr’, ‘Capt’, ‘Don’, ‘Jonkheer’ and Rev’ have a higher chance of not surviving compared to ‘Miss’, ‘Mrs’, ‘the Countess’, ‘Sir’, ‘Ms’, ‘Mme’ and ‘Lady’.

 	 
![Figure 1 Survival Rate for various Family Size](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%201.png)<br>
Figure 1 Survival Rate for various Family Size


![Figure 2 Survival Rate for different Salutation](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%202.png)<br>
Figure 2 Survival Rate for different Salutation

4. Check for string categorical variables and use One-hot encoding to convert all string categorical variables to binary dummies.
Since ‘Sex’, ‘Embarked’ and ‘Salutation’ are string data types, we will convert them into binary dummies using One-hot encoding. ‘Salutation’ has 17 unique values, ‘Embarked’ has 3 unique values and ‘Sex’ has 2 unique values. All these will add new columns to the data set. 


5. Check for skewness of the columns. With the skew limit as 0.75, any features will more than an absolute value of 0.75 will be log transformed. 
In this case, only ‘Age’ is the relevant float variable. However, ‘Age’ skew value is only 0.4344880940129925 which does not meet our criteria of 0.75. So, we do not have to perform log transformation on any features.


6. Check if the dataset is balanced or unbalanced.
To check if the dataset is balanced or not, we have to sum up all the passengers who have survived and those who have not survived. Passengers who did not survive represent 61.6% of the data set while passengers who survived represent 38.4% of the data set. This shows that the data set is unbalanced and we will need to use Stratified KFold to maintain the representation.


### Classifier Models
#### Logistic Regression with Regularization
Using GridSearchCV, the best solver is ‘liblinear’ and the best regularization is L2 Ridge Regularization. Confusion matrix is generated below as Figure 3.

![Figure 3 Confusion Matrix for Logistic Regression](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%203.png)<br>
Figure 3 Confusion Matrix for Logistic Regression

#### Extra Trees
A range of trees [5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200] is run against ExtraTreesClassifier and out-of-bag error is generated as shown in Figure 4. The number of trees that give the least out-of-bag error is 50. Confusion matrix is again generated below as Figure 5.
  
![Figure 4 out-of-bag error for Extra Trees Classifier](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%204.png)<br>
Figure 4 out-of-bag error for Extra Trees Classifier

![Figure 5 Confusion Matrix for Extra Trees](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%205.png)<br>
Figure 5 Confusion Matrix for Extra Trees

#### Boosting
In this case, AdaBoost was used for comparison. Using GridSearchCV, a range of parameters were run against Decision Tree Classifier. The best n_estimators is 100 and the best learning_rate is 0.1.
Confusion matrix is generated as Figure 6 below.
 
![Figure 6 Confusion Matrix for AdaBoost](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%206.png)<br>
Figure 6 Confusion Matrix for AdaBoost

Classification report was run for all the 3 models as shown in Figure 7 where LR represents Logistic Regression Classifier, EF represents Extra Trees Classifier and ABC represents AdaBoost Classifier. Based on Figure 7, AdaBoost is the best model because it has the highest weighted average F1 score.
 

![Figure 7 Classification Report on all 3 models](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%207.png)<br>
Figure 7 Classification Report on all 3 models

### Key Findings and Insights
There were more passengers who did not survive the sinking Titanic then those who survived as mentioned in Data Exploration and Feature Engineering Section. 
Comparing the heatmaps in Figure 8 and Figure 9, it shows that some of the ‘Salutation’ do have some correlations to survival rate. ‘Sex’ also has an impact on survival rate. From Figure 9, it seems that ‘family size’ does not have much correlation to survival rate as previously thought. ‘Parch’ showed some correlations to survival rate in Figure 8 but not Figure 9.
  
![Figure 8 Correlation Heatmap Before Processing](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%208.png)<br>
Figure 8 Correlation Heatmap Before Processing

![Figure 9 Correlation Heatmap After Processing](https://github.com/cweien3008/portfolio/blob/main/Titanic/Images/Picture%209.png)<br>
Figure 9 Correlation Heatmap After Processing

### Future Works
For the next steps, we could tune some more parameters in the model. As shown in Figure 8, it seems that ‘Fare’ do have some correlations to survival rate. Perhaps we could add ‘Fare’ in and re-run the model again. 
We should also collect more data so that our models will be more accurate i.e. weight of passengers, etc. We could also try to run the data with other models to compare the F1 score and confusion matrix.




