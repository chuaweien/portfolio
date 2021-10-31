# Virgin America Airline
The main objective is to analyse tweets mentioning Virgin America Airline using Long-Short Term Memory of Recurrent Neural Networks.  This will help to train the model to understand what the tweets are about and perhaps could be used for text generation in the future to answer those tweets.

The data set was downloaded online with 14,640 rows and 15 columns. The contained columns are as shown below in Figure 1. Some columns have null values and will require data cleaning.

![Figure 1 Attributes of data set](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%201.png)\
Figure 1: Attributes of data set

### Data Exploration and Feature Engineering
1. Remove null-values and irrelevant columns
As shown in Figure 1, ‘negativereason’, ‘negativereason_confidence’, ‘airline_sentiment_gold’, ‘tweet_coord’, ‘tweet_location’ and ‘user_timezone’ have too many null values to fill in to get the accurate picture so they are removed. Also, ‘tweet_id’ and ‘name’ are unique values which do not add value to the model, so they are removed too.

2. One-hot encoding for ‘text’ column.

Since ‘text’ column is in string, we will need to convert them into integers for our model. Before that, we transform all the text in the column to lower case to reduce the vocabulary size that the network needs to learn. 

Based on our text, we have 14,640 characters and 14,409 unique vocabulary. 

### Training Models
To simplify the training model, we will only be using the ‘text’ column. We will be using Long-Short Term Memory of Recurrent Neural Networks with various hyperparameters for comparison.
For the first comparison, we are defining a sequence length of 100 characters with a LSTM layer of 256 memory units. With this, we have a total of 14,540 patterns and a shape of (14540, 100, 1). After compiling the model, the model summary can be found in Figure 2. The loss function used in this case is ‘categorical_crossentropy’ and the optimizer is ‘rmsprop’.
 
![Figure 2 Model Summary of 1 Layer](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%202.png)\
Figure 2: Model Summary of 1 Layer

Figure 3 shows the result of model fitting with epochs of 20 and batch_size of 128. However, in this case, we are not concern with the accuracy of the model. We are concern with the generalisation of the dataset that minimises the loss function.
 
![Figure 3 Result of Model 1 fitting](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%203.png)\
Figure 3: Result of Model 1 fitting

The model was run again but this time with 3 layers, instead of 1, but the hyperparameters remain unchanged.
 
![Figure 4 Result of Model 2 fitting](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%204.png)\
Figure 4: Result of Model 2 fitting

As can be seen from Figure 4, by adding additional layers does not improve the loss function.
As the third variation, we are reducing the sequence length to 30 and increasing the memory units to 300, the rest of the hyperparameters remain unchanged. The total pattern now is 14,560 with a shape of (14610, 30, 1). We also increased epochs to 100 and reduce batch_size to 50. Below shows the loss function of the last 5 epochs (Figure 5) and the model summary (Figure 6).
 
![Figure 5 Result of Model 3 fitting](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%205.png)\
Figure 5: Result of Model 3 fitting
 
![Figure 6 Model 3 Summary](https://github.com/cweien3008/portfolio/blob/main/Virgin%20America%20Airline/Figures/Picture%206.png)\
Figure 6: Model 3 Summary

As can be seen from above results, Model 3 is a better model with a loss function of 9.4918. This shows that with a lower sequence length, higher memory units, higher epochs value and lower batch size can help to improve the model. 

### Key Findings and Insights
One of the key findings is that with lower sequence length, there are more word patterns to train the model, so it makes the model better. 

Also, another key finding is that it takes very long to fit the model because of the epochs, sequence length and batch size. So, perhaps we would need better GPU power or tune the hyperparameters to get the optimal results at optimal speed. 

### Future Works
For future works, perhaps we could remove all the punctuations from the source text so we could reduce the vocabulary size of this model. We could also gather more tweets so that we have a large dataset to train the model with.
Perhaps we can tune the hyperparameters even further such as reducing the sequence length, increasing the memory units, or increasing the training epochs to make a better model.
