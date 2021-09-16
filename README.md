# MLStarter_Project_template

Data Science Interview preparation:
Data Science Interview Questions:

Model Evaluation Questions:
1. Draw a Confusion Matrix and Label all the Parameters - Precision, Recall, Accuracy
2. Which is more important - Precision or Recall? in a Malware Detection Problem?
3. Which is a better Accuracy Parameter? 
4. LogLoss vs ROC_AUC curve? Which is better?
5. Regression problems evaluation parameters?
6. Clustering problems evaluation parameters?

NLP Questions:
1. Components of NLP Pipeline?  
2. Explain Word Embedding? - CBOW vs Skip-gram 
3. Explain TF-IDF vectorization
4. Using Pre-trained Word Embeddings - Word2Vec, GloVE, 
5. How do determine whether 2 documents are similar? Explain?
6. CRF and HMM - Explain with an NER problem 

EDA Questions:
1. Normalization of data?
2. Identify Outlier in the data? How to treat them?
3. Dimensionality Reduction? Pros and Cons? Methods? 
4. Can we use RF algorithm for Dimensionality Reduction? 

Algorithm Questions:
1. Disadvantages of Linear Regression? 
2. Bias Variance Trade off? How to choose the best fit? 
3. Handling Bias problem? Handling Variance Problem? Underfit/Overfit - L1(abs|coeff|) and L2(coeff**2)  
4. Type 1(FP) and Type 2(FN) Error - FP is dangerous (chemotherapy), FN is dangerous (NO medication for Cancer) 
5. Ensemble Methods? Bagging, Boosting, Stacking

Deep Learning Questions:
1. How are weights initialized in the network? Can we initialize same weights in all the network?
2. Vanishing or Exploding Gradient problem?
3. In a multiclass classification problem - which output layer do we use and why?
4. Forward Propogation and Backward Propogation? 
5. Prevent Overfitting in Deep Neural Network

Python Questions:
1. Map, Reduce and Filter function in Python
2. Difference between Append and Extend
3. Combine 2 List into a list of tuples ( a = ['a','b','c'] , b= [1,2,3] , output = [('a',1),('b',2),('c',3)]

-------

Statistics and Probability
Random Forest
XGBoost
SVM, Naive Bayes
Deep Learning Architecture - RNN, LSTM, BiLSTM, other Important concepts
NLP - Everything + NER
SQL, NOSQL database
API - Flask, FAST API
Python 

-------

Normal Distribution: Distribution of the data where mean=median=mode
Mean, Median, Mode : Measure of Central tendency, 
	Mean: Arithmetic Mean of the distribution
	Median: Sort the distribution in ascending order, and the middle most value in the distribution
	Mode: Most frequent value in the distribution

Standard Normal Distribution: Normal Distribution with Mean = 0, and Standard Deviation = 1

Standard Deviation and Variance: Measure of Spread
In Normal Distribution:
	68% of values fall within 1 Standard Deviation
	95% of values fall within 2 Standard Deviation
	99.7% of values fall within 3 Standard Deviation

Z-score or Standard Score: Number of Standard Deviation from the mean. 
	z-score = 3, defines that 99.7% of values in the distribution
z-score = (x - mean)  / stdev --> it is used to standardize data i.e. mean = 0, Stdev = 1

Variance: Average of the squared distance from the mean. It defines the deflection of the data from the Mean 
Standard Deviation: SQRT of Variance, defines the spread of the data.

Histogram: used to plot the distribution of data in defined bins. 

Skewed Data :
Right Skewed or Positive Skewed - when mean > median > mode, the data distribution peak is towards left. 
Left Skewed or Negative Skewed - when mode > median > mean, the data distribution peak is towards right.

Data Skewness is caused due to outliers and can affect regression models.
Handle Skewness - Log Transformation, Square root Transformation, BoxCox Transformation

Population - The whole group
Census - collection of data from whole population
Sample - Collection of data from part of the population - Random Sampling, Stratified Sampling
Stratified Sampling - divide the population into different groups and choose data points in some ratio from all the groups

Central Limit Theorem - States that sampling distribution of the sample mean approaches a normal distribution as the sample size becomes larger
irrespective of the population distribution (uniform, exponential, etc)
Min Sample size is considered to be 30 
Average of the sample mean or average of the standard deviation will be the population mean or population standard deviation

------

Probability - Later

------

Random Forest - works as an ensemble of decision trees, trained as a bagging method - combination of learning models increases the overall result. 
Entropy - measure of impurity, uncertainity, randomness in the dataset. The goal is to minimize the entropy
Information Gain - measure of how much information a feature gives us about the classes. 
Feature which perfectly splits the data has highest information gain.

Random Forest Hyper parameters -
max_depth - longest path between the root node and leaf node
min_sample_split - minimum required number of observations in any given node in order to split
max_leaf_nodes - maximum leaf nodes to be set to stop further splitting 

-----
Vanishing Gradient - happens in Back propogation while calculating the derivate at each layer. In case of deep neural network, the derivative of the 
activation function (sigmoid) tends to a lower value. Also, we multiply it with a learning rate (which is chosen as 0.01). 
Thus, the weights update is vanished, and network proceeds with the similar weight. -- vanishing gradient 

Exploding Gradient - this happens when we chose high value of weights. In the back propogation, while the weight update happens by calculating the 
derivative of the activation function, the chain rule is applied. In the chain rule, if the derivative is written w.r.t (w.x + b), the result is w.
If the w chosen is very high then the weight update will be higher -ve value, and it will never converge to global minima.
And therefore the network will not learn from the training data.
How to identify exploding gradient problem -
model loss is unstable, resulting in large changes in loss.
model loss goes to Nan during training.

fix: 
redesign the network to have fewer layers
weight regularization - check the size of the network weights and apply the penalty to the networks loss funciton.
use Gradient clipping - limiting the size of gradients during the training of the network 

-----
Overfitting problem in Neural network --
1. Regularization - L1 and L2 
	L1 (Lasso) - adds a penalty term in the cost function which is the summation of absolute value of weights.
	L2 (Ridge) - adds a penalty term in the cost function which is the summation of Squared value of weights.
	Elastic Net - combination of L1 and L2 
2. Drop out - neuron are deactivated based on the drop out ratio. in the test data, all the neurons will be connected. at each node the drop out 
				ratio will be multiplied with the weights. in deep layer, we can use p value as > 0.5

-----				
Optimizers: 
1. Gradient Descent, Mini-Batch GD and Stochastic GD
2. Adaptive gradient descent - use different learning rate on different neurons in different iterations in different layers
3. RMSprop - 
	learning rate reduces with each iteration , learning rate is inversely proportional to alpha, which is a square of derivative of loss. 
	This causes very slight change in learning rate in each iteration (causing vanishing gradient problem) 
	To fix this problem, Adagrad and RMSprop uses weighted average to restrict the increase in alpha 
4. ADAM - uses both the concepts of Adagrad and RMSprop

-----
Limitation of Linear regression: 
1. linearity in the data points is expected. 
2. Sensitive to outliers
3. Normal distribution of data 
4. Need more of Feature engineering (removing outliers, null values, multi-collinearity, Dimensionality reduction, etc)
5. assumes independence between features, if the features are correlated, it can affect the performance.

-----
Evaluation metric for Regression problem: 
R2 and Adjusted R2 - 

R2 = 1 - (sum of square of residuals between actual and predicted / sum of square of residuals between actual and average )
R2 values ranges between (0,1) - 1 defining best accuracy or fit line 
Value of R2 can be < 0, when the best fit line is worser than the average fit line. (model is worse)
Adjusted R2 = As we add more independent features, the R2 value might increase, to penalize the R2 value - adjusted R2 is used. 
Adjusted R2 penalized the attribute that are not correlated.

ROC AUC curve :
ROC graph between TPR(y-axis) and FPR(x-axis), it plot the threshold value on the graph.
TPR = TP /(TP + FN)
FPR = FP /(FP + TN)
y 		yhat 
0		0		TN
0		1		FP
1		0		FN
1		1		TP

AUC curve is area under the ROC curve - Area should be greater for good model
How to select threshold - depends on whether we need a model needs a higher TPR irrespective of FPR or Lowest FPR with moderate TPR.

-----
Precision and Recall

Precision = TP/ (TP+FP) - Type 1 error   (e.g. Spam Detection - mail recepient might miss the mail if mail categorized as FP (meaning SpAM mail)
Recall or TPR = TP/ (TP+FN) - Type 2 error (e.g. Cancer Detection - person might not get treated for cancer, if detected as FN (meaning he has cancer but missed in detection)

If False Positive is important - we need to check Precision score
If False Negative is important - we need to check Recall score

Fbeta - (1+b2)* PR*RE / b2 * (Pr + Re)

beta = 1 -> when both FP and FN are equally important 
beta = 0.5 -> when FP is more important
beta = 2 -> when FN is more important 

-----
Bias and Variance Trade-off: 
High Bias generalizes the model and fails to learn accurately from the feature set. low accuracy in both training and 
test set. Solution : remove correlated features, add more data and complexity in the model.

High Variance tunes the model accurately on various feature set including outlier or noise, causing high accuracy in the training set, but
low accuracy in the test set. Solution: use regularization techniques to penalize the cost function. remove outliers/noise from dataset, 
normalize the dataset using standard/Minmax scaler function.

----
Exploratory Data Analysis Steps: 
1. Columns details, Total row counts, Null counts at row level and column level, data types of the columns
2. Junk removal - using regex patterns 
3. Null imputation if required - Mean imputation and most frequent imputation 
4. Outlier removal - using boxplot analysis
5. dimensionality reduction techniques - using PCA
6. plotting histogram for some columns - to check the distribution of the data 
7. plotting heat map for the dataset - to check the collinearity between independent features.
8. Normalizing the data - using Standard Scaler, MinMax Scaler
9. Encoding dependent feature - one hot encoding, label encoder
10. Feature engineering - adding derived features from the given set of features.
11. Class Imbalance check - oversampling, undersampling, SMOTE analysis
12. Random/Stratified sampling from the population if needed 
13. Plotting bar chart or scatter chart if needed 
14. Train test split of dataset 
15. K-fold cross validation or batching  
16. passing the train data set in the default parameter model 
17. prediction on the test dataset
18. Evaluating the performance of the model 
19. hyper parameter tuning - using grid search/random search
20. evaluating the model on the best set hyper parameters 
21. plotting confusion matrix/ roc-auc curve/ 

-----
XGBoost and AdaBoost:

Boosting: dataset is passed to the Sequential Base learners, and further it is checked which records are incorrectly classified - those records are
passed to the next sequential base learner, this process is repeated until the weak learners are trained well. 

Adaboost combines multiple weak learners into Strong learners. 
It creates decision stumps(DT with depth=1) for each feature, and calculates the entropy for each Decision Stump. 
The decision stump with lowest Entropy is taken ahead 
We will assign equal sample weight to each observations
We will check which observations are wrongly classified
we will calculate the total error based on misclassified observations
..

Gradient Boosting - creating sequential decision trees based on the residuals calculated. The main aim is to reduce the residual error. 
hyper parameter alpha is used to reduce the residuals in the DT. 

-----
Bagging (Bootstrap Aggregation): multiple base models/learners. For each model, different set of samples of dataset is provided for training base learners.
Sampling technique - Row Sampling with Replacement. 
During Test data, the complete dataset is provided to all the base learners and voting classifier is applied (majority vote will be the output)

Random Forest - Row Sampling with replacement and Feature Sampling are applied. DT caused Low Bias and High Variance, but after bootstrapping multiple
DT and aggregating the decision trees, the output is low bias and low variance 
In RF model, even if a portion of train dataset changes in future, the model will have very less impact. - best case model 
in RF regressor - either mean or median is taken at the output

-----

Silhoutte Score: bi - ai / max(bi,ai) - 
bi -> average intercluster distance of the point in the cluster
ai -> average intracluster distance of the same point in the cluster

-----

NLP Pipeline:
1. Gathering Text - either from Web Scrapping or Available dataset
2. Data Preprocessing - Remove Header, footer if present - through Beautiful Soup library in Python 
						Removing Punctuation, Junk Characters - through regex, 
						Transform all words to lower case 
						Removing Stop words - through NLTK stopwords library, combining different 
						columns data, dropping some columns, etc. - at last defining the dataset with the dependent feature 
3. Tokenization - Convert sentence into token of words, split by spaces. 
4. Vectorization - 	Converting the text into vectors - using Count Vectorizer or TF-IDF Vectorizer (these are Frequency based Word Embeddings)
					There are few pre-trained vectorizers - WORD2VEC and GLOVE - uses cosine similarity to determine similarity between 2 vectors. 
					It places the words into feature space in such a way that their location in feature space is determined by their meaning.
					Words having same meaning are clustered together, and distance between 2 words also have same meaning. 
					E.g. - Distance between Man and Woman, will be same as Distance between King and Queen
5. Train-Val-Test Splitting of dataset
6. Model Training - Training the model using Shallow Classifiers or DNN. 
7. Model Evaluation - Confusion Matrix
8. Model Hyperparameter tuning - 
9. Model Evaluation - 
10. Finalizing Best Model 

TF-IDF = Term Frequency X Inverse Document Frequency 
TF = No. of time word appear in the Document / Total Number of word in the document. Weighted Term Frequency => wt,d = 1+ log(tf) when tf > 0, or 0 otherwise
	log damps the words with very high frequency
IDF = finding the importance of the word. Based on the fact that less important words are more informative and important. 
		IDF = log(No. of document / No. of document in which word has appeared)

TFIDF = TF X IDF

Frequency Based Word Embeddings -  Count Vectorizer and TF-IDF vectorizers
Prediction Based Word Embeddings - CBOW and Skip-gram 
