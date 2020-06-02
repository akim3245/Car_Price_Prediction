# Car Price Prediction
Predict car prices using machine learning (XGBoost Regressor)

### Project Goal
The purpose of this project is to predict car prices with machine learning using a cars dataset consisting of information such as car brands, model, and year.

### Methods & Technologies
* Machine Learning
* Data Visualization
* Predictive Modeling
* Python
* Pandas 
* Jupyter Notebook

## Project Description
To better understand the dataset used, the predictive features were explored and analyzed to answer questions like, *what are the top selling car brands and models*, *which car brands and models are more on the expensive side*, *what colors are most popular*, *which car is the most expensive*, *what features are correlated with price value*, and etc.
A few of the questions are answered below.

Top brands with above average prices:
[![brands.png](https://i.postimg.cc/dVrqFf0M/brands.png)](https://postimg.cc/K1cXrJLQ)

Most popular colors:
[![colors.png](https://i.postimg.cc/65NK000Y/colors.png)](https://postimg.cc/0rZh57hw)

Most expensive car in the dataset:
[![mostexpensive.png](https://i.postimg.cc/Dzm3ZYvy/mostexpensive.png)](https://postimg.cc/DWVMp64t)

After preprocessing the data, XGBoost Regressor was used to train and make predictions on the test set. 

## Results
Using a random sample of data, the algorithm made predictions that were relatively close to the true value.
[![sample.png](https://i.postimg.cc/vBqh0MC5/sample.png)](https://postimg.cc/zbhTBmzf)

The model produced the following r2 scores and root mean square errors:

[![score.png](https://i.postimg.cc/9M4FqQBS/score.png)](https://postimg.cc/sQrRkyq9)
[![graph.png](https://i.postimg.cc/QNTd8VCK/graph.png)](https://postimg.cc/t79jDqh9)

The train set performs better with a higher r2 score and lower RMSE which suggests that it is overfit to the training data.

## Future Work
To improve the model, I would consider adjusting gamma, L1 and L2 regularization hyperparameters to control the complexity of the model. 
Another reason the model may have overfit to the training data could be because there are many features (more than 200) 
therefore I would run feature selection to keep only the most informative features.

## Notebooks
**Cars_Dev** - Process of exploring the data with feature visualizations (matplotlib, seaborn), cleaning, model selection, optimizing algorithm parameters and testing. (Model Development)

**Cars_Production** - No visualizations or experiments, only processing and cleaning the data, then training and testing the model using the best parameters. 
