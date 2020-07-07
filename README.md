# Machine-Learning
*Machine Learning basic application*

This is an overview and a basic example of what we are capable of in data science machine learning.

For this, we are using the [House Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course) on Kaggle site,

where the objective is to predict house prices from a train and test datasets. 

**Content :**

1. Load data
2. Data vizualisation
3. Data preparation
4. Exploratory data analysis
5. Modeling
6. Conclusion

---
**Introduction**
---

1. Load data

First of all, we load the datasets (after importing the useful modules):
```
train = pd.read_csv('.../train.csv')
test = pd.read_csv('.../test.csv')
```
2. Data vizualisation

Here, we can use the Seaborn package to do some data vizualisations of our dataset and spot some useful informations.

For that, we first split the data in a numerical and a categorical set. For your information, numericals data are data that are only numerical (obviously). For example, the number of bedrooms (1, 2, 3...). And categoricals data are the other ones, for example, the color of bedrooms (blue, green, white...).

We split the train set :
```
numerical_cols = [col for col in train.columns if train[col].dtype in ['int64','float64']]
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
numerical_train_data = train[numerical_cols].copy()
categorical_train_data = train[categorical_cols].copy()
```
Now we are able to plot the numericals data (scatter plot for example) and see what we can do with it. In the same way, we can plot all the categoricals data (using another type of plot) and explore it.

3. Data preparation

In this part, there are successive tasks, to, at the end, obtain a clean dataset, which will be more easily usable to explore and modeling it.

3.1. Remove outliers 

Often, in a dataset, we have outliers : values that are far from the main values observed.
We can at first, plot the values and see which one is an outlier, then to remove them and have a more "contain" dataset.

```
train = train[train['GrLivArea'] < 4000]
train = train[train['LotFrontage'] < 300]
train = train[train['LotArea'] < 100000]
train.reset_index(drop=True, inplace=True)
```

3.2. Handling missing values

Of course, we will have missing values that we have to deal with. One can see the missing values by using `pd.DataFrame.isnull()` method.

Then, there are many possibilities to deal with the missing values. We can simply remove them, but we also remove informations from the dataset. The other way, is to impute missing values with another ones. We will use this method.

The Pandas packages allow us to fill the **NaN** values with a desired one, here, we impute them with the *most frequent* value for the chosen column.

`train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mode()[0])`

3.3. Categorical data

For the categorical data, the process is to encode them to have numerical values instead. For example, again we take the different colors of a bathroom, we create a column for each categorical variable and we impute the presence (1) or not (0) or the variable :

![Capture d’écran 2020-07-07 à 12 20 24](https://user-images.githubusercontent.com/62601686/86767561-71f2b380-c04c-11ea-97ba-563713cbb2d1.png)

It's more convenient to considere categorical variables with a limited number of unique variable.

For that we use the module *One Hot Encoder* :
```
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
categorical_encoded_train_data = pd.DataFrame(encoder.fit_transform(xcat))
categorical_encoded_train_data.index = categorical_train_data.index
```
4. EDA
