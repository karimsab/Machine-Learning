# Machine-Learning
*Machine Learning basic application*

This is an overview and a basic example of what we are capable of in data science machine learning.

For this, we are using the [House Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course) on Kaggle site,

where the objective is to predict house prices from a train and test datasets. 

**Content :**

1. Load data
2. Data preparation
3. Exploratory data analysis
4. Modeling
5. Conclusion

---
**Introduction**
---

1. Load data

First of all, we load the datasets (after importing the useful modules):
```
train = pd.read_csv('.../train.csv')
test = pd.read_csv('.../test.csv')
```

2. Data preparation

In this part, there are successive tasks, to, at the end, obtain a clean dataset, which will be more easily usable to explore and modeling it.

2.1 Remove outliers 

Often, in a dataset, we have outliers : values that are far from the main values observed.
We can at first, plot the values and see which one is an outlier, then to remove them and have a more "contain" dataset.

```
data = data[data['GrLivArea'] < 4000]
data = data[data['LotFrontage'] < 300]
data = data[data['LotArea'] < 100000]
data.reset_index(drop=True, inplace=True)
```

2.2 Handling missing values

Of course, we will have missing values that we have to deal with.

For that, we first split the data in a numerical set and a categorical set. For your information, numericals data is data that are only numerical (obviously). For example, the number of bedrooms. And categoricals data are the other ones, for example, the color of bedrooms.

We split the train set :
```
numerical_cols = [col for col in train.columns if train[col].dtype in ['int64','float64']]
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
```
Then, there are many possibilities to deal with the missing values. We can simply remove them, but we also remove informations from the dataset. The other way, is to impute missing values with another ones. We will use this method.

The Pandas packages allow us to fill the **NaN** values with a desired one, here, we impute them with the *most common* value for the chosen column.
`train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mode()[0])`

