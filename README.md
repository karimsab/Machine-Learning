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

1. Load data

First of all, we load the datasets (after importing the useful modules):
```
train = pd.read_csv('.../train.csv')
test = pd.read_csv('.../test.csv')
```

2. Data preparation

In this part, there are successive tasks, to, at the end, obtain a clean dataset, which will be more easily usable to explore and modeling it.

2.1 Remove outliers 
![outliers](https://user-images.githubusercontent.com/62601686/86583356-67b8b280-bf83-11ea-9173-0c5a1f39092c.png)

Often, in a dataset, we have outliers : values that are far from the main values observed.
We can at first, plot the values and see which one is an outlier, then to remove them and have a more "contain" dataset.

```
data = data[data['GrLivArea'] < 4000]
data = data[data['LotFrontage'] < 300]
data = data[data['LotArea'] < 100000]
data.reset_index(drop=True, inplace=True)
```

2.2 Handling missing values

Of course, we will 
