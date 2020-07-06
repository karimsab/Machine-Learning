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

![](https://fr.wikipedia.org/wiki/Donn%C3%A9e_aberrante#/media/Fichier:Doyens_humanit%C3%A9.png)
