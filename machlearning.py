import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Read the train and test data
x = pd.read_csv('.../train.csv', index_col='Id')
test = pd.read_csv('.../test.csv', index_col='Id')

# Remove the outliers
x = x[x['GrLivArea'] < 4000]
x.reset_index(drop=True, inplace=True)

# Select the target column and drop it from the data set
x.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = x.SalePrice
x.drop(['SalePrice'], axis=1, inplace=True)

# Drop numerical columns that are not pertinent
x.drop(['GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True)
test.drop(['GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True)

# Drop categorical columns that are not pertinent
x.drop(['PoolQC', 'FireplaceQu', 'Utilities'], axis=1, inplace=True)
test.drop(['PoolQC', 'FireplaceQu', 'Utilities'], axis=1, inplace=True)

# Creating features
x['TotalSF'] = x['TotalBsmtSF'] + x['1stFlrSF'] + x['2ndFlrSF']
x['Total_Bathrooms'] = (x['FullBath'] + (0.5 * x['HalfBath']) + x['BsmtFullBath'] + (0.5 * x['BsmtHalfBath']))

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

# Drop columns used to create features
x.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
x.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

test.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
test.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

# Select numerical et categorical columns
numerical = [col for col in x.columns if x[col].dtype in ['int64','float64']]
cat_cols = x.select_dtypes(include='object').columns

# Select numerical and categorical dataframes
xnum = x[numerical].copy()
testnum = test[numerical].copy()
xcat = x[cat_cols].copy()
testcat = test[cat_cols].copy()

# Impute NaN values in numerical columns
xnum['LotFrontage'] = xnum['LotFrontage'].fillna(xnum['LotFrontage'].mode()[0])
xnum['MSSubClass'] = pd.to_numeric(xnum['MSSubClass'])
testnum['LotFrontage'] = testnum['LotFrontage'].fillna(testnum['LotFrontage'].mode()[0])
testnum['MSSubClass'] = pd.to_numeric(testnum['MSSubClass'])

# Impute NaN values in categorical columns - train set
xcat.Alley = xcat.Alley.fillna('NOACCESS')
xcat['MasVnrType'] = xcat['MasVnrType'].fillna(xcat['MasVnrType'].mode()[0])
xcat['Electrical'] = xcat['Electrical'].fillna(xcat['Electrical'].mode()[0])
for col in ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']:
    xcat[col] = xcat[col].fillna('NOGARAGE')
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']:
    xcat[col] = xcat[col].fillna('NOBSMT')

# Impute the NaN values in categorical columns - test set
testcat.Alley = testcat.Alley.fillna('NOACCESS')
_ = ['MasVnrType', 'MSZoning', 'Electrical', 'Functional', 'SaleType', 'Exterior1st', \
    'KitchenQual', 'Exterior2nd']
for col in _:
    testcat[col] = testcat[col].fillna(testcat[col].mode()[0])
for col in ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']:
    testcat[col] = testcat[col].fillna('NOGARAGE')
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']:
    testcat[col] = testcat[col].fillna('NOBSMT')

# Encode the categorical values with One Hot Encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
Xcat = pd.DataFrame(encoder.fit_transform(xcat))
Xcat.index = xcat.index
Testcat = pd.DataFrame(encoder.transform(testcat))
Testcat.index = testcat.index

# Concatene numerical and categorical columns in one dataframe
X = pd.concat([xnum, Xcat], axis=1)
Xtest = pd.concat([testnum, Testcat], axis=1)

# Create the machine learning model - XG Boost
model = XGBRegressor(learning_rate=0.01,n_estimators=1500,
                                     max_depth=2, subsample=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=17,
                                     reg_alpha=0.00006)
                                     
# Split the data in a train and valid set
xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# model.fit(xtrain, ytrain)
predictions = model.predict(xvalid)
mae = mean_absolute_error(predictions, yvalid)
print("Mean Absolute Error:" , mae)

model.fit(X, y)
preds = model.predict(Xtest)
output = pd.DataFrame({'Id': Xtest.index,
                       'SalePrice': preds})
output.to_csv('submission.csv', index=False)