
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math
np.random.seed(20170306)
get_ipython().magic('pylab inline')


# In[2]:

# EDA
X_train = pd.read_csv("midterm_train.csv")
y_train = X_train.pop('y')
X_test = pd.read_csv("midterm_test.csv")


# In[3]:

X_train.describe()


# In[4]:

def describe_categorical(X):
    """
    Just like .describe(), but returns the results for
    categorical variables only.
    """
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))
describe_categorical(X_train)


# In[5]:

# Strip unwanted % and $
# http://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column
X_train['x9'] = pd.to_numeric(X_train['x9'].map(lambda x: str(x)[:-1]), errors='coerce')/100
X_train['x44'] = pd.to_numeric(X_train['x44'].map(lambda x: str(x)[1:]), errors='coerce')


# In[6]:

# Fill missing values
cols = list(X_train.describe())
for x in cols:
    avg = X_train[x].mean()
    X_train[x] = X_train[x].fillna(value=avg)


# In[7]:

X_train.describe()


# In[8]:

describe_categorical(X_train)


# In[9]:

categorical_variables = X_train.columns[X_train.dtypes == "object"].tolist()
for variable in categorical_variables:
    # Fill missing data with the word "Missing"
    X_train[variable].fillna("Missing", inplace=True)
    # Create array of dummies
    dummies = pd.get_dummies(X_train[variable], prefix=variable)
    # Update X to include dummies and drop the main variable
    X_train = pd.concat([X_train, dummies], axis=1)
    X_train.drop([variable], axis=1, inplace=True)


# In[10]:

### Clean Test Set
# Strip unwanted % and $
# http://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column
X_test['x9'] = pd.to_numeric(X_test['x9'].map(lambda x: str(x)[:-1]), errors='coerce')/100
X_test['x44'] = pd.to_numeric(X_test['x44'].map(lambda x: str(x)[1:]), errors='coerce')

# Fill missing values
cols = list(X_test.describe())
for x in cols:
    avg = X_test[x].mean()
    X_test[x] = X_test[x].fillna(value=avg)

# Dummies
categorical_variables = X_test.columns[X_test.dtypes == "object"].tolist()
for variable in categorical_variables:
    # Fill missing data with the word "Missing"
    X_test[variable].fillna("Missing", inplace=True)
    # Create array of dummies
    dummies = pd.get_dummies(X_test[variable], prefix=variable)
    # Update X to include dummies and drop the main variable
    X_test = pd.concat([X_test, dummies], axis=1)
    X_test.drop([variable], axis=1, inplace=True)


# In[11]:

# Confirm Data is Cleaned
# Look at all the columns in the dataset
def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))
    
printall(X_train)


# In[12]:

printall(X_test)


# In[13]:

### Orignially attempted different models with RandomForestRegressors and GridSearch methods
### Then, tested individual models to save on compute time resulting in the following model with best results

model_3_rfr = RandomForestClassifier(n_estimators=10000, 
                              criterion = "entropy",
                              n_jobs=-1, 
                              random_state=42, 
                              max_features=0.4, 
                              min_samples_leaf=1)
model_3_rfr.fit(X_train, y_train)
rocb = roc_auc_score(y_train, model_3_rfr.predict_proba(X_train)[:,1])
print ("C-stat: ", rocb)

df_model_3_rfr = pd.DataFrame(model_3_rfr.predict_proba(X_test)[:,1])
df_model_3_rfr.to_csv("y_hat_model_3_rfc15k.csv", index=True)

