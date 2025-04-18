#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# In[2]:


df = pd.read_excel('../data/Crop_Yield_Prediction.xlsx')
df.head()


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


df = df.dropna()


# In[7]:


df.shape


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates()


# In[10]:


df['Temperatue'] = df['Temperatue'].astype(int)


# In[11]:


df.hist(figsize=(16,10))
plt.savefig("yr_featuredistribution.png", dpi=300, bbox_inches='tight')
plt.show()


# In[12]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot = True)
plt.savefig("yp_correlation_heatmap.png", dpi=300, bbox_inches='tight')


# In[13]:


x = df.drop('Yeild (Q/acre)',axis = 1)
y = df['Yeild (Q/acre)']


# In[14]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state =42)


# In[15]:


scaler = MinMaxScaler()
xtrain_scaled = pd.DataFrame(scaler.fit_transform(xtrain),columns = xtrain.columns)
xtest_scaled = pd.DataFrame(scaler.transform(xtest), columns = xtest.columns)


# In[16]:


from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# Define models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "SVR": SVR(),
    "KNN Regressor": KNeighborsRegressor(),
    
}

# Store results
results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(xtrain_scaled, ytrain)
    y_pred = model.predict(xtest_scaled)

    mse = mean_squared_error(ytest, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, y_pred)

    results.append([name, mse, rmse, r2])

# Create and display results DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R² Score"])
print(results_df)


# In[17]:


import pickle

best_model = KNeighborsRegressor()
best_model.fit(xtrain_scaled, ytrain)

# Save the trained model
with open("../models/yield_prediction_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the scaler
with open("../models/yield_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Best model and scaler saved successfully.")


# In[ ]:




