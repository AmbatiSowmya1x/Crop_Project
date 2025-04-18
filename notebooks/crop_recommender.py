#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("../data/crop_recommendation.csv")  # Adjust path if needed
df.head()


# In[3]:


print(df.isnull().sum())  # Ensure no missing values


# In[5]:


df.duplicated().sum()


# In[6]:


df['label'].unique()


# In[7]:


df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=15)
plt.savefig("cr_feature_distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))
sns.countplot(x='label', data=df, order=df['label'].value_counts().index, palette='viridis')
plt.xticks(rotation=90)
plt.title("Crop Frequency Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("cr_crop_frequency.png", dpi=300, bbox_inches='tight')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("cr_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# In[10]:


selected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'label']
sns.pairplot(df[selected_features], hue='label', corner=True)
plt.savefig("cr_pariplot.png", dpi=300, bbox_inches='tight')
plt.show()


# In[11]:


from sklearn.preprocessing import LabelEncoder

# Encode crop labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Save encoder for later use
import pickle
with open("../models/crop_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

crop_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
print(crop_mapping)  # This will help when decoding predictions later


# In[12]:


from sklearn.model_selection import train_test_split

# Define input features and target variable
X = df.drop(columns=['label'])  # Features (N, P, K, pH, humidity, temperature, rainfall)
y = df['label']  # Target (crop)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# In[13]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Define models
models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear"),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate models
best_model = None
best_accuracy = 0
results = []

for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    
    # Predictions
    y_train_pred = model.predict(X_train)  # Training predictions
    y_test_pred = model.predict(X_test)  # Testing predictions

    # Calculate accuracy
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    # Classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)  
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]


    results.append([name, test_acc, precision, recall, f1_score])

    # Save best model based on test accuracy
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model

# Convert results to DataFrame and print
results_df = pd.DataFrame(results, columns=["Model", "Test Accuracy", "Precision", "Recall", "F1-Score"])
print(results_df)


# In[14]:


import pickle

# Save the trained Linear Regression model (or whichever model is best)
with open("../models/best_crop_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved successfully as 'best_crop_model.pkl'.")
print(best_model)


# In[ ]:





# In[ ]:




