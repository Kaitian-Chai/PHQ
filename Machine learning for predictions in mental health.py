#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Import of data processing and charting functions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.impute import SimpleImputer
sns.set_palette('Set2')

# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[38]:


data_RATINGS = pd.read_csv('C:/Users/jingying/Desktop/dataset-main/data/RATINGS.CSV')
#Print the first few lines of data_RATINGS to test
data_RATINGS.head()


# In[39]:


print('Questions answered by '+str(data_RATINGS.shape[0])+' people.')


# In[40]:


data_RATINGS[['PHQ1', 'PHQ2', 'PHQ3','PHQ4', 'PHQ5', 'PHQ6','PHQ7', 'PHQ8', 'PHQ9','PHQ10','PHQTOTAL']].describe()


# In[41]:


# Calculate the number of values in the PHQTOTAL column that are greater than 10
count_greater_than_10 = data_RATINGS[data_RATINGS['PHQTOTAL'] > 10].shape[0]

# Calculation ratio
ratio = count_greater_than_10 / data_RATINGS.shape[0]
# Visualising proportions using pie charts
labels = ['Greater than 10', 'Not Greater than 10']
sizes = [ratio, 1-ratio]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Proportion of 'PHQTOTAL' greater than 10")
plt.show()


# In[42]:


data_SCREENINGHEALTHSTATUS = pd.read_csv('C:/Users/jingying/Desktop/dataset-main/data/SCREENINGHEALTHSTATUS.CSV',encoding='ISO-8859-1')
#Print the first few lines of data_SCREENINGHEALTHSTATUS to test
data_SCREENINGHEALTHSTATUS.head()


# In[68]:


numbers_to_count = [1, 2]
count_dict = {}
labels = {
    1: "Number of persons with high blood pressure",
    2: "Number of people without high blood pressure"
}

for number in numbers_to_count:
    value_counts = data_SCREENINGHEALTHSTATUS['HYPERTENSION'].value_counts()
    count = value_counts.get(number, 0)
    count_dict[labels[number]] = count

for label, count in count_dict.items():
    print(f'{label}：{count}')

result_df = pd.DataFrame({'Disease condition': count_dict.keys(), 'Number of people': count_dict.values()})
sns.set(style="whitegrid")
plt.figure(figsize=(5, 1))
ax = sns.barplot(y='Disease condition', x='Number of people', data=result_df)  # Exchange of x and y parameters
ax.set(ylabel='Disease condition', xlabel='Number of people')  # Updated label settings
plt.title('Frequency distribution of hypertension')
plt.show()


numbers_to_count = [1, 2]
count_dict = {}
labels = {
    1: "Number of persons with diabetes",
    2: "Number of people without diabetes"
}

for number in numbers_to_count:
    value_counts = data_SCREENINGHEALTHSTATUS['DIABETES'].value_counts()
    count = value_counts.get(number, 0)
    count_dict[labels[number]] = count

for label, count in count_dict.items():
    print(f'{label}：{count}')

result_df = pd.DataFrame({'Disease condition': count_dict.keys(), 'Number of people': count_dict.values()})
sns.set(style="whitegrid")
plt.figure(figsize=(5, 1))
ax = sns.barplot(y='Disease condition', x='Number of people', data=result_df)  # Exchange of x and y parameters
ax.set(ylabel='Disease condition', xlabel='Number of people')  # Updated label settings
plt.title('Frequency distribution of diabetes')
plt.show()


numbers_to_count = [1, 2]
count_dict = {}
labels = {
    1: "Number of persons with depression",
    2: "Number of people without depression"
}

for number in numbers_to_count:
    value_counts = data_SCREENINGHEALTHSTATUS['DEPRESSION'].value_counts()
    count = value_counts.get(number, 0)
    count_dict[labels[number]] = count

for label, count in count_dict.items():
    print(f'{label}：{count}')

result_df = pd.DataFrame({'Disease condition': count_dict.keys(), 'Number of people': count_dict.values()})
sns.set(style="whitegrid")
plt.figure(figsize=(5, 1))
ax = sns.barplot(y='Disease condition', x='Number of people', data=result_df)  # Exchange of x and y parameters
ax.set(ylabel='Disease condition', xlabel='Number of people')  # Updated label settings
plt.title('Frequency distribution of depression')
plt.show()

data_SCREENINGHEALTHSTATUS = data_SCREENINGHEALTHSTATUS.dropna(subset=['FEELAGE'])
data_SCREENINGHEALTHSTATUS = data_SCREENINGHEALTHSTATUS[data_SCREENINGHEALTHSTATUS['FEELAGE'] != 0]
# Setting the graphic style
sns.set(style="whitegrid")
# Access to age data
ages = data_SCREENINGHEALTHSTATUS['FEELAGE']
# Mapping of density
plt.figure(figsize=(10, 6))
sns.kdeplot(ages, fill=True, label="Age distribution")  # Use fill=True instead of shade=True.
plt.xlabel('Age')  # x-axis labels
plt.ylabel('Density')  # y轴标签
plt.title('Age Density Plot')  # y-axis labels
plt.legend()  # Show legend
plt.xlim(0, 130)  # Setting the x-axis range
plt.show()


# In[211]:


path = 'C:/Users/jingying/Desktop/dataset-main/data'
file_names = ["LSNS6.csv", "EQ5D.csv", "GAD7.csv"]

# Merging paths and filenames
full_file_paths = [os.path.join(path, file) for file in file_names]

dataframes = [pd.read_csv(file) for file in full_file_paths]
# Initially set merged_data as the first dataframe
merged_data = dataframes[0]
# Use a for loop to merge with other dataframes in turn
for df in dataframes[1:]:
    merged_data = pd.merge(merged_data, df, on='PID', how='outer')
merged_data = merged_data.drop(columns=['SHOULDSYNC','TS','VID','SID','POS','NID','VID_x','SID_x','POS_x','NID_x','SHOULDSYNC_x','TS_x','VID_y','SID_y','POS_y','NID_y','SHOULDSYNC_y','TS_y','PID'])
merged_data.dropna(inplace=True)
#merged_data.fillna(merged_data.mean(),inplace Truel)
#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#imp.fit(merged_data)
#merged_data = pd.DataFrame(merged_data = imp.transform(merged_data), columns = merged_data.columns)
merged_data.head()
#Save the merged file
#merged_data.to_csv('C:/Users/jingying/Desktop/merged_data.csv', index=False)


# In[219]:


df_encoded = pd.get_dummies(df, columns=df.columns)
print(df_encoded.head())


# In[220]:


merged_data.describe()


# In[221]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[218]:


target=merged_data['GAD6']
X=merged_data
X.drop(['GAD6'],axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
clf = RandomForestClassifier(max_depth=20, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))


# In[217]:


from sklearn.model_selection import GridSearchCV
# Define the parameters to be tested
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialising GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# conduct a search
grid_search.fit(X_train, y_train)
# Output Optimal Parameters
print(grid_search.best_params_)


# In[141]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[140]:


importances = clf.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure(figsize=(12, 12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score

target = df_encoded['GAD6']
X = df_encoded.drop(['GAD6_0.0','GAD6_1.0','GAD6_2.0','GAD6_3.0'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0)
clf = RandomForestClassifier(max_depth=20, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calculating the accuracy of multi-label classification using accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

target = merged_data['GAD6']
X = merged_data
X.drop(['GAD6'], axis=1, inplace=True)

# SVM models are sensitive to the scale of the features, so normalisation is often required
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=0)

clf = SVC(kernel='linear')  # Use linear kernels, but try other kernels such as 'rbf'
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


# In[156]:


import numpy as np
import matplotlib.pyplot as plt

coefficients = svm_model.coef_[0]  # Depends on the number of categories of the target variable


# Get feature name
features = X.columns

# Get sorted coefficient index
indices = np.argsort(coefficients)

# drawings
plt.figure(figsize=(12, 12))
plt.title('Feature Importances for Linear SVM')
plt.barh(range(len(indices)), coefficients[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Coefficient Value')
plt.show()


# In[ ]:





# In[ ]:




