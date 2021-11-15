

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import clf
from sklearn.model_selection import train_test_split
sns.set()

heart = pd.read_csv('heart.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = heart.copy()
from sklearn.preprocessing import StandardScaler
data = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])

# Separating X and y
y = data['target']
X = data.drop(['target'], axis = 1)

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
