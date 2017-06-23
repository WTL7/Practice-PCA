
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

scaler_data = scaler.transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

pca.fit(scaler_data)

k_scaler_data =pca.transform(scaler_data)  # reduce to k dimentions

#plot

fig = plt.figure(figsize = (8,6))
ax = Axes3D(fig)
ax.scatter(k_scaler_data[:,0], k_scaler_data[:,1], #k_scaler_data[:,2], 
        c = cancer.target, cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
# actually two is really good already


print pca.components_

df_component = pd.DataFrame(pca.components_, columns = cancer.feature_names)

plt.figure(figsize = (12,6))
sns.heatmap(df_component, cmap ='coolwarm')

plt.show()

from sklearn.model_selection import train_test_split

X = k_scaler_data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

from sklearn.linear_model import LogisticRegression

lg_reg = LogisticRegression()

lg_reg.fit(X_train, y_train)

lg_pred = lg_reg.predict(X_test)

from sklearn.metrics import classification_report

print 'PCA'
print classification_report(y_test, lg_pred)    
# Amazing!!! We use two dimensions to predict a great result

# Try origin data

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

lg_reg = LogisticRegression()

lg_reg.fit(X_train, y_train)

lg_pred = lg_reg.predict(X_test)

print 'Origin'
print classification_report(y_test, lg_pred)


# Try scaled origin data

X = scaler_data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

lg_reg = LogisticRegression()

lg_reg.fit(X_train, y_train)

lg_pred = lg_reg.predict(X_test)

print 'Scaled Origin'
print classification_report(y_test, lg_pred)