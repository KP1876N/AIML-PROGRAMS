
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from scipy.stats import mode

d=load_iris()
X=d.data
y=d.target

X=StandardScaler().fit_transform(X)

m=GaussianMixture(n_components=3,random_state=42)
m.fit(X)
labels=m.predict(X)

mapped=np.zeros_like(labels)
for c in np.unique(labels):
    mapped[labels==c]=mode(y[labels==c],keepdims=True).mode[0]

print("Accuracy:",round(accuracy_score(y,mapped),4))
print("Precision:",round(precision_score(y,mapped,average="weighted",zero_division=0),4))
print("Recall:",round(recall_score(y,mapped,average="weighted",zero_division=0),4))
print("F1 Score:",round(f1_score(y,mapped,average="weighted",zero_division=0),4))
print("Silhouette Score:",round(silhouette_score(X,labels),4))

plt.scatter(d.data[:,2],d.data[:,3],c=y)
plt.title("Actual Iris Classes")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

plt.scatter(d.data[:,2],d.data[:,3],c=labels)
plt.title("EM - GMM Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
