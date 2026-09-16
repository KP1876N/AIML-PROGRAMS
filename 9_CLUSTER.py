import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris=load_iris()
X=iris.data

m=KMeans(n_clusters=3,random_state=0,n_init=10)
m.fit(X)
labels=m.predict(X)

print("Silhouette Score:",silhouette_score(X,labels))

plt.scatter(X[:,0],X[:,1],c=labels,cmap="viridis")
plt.scatter(m.cluster_centers_[:,0],m.cluster_centers_[:,1],s=200,c="red",marker="X",label="Centroids")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()
