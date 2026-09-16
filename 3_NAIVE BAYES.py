import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

d=pd.read_csv("heart.csv")
X=d.iloc[:,:-1]
Y=d.iloc[:,-1]

Xt,Xv,Yt,Yv=train_test_split(X,Y,test_size=.25,random_state=0)

m=GaussianNB()
m.fit(Xt,Yt)
p=m.predict(Xv)
prob=m.predict_proba(Xv)[:,1]

print("Accuracy:",accuracy_score(Yv,p))
print("F1 Score:",f1_score(Yv,p))

sns.heatmap(confusion_matrix(Yv,p),annot=True,fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr,tpr,_=roc_curve(Yv,prob)
plt.plot(fpr,tpr,label="AUC=%.2f"%auc(fpr,tpr))
plt.plot([0,1],[0,1],"r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.show()
