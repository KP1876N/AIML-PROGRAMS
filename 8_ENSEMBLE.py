import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import *

d=pd.read_csv("heart.csv")
X,Y=d.iloc[:,:-1],d.iloc[:,-1]

Xt,Xv,Yt,Yv=train_test_split(X,Y,test_size=.25,random_state=0)

dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(n_estimators=100,random_state=0)
svm=SVC(kernel="linear",probability=True,random_state=0)

m=VotingClassifier(
    estimators=[("dt",dt),("rf",rf),("svm",svm)],
    voting="soft"
)

m.fit(Xt,Yt)
p=m.predict(Xv)
prob=m.predict_proba(Xv)[:,1]

print("Accuracy:",accuracy_score(Yv,p))
print("F1 Score:",f1_score(Yv,p))

sns.heatmap(confusion_matrix(Yv,p),annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

f,t,_=roc_curve(Yv,prob)
a=auc(f,t)

plt.plot(f,t,label=f"AUC = {a:.2f}")
plt.plot([0,1],[0,1],"r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
