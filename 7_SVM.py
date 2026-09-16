import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve,auc

d=pd.read_csv("heart.csv")
X,Y=d.iloc[:,:-1],d.iloc[:,-1]

Xt,Xv,Yt,Yv=train_test_split(X,Y,test_size=.25,random_state=0)

s=StandardScaler()
Xt=s.fit_transform(Xt); Xv=s.transform(Xv)

m=SVC(kernel="linear",probability=True,random_state=0)
m.fit(Xt,Yt)
p=m.predict(Xv); prob=m.predict_proba(Xv)[:,1]

print("Accuracy:",accuracy_score(Yv,p))
print("F1 Score:",f1_score(Yv,p))

sns.heatmap(confusion_matrix(Yv,p),annot=True,fmt="d",
            xticklabels=["No Disease","Disease"],
            yticklabels=["No Disease","Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")

f,t,_=roc_curve(Yv,prob); a=auc(f,t)
plt.figure()
plt.plot(f,t,label=f"AUC = {a:.2f}")
plt.plot([0,1],[0,1],"r--",label="Random Classifier")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.grid()

plt.show()
print("AUC:",a)
