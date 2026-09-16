import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
d=pd.read_csv("student_performance.csv").drop("student_id",axis=1)
X = pd.get_dummies(d.drop("final_grade", axis=1), drop_first=True)
e = LabelEncoder()
Y = e.fit_transform(d.final_grade)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.25, random_state=0, stratify=Y)

m = DecisionTreeClassifier(random_state=0)
m.fit(Xtr, Ytr)
p = m.predict(Xte)
prob = m.predict_proba(Xte)

print("Accuracy:", accuracy_score(Yte,p))
print("Precision:", precision_score(Yte,p,average="weighted",zero_division=0))
print("Recall:", recall_score(Yte,p,average="weighted",zero_division=0))
print("F1 Score:", f1_score(Yte,p,average="weighted",zero_division=0))

sns.heatmap(confusion_matrix(Yte,p), annot=True, fmt="d",xticklabels=e.classes_, yticklabels=e.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")

yb = pd.get_dummies(Yte).reindex(columns=range(len(e.classes_)),fill_value=0)
plt.figure()
for i in range(prob.shape[1]):
    f,t,_ = roc_curve(yb.iloc[:,i],prob[:,i])
    plt.plot(f,t,label=f"{e.classes_[i]} AUC={auc(f,t):.2f}")
plt.plot([0,1],[0,1],"r--")
plt.title("ROC Curve"); plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate"); plt.legend(); plt.grid()

plt.figure(figsize=(20,10))
plot_tree(m, feature_names=X.columns, class_names=e.classes_,filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree")

plt.show()
