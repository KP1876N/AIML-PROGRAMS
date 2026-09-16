import bayespy as bp
import numpy as np
import csv

with open("heart.csv") as f:
    data = np.array(list(csv.reader(f))[1:])

# Age category, Gender, Heart disease
X = np.array([
    [
        0 if int(r[0]) <= 40 else 1 if int(r[0]) <= 60 else 2,
        int(r[1]),
        int(r[-1])
    ]
    for r in data
])

N = len(X)

# Age
p_age = bp.nodes.Dirichlet(np.ones(3))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(X[:, 0])

# Gender
p_gender = bp.nodes.Dirichlet(np.ones(2))
gender = bp.nodes.Categorical(p_gender, plates=(N,))
gender.observe(X[:, 1])

# Heart disease
p_heart = bp.nodes.Dirichlet(np.ones(2), plates=(3, 2))

heart = bp.nodes.MultiMixture(
    [age, gender],
    bp.nodes.Categorical,
    p_heart
)

heart.observe(X[:, 2])
p_heart.update()

a = int(input("Enter Age: "))
g = int(input("Enter Gender (0=Female, 1=Male): "))

a = 0 if a <= 40 else 1 if a <= 60 else 2

result = bp.nodes.MultiMixture(
    [a, g],
    bp.nodes.Categorical,
    p_heart
).get_moments()[0][0]

print("Probability of Heart Disease =", result)
