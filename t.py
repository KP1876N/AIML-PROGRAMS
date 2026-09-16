import numpy as np

path = r"C:\Users\nandh\Downloads\AIML\mnist.npz"

with np.load(path) as d:
    print(d.files)
    print(d["x_train"].shape)
    print(d["x_test"].shape)
