import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
path = r"C:\Users\nandh\Downloads\AIML\mnist.npz"

with np.load(path) as data:
    xtr, ytr = data["x_train"], data["y_train"]
    xte, yte = data["x_test"], data["y_test"]

# Use only 10,000 training images to make training faster
xtr, ytr = xtr[:10000], ytr[:10000]

# Preprocessing
xtr = xtr.reshape(-1,784).astype("float32") / 255
xte = xte.reshape(-1,784).astype("float32") / 255

ytr = to_categorical(ytr,10)
yte = to_categorical(yte,10)

# MLP model
m = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

m.compile(optimizer=Adam(),
          loss="categorical_crossentropy",
          metrics=["accuracy"])

# Train
h = m.fit(xtr, ytr,
          epochs=3,
        batch_size=256,
          validation_split=0.2,
          verbose=1)

# Test
loss, acc = m.evaluate(xte, yte, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# Window 1 - Accuracy
plt.figure()
plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training","Validation"])
plt.show()

# Window 2 - Loss
plt.figure()
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training","Validation"])
plt.show()

# Window 3 - Sample predictions
pred = m.predict(xte[:10], verbose=0)

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(xte[i].reshape(28,28), cmap="gray")
    plt.title("Pred: " + str(np.argmax(pred[i])))
    plt.axis("off")
plt.tight_layout()
plt.show()
