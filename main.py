import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers

df = pd.read_csv("cars.csv")

y = df.mpg
X = df[["cylinders", "acceleration"]]

model = Sequential()
model.add(Dense(1, activation="linear", input_shape=(2,)))
opt = optimizers.SGD()
model.compile(loss="mse", optimizer=opt, metrics=["mse"])
model.summary()

model.fit(X, y, epochs=10, batch_size=32, verbose=2, validation_split=0.3, shuffle=True)
