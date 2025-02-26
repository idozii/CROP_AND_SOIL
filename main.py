import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the data
data = pd.read_csv('data/data.csv')
print(data.info())

# Split the data into features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute the missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
model = Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Fit the model
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400, callbacks=[early_stop])

# Evaluate the model
losses = pd.DataFrame(model.history.history)
losses.plot()