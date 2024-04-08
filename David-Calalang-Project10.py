# Question 1
import pandas as pd
pd.set_option('display.max_columns', None)

mycols = [
    'DepDelay', 'ArrDelay', 'Distance',
    'CarrierDelay', 'WeatherDelay',
    'DepTime', 'ArrTime', 'Diverted', 'AirTime'
]

# Part A
mycoltypes = {
    'DepDelay': 'float64',
    'ArrDelay': 'float64',
    'Distance': 'float64',
    'CarrierDelay': 'float64',
    'WeatherDelay': 'float64',
    'DepTime': 'float64',
    'ArrTime': 'Int64',
    'Diverted': 'float64',
    'AirTime': 'float64'
}

# Part B
df = pd.read_csv("/anvil/projects/tdm/data/flights/2014.csv", nrows=10000, usecols=mycols, dtype=mycoltypes)


# Question 2
# Part A
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time

# Part B
for col in df.columns:
    df[col] = df[col].fillna(df[col].median())
    
    
# Question 3
# Splitting features and labels
features = df.drop('ArrDelay', axis=1)
labels = df['ArrDelay']

# Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Part A
# Features serve as the variables we are taking into account when we are making a prediction, while labels represent the actual expected outcome.

# Part B
# Having a training and test split allows for us to use some part of our data in training our model, and by taking the outputs of that training, we can validate our model predictions using the testing data.
# In this case we use a large (80%) of our data in training our model, with the rest being used to validate our data.


# Question 4
# Part A
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Part B
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(14)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(14)


# Question 5
# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Train
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Cleanup
del X_train_scaled, X_test_scaled, train_dataset, test_dataset

# This neural network model analyzes our dataset (using the training data) in order to identify patterns, creating various layers by parsing through our data iteratively (10 times, 10 epochs). With this, the model
# will create predictions based on what its learned in the patterns its collected.