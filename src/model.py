import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

# Parameters 
time_steps = 100  # Number of time steps per sequence
stride = 50       # Step size for overlapping windows
num_features = 4  # Number of features (LX, LY, RX, RY)

raw_data = pd.read_csv("../Data/processed/combined_raw_data.csv")

def create_sequences(data, time_steps, stride):
    sequences = []
    labels = []
    for label in data['label'].unique():  # label is 0 or 1
        label_data = data[data['label'] == label]
        for i in range(0, len(label_data) - time_steps, stride):
            window = label_data.iloc[i:i + time_steps]
            if len(window) == time_steps:  # Ensure each sequence is the correct length
                sequence = window[['LX', 'LY', 'RX', 'RY']].values
                sequences.append(sequence)
                labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(raw_data, time_steps, stride)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, num_features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Bidirectional(LSTM(100, return_sequences=True)),
    Bidirectional(LSTM(100)),

    Dense(200, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid
])


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
model.save("../Data/saved_model/dyslexia_detection_model.h5")
print("Model saved successfully.")
