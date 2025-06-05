from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd
import joblib

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def train_lstm(X_train, y_train, input_shape, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    # Load your dataset here
    data = pd.read_csv('path_to_your_data.csv')
    
    # Preprocess your data
    # Assuming 'features' and 'target' are defined
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = train_random_forest(X_train, y_train)
    svr_model = train_svr(X_train, y_train)
    
    # Reshape data for LSTM
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    lstm_model = train_lstm(X_train_lstm, y_train.values, input_shape=(X_train.shape[1], 1))
    
    # Save models
    save_model(rf_model, 'random_forest_model.pkl')
    save_model(svr_model, 'svr_model.pkl')
    lstm_model.save('lstm_model.h5')

if __name__ == "__main__":
    main()