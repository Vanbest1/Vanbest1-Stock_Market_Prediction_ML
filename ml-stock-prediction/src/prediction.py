def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def make_prediction(model, input_data):
    return model.predict(input_data)

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def predict_stock_prices(model_path, input_data, scaler):
    model = load_model(model_path)
    predictions = make_prediction(model, input_data)
    return inverse_transform(scaler, predictions)