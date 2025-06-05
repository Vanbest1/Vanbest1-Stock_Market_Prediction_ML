def load_data(file_path):
    import pandas as pd
    
    # Load the dataset from the specified file path
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Fill missing values (you can customize this method)
    data = data.fillna(method='ffill')
    
    return data

def preprocess_data(file_path):
    # Load the data
    data = load_data(file_path)
    
    # Clean the data
    cleaned_data = clean_data(data)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    file_path = 'path_to_your_data.csv'  # Update this path
    processed_data = preprocess_data(file_path)
    print(processed_data.head())