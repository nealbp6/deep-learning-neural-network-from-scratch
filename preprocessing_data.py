# preprocessing_data.py
import numpy as np

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data =  (data - mean) / std
    return normalized_data , mean, std

def denormalize_data(normalized_data, mean, std):
    return normalized_data * std + mean 

def split_data(data, train_ratio=0.8):
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def create_windows(data, window_size):
    inputs = []
    labels = []

    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        labels.append(data[i + window_size])

    return np.array(inputs), np.array(labels).reshape(-1, 1)