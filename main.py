# main.py
from fetch_data import get_data
from preprocessing_data import create_windows, normalize_data, denormalize_data, split_data

from ffnn import model
from evaluation import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from plot import plot

if __name__ == "__main__":
    # data constants
    symbol = "TSLA"
    interval = "1h"
    window_size = 40
    train_ratio = 0.8

    # training constants 
    learning_rate = 0.1
    epochs = 1000 

    # data preperation
    data = get_data(symbol, interval) # log retuns
    
    data, mean, std = normalize_data(data) 

    train, test = split_data(data, train_ratio)

    x_train, y_train = create_windows(train, window_size)
    x_test, y_test = create_windows(test, window_size)

    # ffnn model
    y_test_pred = model(x_train, y_train, x_test, learning_rate, epochs)

    # denormalize
    y_pred_denormalized = denormalize_data(y_test_pred, mean, std)
    y_test_denormalized = denormalize_data(y_test, mean, std)
    
    # evaluation
    mse  = mean_squared_error(y_test_denormalized, y_pred_denormalized)
    rmse = root_mean_squared_error(y_test_denormalized, y_pred_denormalized)
    mae  = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
    r2   = r2_score(y_test_denormalized, y_pred_denormalized)

    print(f"MSE:  {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE:  {mae}")
    print(f"R²:   {r2}")

    # plot
    plot(y_pred_denormalized, y_test_denormalized, symbol, interval)