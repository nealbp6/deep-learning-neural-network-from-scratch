# plot.py
import numpy as np 
import matplotlib.pyplot as plt

def plot(y_pred, y_true, symbol, interval):

    plt.title(f"Price Prediction Of {symbol}")
    plt.xlabel(f"Time In {interval}")
    plt.ylabel("Price In $")
    
    x = np.arange(len(y_pred))
    plt.plot(x, y_true, color="green", label="True Prices")
    plt.plot(x, y_pred, color="red", label="Predicted Prices")
    
    plt.legend()
    plt.show()
