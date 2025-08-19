import numpy as np
import matplotlib.pyplot as plt

def calculate_loss_function(real_data, predicted_data):
    return np.mean((real_data - predicted_data)**2)

def plot_decoding(t, true_signal, decoded_signal, nengo_decoded, title="Delay decoding"):
    plt.figure(figsize=(12, 4))
    plt.plot(t, true_signal, label="Input signal", linewidth=1, linestyle='--')
    plt.plot(t, decoded_signal, label="Delay decoding", linewidth=1, alpha=0.6)
    plt.plot(t, nengo_decoded, label="Instantaneous decoding", linewidth=1, alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Signal value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_coeffs(activity, input_values): 
    coeffs, *_ = np.linalg.lstsq(activity, input_values, rcond=None)
    return coeffs
