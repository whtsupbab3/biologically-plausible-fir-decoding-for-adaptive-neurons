import numpy as np
import pandas as pd
import nengo
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from infrastructure.utils import calculate_coeffs, calculate_loss_function, plot_decoding
from infrastructure.DelayNetwork import DelayNetwork

# Constants
n_neurons = 300
readout_synapse = 0.05
neuron_type = nengo.AdaptiveLIF(tau_n=0.5, inc_n=0.01)
run_time = 10.0
dt = 0.001
default_delay_mode = "discrete"

np.random.seed(42)

def train_and_evaluate_decoders(train_input, test_input, delay_mode):
    with nengo.Network(seed=10) as model:
        input_node = nengo.Node(train_input, size_out=1)
        delay_network = DelayNetwork(num_neurons=n_neurons, readout_synapse=readout_synapse, neuron_type=neuron_type, delay_mode=delay_mode)
        nengo.Connection(input_node, delay_network.ens, synapse=None)

        p_input = nengo.Probe(input_node, sample_every=dt)
        p_delay_activity = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)  

    with nengo.Simulator(model) as sim:
        sim.run(run_time)

    t = sim.trange()
    coeffs = calculate_coeffs(sim.data[p_delay_activity], sim.data[p_input])

    with nengo.Network(seed=10) as model:
        input_node = nengo.Node(test_input, size_out=1)
        delay_network = DelayNetwork(num_neurons=n_neurons, decoder_weights=coeffs, readout_synapse=readout_synapse, neuron_type=neuron_type, delay_mode=delay_mode)
        nengo.Connection(input_node, delay_network.ens, synapse=None)

        p_input = nengo.Probe(input_node, sample_every=dt)
        p_delay_decoded = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

    with nengo.Simulator(model) as sim:
        sim.run(run_time)

    decoded = sim.data[p_delay_decoded]
    loss = calculate_loss_function(sim.data[p_input], decoded)

    if delay_mode is "range":
        plot_title = f"Range Delay Decoding (loss={loss:.6f})"
    elif delay_mode is "discrete":
        plot_title = f"Discrete Delay Decoding (loss={loss:.6f})" 
    else:
        plot_title = f"Instantaneous Decoding (loss={loss:.6f})"

    try:
        plt.figure(figsize=(12, 4))
        plt.plot(t, sim.data[p_input], label="Input signal", linewidth=1, linestyle='--')
        plt.plot(t, decoded, label="Delay decoding", linewidth=1, alpha=0.6)

        plt.xlabel("Time (s)")
        plt.ylabel("Signal value")
        plt.title(plot_title)
        plt.legend()
        filename = f"{delay_mode}_delay_decoding_syn={readout_synapse}.png"
        save_path = os.path.join(os.path.join(current_dir, "../figures"), filename)
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    except Exception:
        pass

    return {
        "t": t,
        "input": sim.data[p_input],
        "coeffs": coeffs,
        "decoded": decoded,
        "loss": loss,
    }

def run_experiments():

    run_time = 5.0

    train_white_signal = nengo.processes.WhiteSignal(period=run_time, high=10, rms=0.25, seed=121)
    test_white_signal = nengo.processes.WhiteSignal(period=run_time, high=10, rms=0.25, seed=223)

    print("Running delay decoder experiment...")
    train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode="range")
    train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode="discrete")
    train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode="zero")

if __name__ == "__main__":
    run_experiments()
