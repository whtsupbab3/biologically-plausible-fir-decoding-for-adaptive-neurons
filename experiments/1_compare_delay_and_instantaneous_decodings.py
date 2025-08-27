import numpy as np
import pandas as pd
import nengo
import sys
import os
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from infrastructure.utils import calculate_coeffs, calculate_loss_function, add_noise_to_activity
from infrastructure.DelayNetwork import DelayNetwork

# Constants
n_neurons = 300
readout_synapse = 0.05
neuron_type = nengo.AdaptiveLIF(tau_n=0.5, inc_n=0.01)
run_time = 10.0
dt = 0.001
default_delay_mode = "discrete"
input_high = 0.5

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
    if delay_mode == "zero":
        activity = sim.data[p_delay_activity]
    else: 
        activity = add_noise_to_activity(sim.data[p_delay_activity])
    coeffs = calculate_coeffs(activity, sim.data[p_input])

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
        plt.plot(t, sim.data[p_input], linewidth=1, linestyle='--')
        plt.plot(t, decoded, linewidth=1, alpha=0.6)

        plt.xlabel("Time (s)")
        plt.ylabel("Signal value")
        plt.title(plot_title)
        plt.legend()
        filename = f"{delay_mode}_delay_decoding_syn={readout_synapse}_lf.pdf"
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
    global run_time
    run_time = 5.0

    n_experiments = 10
    base_train_seed = 121
    base_test_seed = 223
    modes = ["range", "discrete", "zero"]

    losses = {m: [] for m in modes}

    print(f"Running {n_experiments} experiments for modes: {modes} (run_time={run_time}s)")
    for i in range(n_experiments):
        train_seed = base_train_seed + i
        test_seed = base_test_seed + i

        train_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=train_seed)
        test_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=test_seed)

        print(f"Experiment {i+1}/{n_experiments} - train_seed={train_seed}, test_seed={test_seed}")
        for mode in modes:
            res = train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode=mode)
            losses[mode].append(res["loss"])

    try:
        x = np.arange(1, n_experiments + 1)
        plt.figure(figsize=(10, 5))
        for mode in modes:
            plt.plot(x, losses[mode], marker='o', label=f"{mode} (mean={np.mean(losses[mode]):.4e})")

        plt.xlabel("Experiment #")
        plt.ylabel("Loss")
        plt.title("Decoding loss across experiments (high=10)")
        plt.legend()
        plt.grid(True)

        filename = f"loss_comparison_syn={readout_synapse}_lf.pdf"
        save_path = os.path.join(os.path.join(current_dir, "../figures"), filename)
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved loss comparison plot to {save_path}")
        except Exception as e:
            print(f"Failed to save loss comparison plot: {e}")

    except Exception as e:
        print(f"Failed to generate loss comparison plot: {e}")

    for mode in modes:
        vals = np.array(losses[mode])
        print(f"{mode} losses: {vals}")
        print(f"{mode} mean loss: {np.mean(vals):.6e}, std: {np.std(vals):.6e}")

if __name__ == "__main__":
    run_experiments()
