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
input_high = 10
taus = [round(0.1 * i, 1) for i in range(1, 10)] + [1.0]
incs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
modes = ["range", "discrete", "zero"]
n_experiments = 3
base_train_seed = 321
base_test_seed = 421
tau_val = 0.2

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

    return {
        "t": t,
        "input": sim.data[p_input],
        "coeffs": coeffs,
        "decoded": decoded,
        "loss": loss,
    }

def run_experiments_adaptation_tc():
    global neuron_type
    mean_losses = {m: [] for m in modes}

    print(f"Running adaptation sweep for taus={taus}, modes={modes}, n_experiments={n_experiments}, run_time={run_time}s")

    for tau in taus:
        neuron_type = nengo.AdaptiveLIF(tau_n=tau, inc_n=0.01)
        print(f"\n=== tau_n={tau} ===")

        losses = {m: [] for m in modes}

        for i in range(n_experiments):
            train_seed = base_train_seed + i
            test_seed = base_test_seed + i

            train_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=train_seed)
            test_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=test_seed)

            print(f" experiment {i+1}/{n_experiments} (train_seed={train_seed}, test_seed={test_seed})")
            for mode in modes:
                try:
                    res = train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode=mode)
                    losses[mode].append(res['loss'])
                except Exception as e:
                    print(f"  failed run for mode={mode}: {e}")

        for m in modes:
            vals = np.array(losses[m]) if len(losses[m]) > 0 else np.array([np.nan])
            mean_val = np.nanmean(vals)
            mean_losses[m].append(mean_val)
            print(f"  mode={m} mean loss over {len(vals)} runs: {mean_val:.6e}")

    try:
        x = np.array(taus)
        plt.figure(figsize=(8, 5))
        for m in modes:
            plt.plot(x, mean_losses[m], marker='o', label=f"{m} (mean={np.nanmean(mean_losses[m]):.4e})")

        plt.xlabel("tau_n (s)")
        plt.ylabel("Mean decoding loss (MSE)")
        plt.title("Decoding loss vs adaptation time constant (tau_n) ")
        plt.grid(True)
        plt.legend()

        filename = f"tau_vs_loss_comparison_syn={readout_synapse}_hf.pdf"
        save_path = os.path.join(os.path.join(current_dir, "../figures/3"), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved tau sweep plot to {save_path}")
        except Exception as e:
            print(f"Failed to save tau sweep plot: {e}")

    except Exception as e:
        print(f"Failed to generate tau sweep plot: {e}")

def run_experiments_adaptation_inc():
    global neuron_type
    mean_losses = {m: [] for m in modes}

    print(f"Running adaptation increment sweep for incs={incs}, modes={modes}, n_experiments={n_experiments}, run_time={run_time}s")

    for inc in incs:
        neuron_type = nengo.AdaptiveLIF(tau_n=tau_val, inc_n=inc)
        print(f"\n=== inc_n={inc} (tau_n={tau_val}) ===")

        losses = {m: [] for m in modes}

        for i in range(n_experiments):
            train_seed = base_train_seed + i
            test_seed = base_test_seed + i

            train_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=train_seed)
            test_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=test_seed)

            print(f" experiment {i+1}/{n_experiments} (train_seed={train_seed}, test_seed={test_seed})")
            for mode in modes:
                try:
                    res = train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode=mode)
                    losses[mode].append(res['loss'])
                except Exception as e:
                    print(f"  failed run for mode={mode}: {e}")

        for m in modes:
            vals = np.array(losses[m]) if len(losses[m]) > 0 else np.array([np.nan])
            mean_val = np.nanmean(vals)
            mean_losses[m].append(mean_val)
            print(f"  mode={m} mean loss over {len(vals)} runs: {mean_val:.6e}")

    try:
        x = np.array(incs)
        plt.figure(figsize=(8, 5))
        for m in modes:
            plt.plot(x, mean_losses[m], marker='o', label=f"{m} (mean={np.nanmean(mean_losses[m]):.4e})")

        plt.xlabel("adaptation increment (inc_n)")
        plt.ylabel("Mean decoding loss (MSE)")
        plt.title("Decoding loss vs adaptation increment (inc_n)")
        plt.grid(True)
        plt.legend()

        filename = f"inc_vs_loss_comparison_syn={readout_synapse}_hf.pdf"
        save_path = os.path.join(os.path.join(current_dir, "../figures/3"), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved inc sweep plot to {save_path}")
        except Exception as e:
            print(f"Failed to save inc sweep plot: {e}")

    except Exception as e:
        print(f"Failed to generate inc sweep plot: {e}")

if __name__ == "__main__":
    #run_experiments_adaptation_tc()
    run_experiments_adaptation_inc()
