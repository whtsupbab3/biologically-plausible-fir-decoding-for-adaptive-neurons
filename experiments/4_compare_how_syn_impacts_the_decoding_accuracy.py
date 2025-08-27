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
neuron_type = nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)
run_time = 10.0
dt = 0.001
default_delay_mode = "discrete"
modes = ["range", "discrete", "zero"]
input_high = 10
readout_syns = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
n_experiments = 3
base_train_seed = 767
base_test_seed = 9

np.random.seed(42)

def plot_loss_vs_syn(rows, modes=None, save_dir=None):
    if isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        df = rows.copy()

    if modes is None:
        modes = sorted(df['mode'].unique())

    agg = df.groupby(['syn', 'mode'])['loss'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(8, 5))

    mode_labels = {
        'range': 'Range delay',
        'discrete': 'Discrete delay',
        'zero': 'Instantaneous'
    }

    prop_cycle = plt.rcParams.get('axes.prop_cycle')
    colors = prop_cycle.by_key().get('color', []) if prop_cycle is not None else []

    for idx, mode in enumerate(modes):
        sub = agg[agg['mode'] == mode].sort_values('syn')
        if sub.empty:
            continue
        x = sub['syn'].values
        y = sub['mean'].values

        color = colors[idx % len(colors)] if colors else None

        plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=1.5,
                 label=mode_labels.get(mode, mode))

    plt.xlabel('Readout synapse (syn)')
    plt.ylabel('Loss')
    plt.title(f'Loss vs readout synapse across decoding modes (high={input_high})')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title='Decoding mode', loc='upper left')

    if save_dir is None:
        save_dir = os.path.join(current_dir, "../figures/4")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"loss_vs_syn_high={input_high}.pdf")
    try:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved loss vs syn plot to {filename}")
    except Exception as e:
        print(f"Failed to save loss vs syn plot: {e}")

def train_and_evaluate_decoders(train_input, test_input, delay_mode, readout_syn, plot=False):
    with nengo.Network(seed=10) as model:
        input_node = nengo.Node(train_input, size_out=1)
        delay_network = DelayNetwork(num_neurons=n_neurons, readout_synapse=readout_syn, neuron_type=neuron_type, delay_mode=delay_mode)
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
        delay_network = DelayNetwork(num_neurons=n_neurons, decoder_weights=coeffs, readout_synapse=readout_syn, neuron_type=neuron_type, delay_mode=delay_mode)
        nengo.Connection(input_node, delay_network.ens, synapse=None)

        p_input = nengo.Probe(input_node, sample_every=dt)
        p_delay_decoded = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

    with nengo.Simulator(model) as sim:
        sim.run(run_time)

    decoded = sim.data[p_delay_decoded]
    loss = calculate_loss_function(sim.data[p_input], decoded)

    # if delay_mode == "range":
    #     plot_title = f"Range Delay Decoding (loss={loss:.6f}, syn={readout_syn})"
    # elif delay_mode == "discrete":
    #     plot_title = f"Discrete Delay Decoding (loss={loss:.6f}, syn={readout_syn})"
    # else:
    #     plot_title = f"Instantaneous Decoding (loss={loss:.6f}, syn={readout_syn})"

    # if plot: 
    #     try:
    #         plt.figure(figsize=(12, 4))
    #         plt.plot(t, sim.data[p_input], linewidth=1, linestyle='--')
    #         plt.plot(t, decoded, linewidth=1, alpha=0.6)

    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Signal value")
    #         plt.title(plot_title)
    #         plt.legend()

    #         filename = f"{delay_mode}_delay_decoding_syn={readout_syn}_hf.pdf"
    #         save_path = os.path.join(os.path.join(current_dir, "../figures"), filename)
    #         try:
    #             plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #             print(f"Saved plot to {save_path}")
    #         except Exception as e:
    #             print(f"Failed to save plot: {e}")

    #     except Exception:
    #         pass

    return {
        "t": t,
        "input": sim.data[p_input],
        "coeffs": coeffs,
        "decoded": decoded,
        "loss": loss,
    }

def run_experiments():
    rows = []

    print(f"Running {n_experiments} experiments across synapses: {readout_syns} and modes: {modes} (run_time={run_time}s)")
    for syn in readout_syns:
        losses = {m: [] for m in modes}
        print(f"\n=== Testing readout synapse = {syn} ===")
        for i in range(n_experiments):
            train_seed = base_train_seed + i
            test_seed = base_test_seed + i

            train_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=train_seed)
            test_white_signal = nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=test_seed)

            for mode in modes:
                print(f"Syn={syn} | Experiment {i+1}/{n_experiments} | mode={mode} | train_seed={train_seed} test_seed={test_seed}")
                res = train_and_evaluate_decoders(train_input=train_white_signal, test_input=test_white_signal, delay_mode=mode, readout_syn=syn)
                losses[mode].append(res["loss"])
                rows.append({"syn": syn, "mode": mode, "experiment": i + 1, "loss": res["loss"]})

        try:
            x = np.arange(1, n_experiments + 1)
            plt.figure(figsize=(10, 5))
            for mode in modes:
                plt.plot(x, losses[mode], marker='o', label=f"{mode} (mean={np.mean(losses[mode]):.4e})")

            plt.xlabel("Experiment #")
            plt.ylabel("Loss")
            plt.title(f"Decoding loss across experiments (syn={syn}, high={input_high})")
            plt.legend()
            plt.grid(True)

            filename = f"loss_comparison_syn={syn}_hf.pdf"
            save_path = os.path.join(os.path.join(current_dir, "../figures"), filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved loss comparison plot to {save_path}")
        except Exception as e:
            print(f"Failed to generate or save loss comparison plot for syn={syn}: {e}")

        for mode in modes:
            vals = np.array(losses[mode])
            print(f"syn={syn} {mode} mean loss: {np.mean(vals):.6e}, std: {np.std(vals):.6e}")

        try:
            plot_loss_vs_syn(rows, modes=modes)
        except Exception as e:
            print(f"Failed to generate loss vs syn plot: {e}")

if __name__ == "__main__":
    run_experiments()

