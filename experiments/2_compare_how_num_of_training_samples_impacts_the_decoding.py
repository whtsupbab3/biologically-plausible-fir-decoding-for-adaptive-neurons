import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import nengo

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from infrastructure.utils import calculate_coeffs, calculate_loss_function
from infrastructure.DelayNetwork import DelayNetwork

np.random.seed(42)

# Constants
n_neurons = 300
readout_synapse = 0.05
neuron_type = nengo.AdaptiveLIF(tau_n=0.5, inc_n=0.01)
run_time = 10.0
dt = 0.001
default_delay_mode = "zero"
input_high = 10
n_experiments = 1
slice_counts = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
base_train_seed = 3244
base_test_seed = 2000

result_dir = os.path.abspath(os.path.join(current_dir, "..", "figures"))
os.makedirs(result_dir, exist_ok=True)

def run_single_training_run(train_seed):
	with nengo.Network(seed=10) as model:
		input_node = nengo.Node(nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=train_seed), size_out=1)
		delay_network = DelayNetwork(num_neurons=n_neurons, readout_synapse=readout_synapse, neuron_type=neuron_type, delay_mode=default_delay_mode)
		nengo.Connection(input_node, delay_network.ens, synapse=None)

		p_input = nengo.Probe(input_node, sample_every=dt)
		p_delay_activity = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

	with nengo.Simulator(model) as sim:
		sim.run(run_time)

	return sim.data[p_delay_activity], sim.data[p_input], sim.trange()

def train_decoder_from_stacked_runs(activity_list, input_list):
	X = np.vstack(activity_list)
	y = np.vstack(input_list)
	coeffs = calculate_coeffs(X, y)
	return np.asarray(coeffs)

def evaluate_decoder_on_test(coeffs, test_seed):
	with nengo.Network(seed=10) as model:
		input_node = nengo.Node(nengo.processes.WhiteSignal(period=run_time, high=input_high, rms=0.25, seed=test_seed), size_out=1)
		delay_network = DelayNetwork(num_neurons=n_neurons, decoder_weights=coeffs, readout_synapse=readout_synapse, neuron_type=neuron_type, delay_mode="discrete")
		nengo.Connection(input_node, delay_network.ens, synapse=None)

		p_input = nengo.Probe(input_node, sample_every=dt)
		p_delay_decoded = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

	with nengo.Simulator(model) as sim:
		sim.run(run_time)

	decoded = sim.data[p_delay_decoded]
	loss = calculate_loss_function(sim.data[p_input], decoded)
	return loss

def run_sweep():
	results = {N: [] for N in slice_counts}

	print(f"Evaluating decoder sizes: {slice_counts}")

	for exp_idx in range(n_experiments):
		print(f"Experiment repeat {exp_idx+1}/{n_experiments}")
		test_seed = base_test_seed + exp_idx

		records = []
		for j in range(slice_counts[-1]):
			train_seed = base_train_seed + exp_idx * 100 + j
			activity, inputs, _t = run_single_training_run(train_seed)
			records.append({"activity": activity, "input": inputs})

		for N in slice_counts:
			activity_list = [rec['activity'] for rec in records[:N]]
			input_list = [rec['input'] for rec in records[:N]]

			coeffs = train_decoder_from_stacked_runs(activity_list, input_list)
			loss = evaluate_decoder_on_test(coeffs, test_seed)
			results[N].append(loss)
			
	Ns = []
	means = []
	stds = []
	out_lines = ["N,loss_mean,loss_std,all_losses"]
	for N in slice_counts:
		arr = np.array(results[N])
		Ns.append(N)
		means.append(arr.mean())
		stds.append(arr.std())
		out_lines.append(f"{N},{arr.mean():.6e},{arr.std():.6e},\"{arr.tolist()}\"")
		
	plt.figure(figsize=(10, 5))
	plt.plot(Ns, means, marker='o')
	plt.fill_between(Ns, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
	plt.xlabel('Number of independent training samples (N)')
	plt.ylabel('Test MSE loss')
	plt.title('Effect of number of training samples on decoding accuracy')
	plt.grid(True)
	plot_path_pdf = os.path.join(result_dir, f"training_samples_sweep.pdf")
	try:
		plt.savefig(plot_path_pdf, bbox_inches='tight', dpi=300)
		print(f"Saved plot to {plot_path_pdf}.")
	except Exception as e:
		print(f"Failed to save outputs: {e}")

	for N, m, s in zip(Ns, means, stds):
		print(f"N={N:2d} mean_loss={m:.6e} std={s:.6e}")

	return results

if __name__ == '__main__':
	run_sweep()