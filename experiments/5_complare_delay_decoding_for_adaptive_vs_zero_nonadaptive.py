import os
import sys
import numpy as np
import nengo
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from infrastructure.utils import (
	calculate_coeffs,
	calculate_loss_function,
	add_noise_to_activity,
)
from infrastructure.DelayNetwork import DelayNetwork

n_neurons = 300
readout_synapse = 0.05
run_time = 10.0
dt = 0.001
input_high = 10
n_experiments = 10
base_train_seed = 121
base_test_seed = 223
rms = 0.25

np.random.seed(42)


def train_and_evaluate(train_input, test_input, neuron_type, delay_mode, label):
	with nengo.Network(seed=10) as model:
		input_node = nengo.Node(train_input, size_out=1)
		delay_network = DelayNetwork(
			num_neurons=n_neurons,
			readout_synapse=readout_synapse,
			neuron_type=neuron_type,
			delay_mode=delay_mode,
		)
		nengo.Connection(input_node, delay_network.ens, synapse=None)

		p_input = nengo.Probe(input_node, sample_every=dt)
		p_activity = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

	with nengo.Simulator(model) as sim:
		sim.run(run_time)

	t = sim.trange()
	activity = add_noise_to_activity(sim.data[p_activity])
	coeffs = calculate_coeffs(activity, sim.data[p_input])
	
	with nengo.Network(seed=10) as model:
		input_node = nengo.Node(test_input, size_out=1)
		delay_network = DelayNetwork(
			num_neurons=n_neurons,
			decoder_weights=coeffs,
			readout_synapse=readout_synapse,
			neuron_type=neuron_type,
			delay_mode=delay_mode,
		)
		nengo.Connection(input_node, delay_network.ens, synapse=None)

		p_input = nengo.Probe(input_node, sample_every=dt)
		p_decoded = nengo.Probe(delay_network.readout, synapse=None, sample_every=dt)

	with nengo.Simulator(model) as sim:
		sim.run(run_time)

	decoded = sim.data[p_decoded]
	loss = calculate_loss_function(sim.data[p_input], decoded)

	return {
		"t": t,
		"input": sim.data[p_input],
		"coeffs": coeffs,
		"decoded": decoded,
		"loss": loss,
		"label": label,
		"delay_mode": delay_mode,
	}


def run_experiments():
	fig_dir = os.path.join(current_dir, "../figures/5")
	os.makedirs(fig_dir, exist_ok=True)

	scenarios = [
		(
			nengo.AdaptiveLIF(
				tau_rc=0.05,
				tau_ref=0.002,
				tau_n=0.2,   
				inc_n=0.1, 
			),
			"range",
			"AdaptiveLIF (range delay)",
		),
		(
			nengo.LIF(tau_rc=0.05, tau_ref=0.002),
			"zero",
			"LIF (instantaneous)",
		),
	]

	losses = {label: [] for (_, _, label) in scenarios}

	print(
		f"Running {n_experiments} experiments comparing AdaptiveLIF range-delay vs LIF zero-delay (run_time={run_time}s)"
	)

	example_traces = {}  

	for i in range(n_experiments):
		train_seed = base_train_seed + i
		test_seed = base_test_seed + i

		train_ws = nengo.processes.WhiteSignal(
			period=run_time, high=input_high, rms=rms, seed=train_seed
		)
		test_ws = nengo.processes.WhiteSignal(
			period=run_time, high=input_high, rms=rms, seed=test_seed
		)

		print(f"Experiment {i + 1}/{n_experiments} - train_seed={train_seed}, test_seed={test_seed}")

		for neuron_type, delay_mode, label in scenarios:
			res = train_and_evaluate(
				train_input=train_ws,
				test_input=test_ws,
				neuron_type=neuron_type,
				delay_mode=delay_mode,
				label=label,
			)
			losses[label].append(res["loss"])

			if i == 0:
				example_traces[label] = res

	for label, res in example_traces.items():
		try:
			plt.figure(figsize=(12, 4))
			plt.plot(res["t"], res["input"], linewidth=1, linestyle="--", label="input")
			plt.plot(res["t"], res["decoded"], linewidth=1, alpha=0.7, label="decoded")
			plt.xlabel("Time (s)")
			plt.ylabel("Signal value")
			plt.title(f"{label} (loss={res['loss']:.6f})")
			plt.legend()
			filename = (
				f"{label.replace(' ', '_').replace('/', '_')}_syn={readout_synapse}_high={input_high}_rms={rms}.pdf"
			)
			save_path = os.path.join(fig_dir, filename)
			plt.savefig(save_path, bbox_inches="tight", dpi=300)
			print(f"Saved example trace to {save_path}")
		except Exception as e:
			print(f"Failed to save example trace for {label}: {e}")

	try:
		x = np.arange(1, n_experiments + 1)
		plt.figure(figsize=(10, 5))
		for (_, _, label) in scenarios:
			series = losses[label]
			plt.plot(x, series, marker="o", label=f"{label} (mean={np.mean(series):.4e})")

		plt.xlabel("Experiment #")
		plt.ylabel("Loss")
		plt.title(
			f"Loss across experiments: AdaptiveLIF(range) vs LIF(zero) (syn={readout_synapse}, high={input_high}, rms={rms})"
		)
		plt.legend()
		plt.grid(True)

		filename = (
			f"loss_adaptive_range_vs_nonadaptive_zero_syn={readout_synapse}_high={input_high}_rms={rms}.pdf"
		)
		save_path = os.path.join(fig_dir, filename)
		plt.savefig(save_path, bbox_inches="tight", dpi=300)
		print(f"Saved loss comparison plot to {save_path}")
	except Exception as e:
		print(f"Failed to generate loss comparison plot: {e}")


if __name__ == "__main__":
	run_experiments()