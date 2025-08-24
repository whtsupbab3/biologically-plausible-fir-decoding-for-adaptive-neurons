import nengo
from typing import Optional, Union, Literal
import random
import numpy as np
from .PerNeuronDelayNode import PerNeuronDelayNode

class DelayNetwork(nengo.Network):
    def __init__(
        self,
        label: str = 'Delay Network',
        dt: float = 0.001,
        neuron_type: Optional[nengo.neurons.NeuronType] = nengo.AdaptiveLIF(tau_n=0.01),
        rms: float = 0.25,
        readout_synapse: float = 0.003,
        seed: int = 45,
        randomness_seed: int = 44,
        ensemble_seed: int = 452,
        num_neurons: int = 300,
        dimensions: int = 1,
        decoder_weights: Union[np.ndarray, list[float]] = [],
        delay_mode: Literal["zero", "range", "discrete"] = "range",
        delay_range: Optional[tuple[float, float]] = (0.002, 0.025),
        discrete_delays_set: Optional[list[float]] = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,  0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025]
    ):
        super().__init__(label=label, seed=seed)
        self.dt             = dt
        self.neuron_type    = neuron_type or nengo.AdaptiveLIF(tau_n=0.01)
        self.randomness_seed = randomness_seed
        self.rms            = rms
        self.readout_synapse = readout_synapse
        self.ensemble_seed = ensemble_seed
        self.num_neurons = num_neurons
        self.delay_mode = delay_mode
        self.delay_range = delay_range
        self.discrete_delays_set = discrete_delays_set
        self.dimensions = dimensions
        self.decoder_weights = decoder_weights
        np.random.seed(self.randomness_seed)

        with self:
            self.ens = nengo.Ensemble(
                self.num_neurons,
                dimensions=self.dimensions,
                neuron_type=self.neuron_type,
                max_rates=nengo.dists.Uniform(30, 60),
                intercepts=nengo.dists.Uniform(-0.7, 0.7),
                seed=self.ensemble_seed
            )

            delay_node = self._build_delay_node()
            nengo.Connection(self.ens.neurons, delay_node, synapse=None, seed=1)

            if len(decoder_weights) > 0:
                self.readout = nengo.Node(size_in=1)
                nengo.Connection(delay_node, self.readout, synapse=self.readout_synapse, transform=self.decoder_weights.T, seed=1)
            else:
                self.readout = nengo.Node(size_in=self.num_neurons)
                nengo.Connection(delay_node, self.readout, synapse=self.readout_synapse, seed=1)
                    
            self.nengo_readout = nengo.Node(size_in=1)
            nengo.Connection(self.ens, self.nengo_readout, synapse=self.readout_synapse)

    def _build_delay_node(self):
        rng = np.random.default_rng(self.randomness_seed)
        
        if self.delay_mode == "zero":
            delay_steps = np.zeros(self.num_neurons, dtype=int)
        elif self.delay_mode == "range":
            lo, hi = 0.002, 0.025
            delay_steps = np.rint(
                rng.uniform(lo, hi, size=self.num_neurons) / self.dt
            ).astype(int)
        elif self.delay_mode == "discrete":
            chosen = rng.choice(self.discrete_delays_set, size=self.num_neurons, replace=True)
            delay_steps = np.rint(np.array(chosen) / self.dt).astype(int)
        else:
            raise ValueError(f"Unknown delay_mode {self.delay_mode}")

        delays = delay_steps * self.dt

        return nengo.Node(
                PerNeuronDelayNode(self.num_neurons, delays, self.dt), 
                size_in=self.num_neurons,
                size_out=self.num_neurons,
                label="per_neuron_delay",
            )
