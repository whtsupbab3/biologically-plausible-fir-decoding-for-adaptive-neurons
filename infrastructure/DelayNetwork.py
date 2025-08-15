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
        delay_range: Optional[tuple[float, float]] = (0.002, 0.02),
        discrete_delays_set: Optional[list[float]] = [0.002, 0.01, 0.02, 0.03]
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
        if self.delay_mode is "zero":
            delays = np.zeros(self.num_neurons, dtype=int) 
        elif self.delay_mode is "range":
            delays = np.random.uniform(low=self.delay_range[0], high=self.delay_range[1], size=self.num_neurons)
        else:
            delays = random.choices(self.discrete_delays_set, k=self.num_neurons)
       
        return nengo.Node(
                PerNeuronDelayNode(self.num_neurons, delays, self.dt), 
                size_in=self.num_neurons,
                size_out=self.num_neurons,
                label="per_neuron_delay",
            )
