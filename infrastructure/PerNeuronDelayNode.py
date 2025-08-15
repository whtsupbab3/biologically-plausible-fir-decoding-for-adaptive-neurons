import bisect
import numpy as np
from typing import Sequence

class SpikeTrainBuffer():
    def __init__(self, sw=[]):
        self.spikes = []

    def insert(self, time, weight):
        bisect.insort_right(self.spikes, (time, weight))

    def spike_now(self, t):
        output = 0.
        while (self.spikes!=[] and self.spikes[0][0]<=t):
            output += self.spikes[0][1]
            self.spikes.pop(0)
        return output
    
class PerNeuronDelayNode:
    def __init__(self, n_neurons: int, delays: Sequence[float], dt: float):
        assert len(delays) == n_neurons, "delays must match n_neurons"
        self.dt = dt
        self.delays = np.asarray(delays)
        self.buffers = [SpikeTrainBuffer() for _ in range(n_neurons)]

    def __call__(self, t, x):
        for idx, value in enumerate(x):
            if value != 0.0:
                self.buffers[idx].insert(t + self.delays[idx] - self.dt, value)

        return np.array([buf.spike_now(t) for buf in self.buffers])
