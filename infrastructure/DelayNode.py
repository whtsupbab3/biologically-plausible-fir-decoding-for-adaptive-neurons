import numpy as np

class DelayNode:
    def __init__(self, N, delay, dt=0.001):
        self.buf_len = int(delay / dt)
        self.buffer = np.zeros((self.buf_len, N))
        self.ptr = 0

    def update(self, t, x):
        v = self.buffer[self.ptr].copy()
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.buf_len
        return v