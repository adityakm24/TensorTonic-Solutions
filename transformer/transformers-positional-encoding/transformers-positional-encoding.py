import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model))
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(position / div_term)   # even indices
    pe[:, 1::2] = np.cos(position / div_term)   # odd  indices

    return pe