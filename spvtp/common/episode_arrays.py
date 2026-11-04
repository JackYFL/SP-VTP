"""Align processed pose/gripper arrays with raw-video validity masks."""
import numpy as np


def align_valid_frames(values, valid):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if len(values) == len(valid):
        return values
    if len(values) != int(valid.sum()):
        raise ValueError(f'Array length {len(values)} matches neither raw nor valid frame count')
    aligned = np.full((len(valid), *values.shape[1:]), np.nan, dtype=np.result_type(values.dtype, np.float32))
    aligned[valid] = values
    return aligned
