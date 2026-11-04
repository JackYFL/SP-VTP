"""Compatibility entry point; implementation lives in the structured package."""
from importlib import import_module
import sys
_module = import_module('spvtp.vision_traj.baselines.smoke_task_encoder_pi05_obs')
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
