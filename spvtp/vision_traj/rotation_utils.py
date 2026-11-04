"""Compatibility entry point; implementation lives in the structured package."""
from importlib import import_module
import sys
_module = import_module('spvtp.vision_traj.utils.rotation_utils')
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
