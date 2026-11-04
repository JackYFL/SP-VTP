"""Vision-conditioned trajectory prediction module.

See ``docs/README.md`` and ``docs/model_arch.md`` for design notes.
"""

from importlib import import_module

__all__ = [
    "VisionTrajDataset",
    "build_episode_index",
    "FlowMatchingHead",
    "TaskEncoder",
    "ObsEncoder",
    "VisionTrajPolicy",
    "VisionTrajPolicyConfig",
]

_MODULES = {
    'VisionTrajDataset': 'data.dataset', 'build_episode_index': 'data.dataset',
    'FlowMatchingHead': 'models.flow_matching_head', 'TaskEncoder': 'models.task_encoder',
    'ObsEncoder': 'models.obs_encoder', 'VisionTrajPolicy': 'models.policy',
    'VisionTrajPolicyConfig': 'models.policy',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{_MODULES[name]}'), name)
    globals()[name] = value
    return value
