import numpy as np

from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TopDownGetRandomRotation90:
    """Random rotations with +90/-90 degrees"""

    def __init__(self, rot_prob=0.5):
        self.rot_prob = rot_prob

    def __call__(self, results):
        if np.random.rand() <= self.rot_prob:
            rotation_angle = 90 if np.random.rand() <= 0.5 else -90
            results["rotation"] += rotation_angle
        return results
