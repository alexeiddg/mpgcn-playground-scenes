import numpy as np
import random
import cv2


def _joint_jitter(x, sigma=0.01):
    noise = np.random.randn(*x.shape) * sigma
    x[..., :2] += noise[..., :2]
    return x


class SkeletonAugmenter:
    def __init__(self,
                 enable=True,
                 temporal_jitter=True,
                 random_crop=True,
                 random_drop=True,
                 speed_scale=True,
                 joint_jitter=True,
                 joint_translate=True,
                 joint_scale=True,
                 horizontal_flip=False,
                 object_jitter=True):

        self.enable = enable

        # Temporal
        self.temporal_jitter = temporal_jitter
        self.random_crop = random_crop
        self.random_drop = random_drop
        self.speed_scale = speed_scale

        # Joint space
        self.joint_jitter = joint_jitter
        self.joint_translate = joint_translate
        self.joint_scale = joint_scale
        self.horizontal_flip = horizontal_flip

        # Objects
        self.object_jitter = object_jitter

    def _temporal_jitter(self, x, obj):
        shift = np.random.randint(-2, 3)
        if shift > 0:
            x = np.concatenate([x[shift:], np.repeat(x[-1:], shift, axis=0)], axis=0)
            obj = np.concatenate([obj[shift:], np.repeat(obj[-1:], shift, axis=0)], axis=0)
        elif shift < 0:
            shift = -shift
            x = np.concatenate([np.repeat(x[:1], shift, axis=0), x[:-shift]], axis=0)
            obj = np.concatenate([np.repeat(obj[:1], shift, axis=0), obj[:-shift]], axis=0)
        return x, obj

    def _random_crop(self, x, obj, T):
        start = np.random.randint(0, x.shape[0] - T)
        return x[start:start+T], obj[start:start+T]

    def _random_drop(self, x, obj, prob=0.05):
        mask = np.random.rand(x.shape[0]) > prob
        if mask.sum() == 0:
            return x, obj
        x = x[mask]
        obj = obj[mask]
        return x, obj

    def _speed_scale(self, x, obj):
        scale = np.random.uniform(0.8, 1.2)
        T = x.shape[0]
        new_T = int(T * scale)
        idx = np.linspace(0, T - 1, new_T)
        x2 = np.stack([x[int(i)] for i in idx])
        obj2 = np.stack([obj[int(i)] for i in idx])
        return x2, obj2

    def _joint_translate(self, x, max_shift=0.05):
        shift = (np.random.rand(1, 1, 1, 2) - 0.5) * max_shift
        x[..., :2] += shift
        return x

    def _joint_scale(self, x, scale_range=(0.95, 1.05)):
        s = np.random.uniform(*scale_range)
        center = x[..., :2].mean(axis=(0,1), keepdims=True)
        x[..., :2] = (x[..., :2] - center) * s + center
        return x

    def _horizontal_flip(self, x):
        x[..., 0] = 1.0 - x[..., 0]
        return x

    def _object_jitter(self, obj, sigma=0.02):
        noise = np.random.randn(*obj.shape) * sigma
        obj[..., :2] += noise[..., :2]
        return obj

    def __call__(self, pose, obj, T_target=48):
        if not self.enable:
            return pose, obj

        # Temporal
        if self.temporal_jitter:
            pose, obj = self._temporal_jitter(pose, obj)

        if self.random_crop and pose.shape[0] > T_target:
            pose, obj = self._random_crop(pose, obj, T_target)

        if self.random_drop:
            pose, obj = self._random_drop(pose, obj)

        if self.speed_scale and np.random.rand() < 0.3:
            pose, obj = self._speed_scale(pose, obj)

        pose = pose[:T_target]
        obj = obj[:T_target]

        # Joint space
        if self.joint_jitter:
            pose = _joint_jitter(pose)

        if self.joint_translate:
            pose = self._joint_translate(pose)

        if self.joint_scale:
            pose = self._joint_scale(pose)

        if self.horizontal_flip and np.random.rand() < 0.5:
            pose = self._horizontal_flip(pose)

        # Object jitter
        if self.object_jitter:
            obj = self._object_jitter(obj)

        return pose, obj