"""
dataset.py - Data loading and preprocessing for PD turning assessment

Expects skeleton sequences in JSON format from the REMAP study.
Each sequence is normalized, temporally resampled, and augmented during training.
"""

import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from model import NUM_KEYPOINTS, TARGET_LENGTH


# Preprocessing
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    num_frames = len(data['data'])
    keypoints  = np.zeros((num_frames, NUM_KEYPOINTS, 2), dtype=np.float32)
    for i, frame_data in enumerate(data['data']):
        skeleton = frame_data['skeleton'][0]
        pose     = np.array(skeleton['pose'])
        for j in range(NUM_KEYPOINTS):
            keypoints[i, j, 0] = pose[j * 2]
            keypoints[i, j, 1] = pose[j * 2 + 1]
    return keypoints


def normalize_skeleton(keypoints):
    hip_center    = (keypoints[:, 1, :] + keypoints[:, 4, :]) / 2
    centered      = keypoints - hip_center[:, np.newaxis, :]
    shoulder_center = (centered[:, 11, :] + centered[:, 14, :]) / 2
    torso_length  = np.mean(np.linalg.norm(shoulder_center, axis=1))
    if torso_length > 0:
        centered = centered / (torso_length + 1e-6)
    return centered


def interpolate_sequence(keypoints, target_length):
    T, K, C      = keypoints.shape
    indices      = np.linspace(0, T - 1, target_length)
    interpolated = np.zeros((target_length, K, C), dtype=np.float32)
    for k in range(K):
        for c in range(C):
            interpolated[:, k, c] = np.interp(indices, np.arange(T), keypoints[:, k, c])
    return interpolated


def compute_enhanced_features(keypoints):
    """Concatenate position, velocity, and acceleration → 6-channel input."""
    velocity     = np.zeros_like(keypoints)
    velocity[1:] = keypoints[1:] - keypoints[:-1]
    velocity[0]  = velocity[1]
    acceleration     = np.zeros_like(keypoints)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0]  = acceleration[1]
    return np.concatenate([keypoints, velocity, acceleration], axis=2).astype(np.float32)


def augment_skeleton(keypoints):
    """Random scale, rotation, noise, and horizontal flip with joint swapping."""
    aug = keypoints.copy()
    if random.random() < 0.5:
        aug = aug * random.uniform(0.9, 1.1)
    if random.random() < 0.5:
        angle    = random.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot      = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        for t in range(len(aug)):
            for k in range(aug.shape[1]):
                aug[t, k] = rot @ aug[t, k]
    if random.random() < 0.5:
        aug = aug + np.random.normal(0, 0.02, aug.shape).astype(np.float32)
    if random.random() < 0.5:
        aug[:, :, 0] = -aug[:, :, 0]
        left  = [4, 5, 6, 11, 12, 13]
        right = [1, 2, 3, 14, 15, 16]
        tmp               = aug[:, left,  :].copy()
        aug[:, left,  :]  = aug[:, right, :]
        aug[:, right, :]  = tmp
    return aug


# Dataset
class TurningDataset(Dataset):
    def __init__(self, data, labels, training=True):
        self.data     = data
        self.labels   = labels
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        keypoints = self.data[idx].copy()
        if self.training:
            keypoints = augment_skeleton(keypoints)
        features = compute_enhanced_features(keypoints)
        x = np.transpose(features, (2, 0, 1))
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


# Data loading
def load_dataset(data_dir):
    """
    Load the REMAP turning dataset.

    Directory layout expected:
        <data_dir>/../REMAP/2-labelled_Turning/2_group/
            0_json_remove1/   (healthy controls)
            1_json_remove1/   (PD patients)

    Filename convention:  <SubjectID>_<...>.json
        e.g. "Pt204_C_n_350.json"  → subject_id = "Pt204"

    Returns
    -------
    data     : np.ndarray  (N, T, K, C)
    labels   : np.ndarray  (N,)
    subjects : np.ndarray  (N,)  string subject IDs
    """
    all_data, all_labels, all_subjects = [], [], []
    class_0_dir = os.path.join(data_dir, '../REMAP/2-labelled_Turning/2_group/0_json_remove1')
    class_1_dir = os.path.join(data_dir, '../REMAP/2-labelled_Turning/2_group/1_json_remove1')

    for label, label_dir in [(0, class_0_dir), (1, class_1_dir)]:
        if not os.path.exists(label_dir):
            continue
        for filename in sorted(os.listdir(label_dir)):
            if not filename.endswith('.json'):
                continue
            subject_id = filename.split('_')[0]
            filepath   = os.path.join(label_dir, filename)
            try:
                keypoints = load_json_file(filepath)
                if len(keypoints) < 10:
                    continue
                keypoints = normalize_skeleton(keypoints)
                keypoints = interpolate_sequence(keypoints, TARGET_LENGTH)
                all_data.append(keypoints)
                all_labels.append(label)
                all_subjects.append(subject_id)
            except Exception:
                continue

    return np.array(all_data), np.array(all_labels), np.array(all_subjects)
