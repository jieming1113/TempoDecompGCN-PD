
# train.py - Subject-level 5-fold cross-validation


import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score)

from model import TempoDecompGCN, NUM_PHASES, ADJ_INIT_TEMPERATURE
from dataset import TurningDataset, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training hyperparameters
RANDOM_SEED  = 42
BATCH_SIZE   = 16
EPOCHS       = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 0.01
DROPOUT       = 0.5
LABEL_SMOOTHING = 0.1

ADJ_FINAL_TEMPERATURE  = 0.5
ADJ_ANNEAL_EPOCHS      = 100
ADJ_SPARSITY_WEIGHT    = 0.01
BOUNDARY_SPARSITY_WEIGHT   = 0.05
BOUNDARY_SMOOTHNESS_WEIGHT = 0.1

JOINT_NAMES = [
    'Neck/Spine', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Thorax', 'Pelvis', 'Neck_Base', 'Head',
    'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist',
]



# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha          = alpha
        self.gamma          = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss   = F.cross_entropy(inputs, targets, weight=self.alpha,
                                    reduction='none', label_smoothing=self.label_smoothing)
        pt        = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# Training
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    progress    = min(epoch / ADJ_ANNEAL_EPOCHS, 1.0)
    temperature = ADJ_INIT_TEMPERATURE - progress * (ADJ_INIT_TEMPERATURE - ADJ_FINAL_TEMPERATURE)
    model.set_adj_temperature(temperature)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs, losses = model(x, return_losses=True)
        cls_loss = criterion(outputs, y)
        reg_loss = (BOUNDARY_SPARSITY_WEIGHT   * losses['boundary_sparsity'] +
                    BOUNDARY_SMOOTHNESS_WEIGHT  * losses['boundary_smoothness'] +
                    ADJ_SPARSITY_WEIGHT         * losses['adj_sparsity'])
        loss = cls_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += y.size(0)
        correct += predicted.eq(y).sum().item()

    return {'loss': total_loss / len(train_loader), 'accuracy': correct / total}


# Evaluation
def comprehensive_evaluation(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            x      = x.to(device)
            outputs = model(x)
            probs  = F.softmax(outputs, dim=1)
            preds  = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy':    accuracy_score(all_labels, all_preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score':    f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc':     roc_auc_score(all_labels, all_probs)
                       if len(np.unique(all_labels)) > 1 else 0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'predictions':   all_preds,
        'labels':        all_labels,
        'probabilities': all_probs,
    }


def print_fold_metrics(metrics, fold_num, dataset_name='Test'):
    print(f"\n  {dataset_name} Set Metrics (Fold {fold_num}):")
    print(f"  {'-'*50}")
    for name, key in [('Accuracy',    'accuracy'),
                      ('Sensitivity', 'sensitivity'),
                      ('Specificity', 'specificity'),
                      ('F1 Score',    'f1_score'),
                      ('AUC-ROC',     'auc_roc')]:
        print(f"  {name:<25} {metrics[key]:>10.4f}")
    print(f"  {'-'*50}")
    print(f"  Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, "
          f"FP={metrics['fp']}, FN={metrics['fn']}")


def print_summary_metrics(all_fold_metrics, dataset_name='Test'):
    metric_keys = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc_roc']
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name} Metrics Across All Folds")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*50}")
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        print(f"  {key:<25} {np.mean(values):>10.4f} {np.std(values):>10.4f}")


# Subject-Level 5-Fold Cross-Validation
def main():
    print("=" * 70)
    print("TempoDecompGCN: Weakly-Supervised Temporal Decomposition GCN")
    print("Subject-Level 5-Fold Cross-Validation")
    print("=" * 70)

    data, labels, subjects = load_dataset('.')
    print(f"Loaded {len(labels)} samples  |  HC={np.sum(labels==0)}  PD={np.sum(labels==1)}")

    unique_subjects   = np.unique(subjects)
    subject_to_label  = {s: labels[subjects == s][0] for s in unique_subjects}
    subject_label_arr = np.array([subject_to_label[s] for s in unique_subjects])
    print(f"Subjects: {len(unique_subjects)} total  "
          f"(HC={int(np.sum(subject_label_arr==0))}, PD={int(np.sum(subject_label_arr==1))})")

    outer_skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_test_metrics = []

    for fold, (train_subj_idx, test_subj_idx) in enumerate(
            outer_skf.split(unique_subjects, subject_label_arr)):

        print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")
        train_subjects = unique_subjects[train_subj_idx]
        test_subjects  = unique_subjects[test_subj_idx]
        print(f"  Test  subjects ({len(test_subjects)}): {list(test_subjects)}")
        print(f"  Train subjects ({len(train_subjects)}): {list(train_subjects)}")

        train_mask = np.isin(subjects, train_subjects)
        test_mask  = np.isin(subjects, test_subjects)
        X_train, y_train = data[train_mask], labels[train_mask]
        X_test,  y_test  = data[test_mask],  labels[test_mask]
        print(f"  Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")

        train_loader = DataLoader(TurningDataset(X_train, y_train, training=True),
                                  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(TurningDataset(X_test,  y_test,  training=False),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        class_counts  = np.bincount(y_train)
        class_weights = torch.tensor(len(y_train) / (2 * class_counts),
                                     dtype=torch.float32).to(device)

        model = TempoDecompGCN(in_channels=6, hidden_dim=64, num_classes=2,
                       num_phases=NUM_PHASES, dropout=DROPOUT).to(device)

        if fold == 0:
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        for epoch in range(EPOCHS):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            scheduler.step()
            if (epoch + 1) % 25 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}: Train Acc={train_metrics['accuracy']:.4f}")

        test_metrics = comprehensive_evaluation(model, test_loader, device)
        fold_test_metrics.append(test_metrics)
        print_fold_metrics(test_metrics, fold + 1, 'Outer Test')

    # Summary across folds
    print_summary_metrics(fold_test_metrics, 'Outer Test (Subject-Level 5-Fold)')

    # Combined aggregate (each sample predicted exactly once)
    print("\n" + "=" * 70)
    print("COMBINED AGGREGATE RESULTS")
    print("=" * 70)
    all_labels_comb = np.concatenate([m['labels']        for m in fold_test_metrics])
    all_preds_comb  = np.concatenate([m['predictions']   for m in fold_test_metrics])
    all_probs_comb  = np.concatenate([m['probabilities'] for m in fold_test_metrics])
    cm = confusion_matrix(all_labels_comb, all_preds_comb)
    tn, fp, fn, tp = cm.ravel()
    combined = {
        'Accuracy':    accuracy_score(all_labels_comb, all_preds_comb),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1':          f1_score(all_labels_comb, all_preds_comb, zero_division=0),
        'AUC-ROC':     roc_auc_score(all_labels_comb, all_probs_comb)
                       if len(np.unique(all_labels_comb)) > 1 else 0,
    }
    for k, v in combined.items():
        print(f"  {k:<15} {v:.4f}")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")


if __name__ == '__main__':
    main()
