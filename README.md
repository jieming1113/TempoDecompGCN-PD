# TempoDecompGCN-PD

## Overview

We propose a weakly-supervised temporal decomposition framework for skeleton-based Parkinsonian turning assessment. The framework learns to partition turning sequences into temporal instances from video-level diagnostic labels alone, without any frame-level phase annotations.


## Requirements

```
python >= 3.8
torch >= 1.12
numpy
scikit-learn
```

Install dependencies:

```bash
pip install torch numpy scikit-learn
```

## Data Preparation

This code uses the **REMAP dataset** [(Morgan et al., Scientific Data 2023)](https://doi.org/10.1038/s41597-023-02663-5).

Expected directory structure:

```
REMAP/
└── 2-Turning/
    └── 2_group/
        ├── 0_json_remove1/    # Healthy controls (label 0)
        │   ├── Pt001_xxx.json
        │   └── ...
        └── 1_json_remove1/    # PD patients (label 1)
            ├── Pt204_xxx.json
            └── ...
```

Each JSON file contains a skeleton sequence with 17 keypoints (HRNet 2D pose format). The filename prefix (e.g., `Pt204`) is used as the subject ID for subject-level cross-validation.

Place the `REMAP/` folder one level above the code directory, or update the path in `dataset.py`:

```python
class_0_dir = os.path.join(data_dir, '../REMAP/2-Turning/2_group/0_json_remove1')
class_1_dir = os.path.join(data_dir, '../REMAP/2-Turning/2_group/1_json_remove1')
```

## Usage

```bash
python train.py
```

Training runs subject-level 5-fold cross-validation and prints per-fold and summary metrics (Accuracy, Sensitivity, Specificity, F1, AUC-ROC).



## File Structure

```
├── graph.py       # Skeleton graph definition (17-joint layout)
├── model.py       # Model architecture (TempoDecompGCN)
├── dataset.py     # Data loading, preprocessing, augmentation
└── train.py       # Training and evaluation
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{zhang2026weakly,
  title     = {Weakly-Supervised Temporal Decomposition with Graph Convolutional Networks for Parkinsonian Turning Assessment},
  author    = {Jieming Zhang, Tai-Myoung Chung and Hogun Park},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

The skeleton graph implementation is adapted from [ST-GCN](https://github.com/yysijie/st-gcn). We thank the REMAP study team for providing the clinical dataset.
