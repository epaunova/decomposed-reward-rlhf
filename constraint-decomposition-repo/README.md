# Constraint Decomposition for Multi-Objective RLHF

[![Paper](https://img.shields.io/badge/Technical%20Report-PDF-red)](docs/NVIDIA_Technical_Report.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Constraint Decomposition for Multi-Objective RLHF in Large Language Models"** (NVIDIA Technical Report, December 2025).

<p align="center">
  <img src="docs/method_diagram.png" width="800">
</p>

## 🎯 Key Results

| Method | IFEval (%) | GSM8K | HumanEval | MT-Bench |
|--------|------------|-------|-----------|----------|
| Standard RLHF | 41.2 | Baseline | Baseline | Baseline |
| LLaMA-3 RLHF | 52.5 | +8.2 | +7.5 | +0.8 |
| PAMA | 65.4 | +12.1 | +9.3 | +1.1 |
| **Ours** | **73.8** | **+15.6** | **+11.1** | **+1.4** |

**+32.6 percentage points** improvement over standard RLHF on IFEval benchmark.

## 💡 TL;DR

Standard RLHF uses a single reward model that struggles with multi-constraint instructions. We decompose rewards into four orthogonal components (semantic, structural, format, meta), train specialized reward models for each, and combine them hierarchically with conflict-aware adaptation.

**Ablation contributions:**
- Decomposition: 54%
- Hierarchical combination: 26%
- Conflict adaptation: 20%

## 🏗️ Architecture

```
Input Prompt
     │
     ▼
┌─────────────────────────────────────────┐
│         Constraint Decomposition         │
├──────────┬──────────┬─────────┬─────────┤
│R_semantic│R_struct  │R_format │R_meta   │
│  (84%)   │  (89%)   │  (92%)  │  (87%)  │
└────┬─────┴────┬─────┴────┬────┴────┬────┘
     │          │          │         │
     ▼          ▼          ▼         ▼
┌─────────────────────────────────────────┐
│      Hierarchical Combination           │
│  R = αR_sem + βR_str + γR_fmt + δR_meta │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     Conflict Detection & Adaptation      │
│         (89.3% detection accuracy)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
         Optimized Response
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/epaunova/constraint-decomposition.git
cd constraint-decomposition
pip install -e .
```

### Requirements

```
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
deepspeed>=0.12.0
wandb>=0.16.0
datasets>=2.14.0
trl>=0.7.0
```

### Training Decomposed Reward Models

```python
from constraint_decomposition import DecomposedRewardTrainer

# Initialize trainer for semantic reward model
trainer = DecomposedRewardTrainer(
    base_model="nvidia/nemotron-7b",
    reward_type="semantic",  # Options: semantic, structural, format, meta
    output_dir="./reward_models/semantic"
)

# Train on aspect-specific preferences
trainer.train(
    train_dataset=semantic_preferences,
    eval_dataset=semantic_eval,
    num_epochs=3,
    batch_size=32
)
```

### PPO Training with Decomposed Rewards

```python
from constraint_decomposition import ConstraintDecompositionPPO

# Load decomposed reward models
ppo_trainer = ConstraintDecompositionPPO(
    policy_model="nvidia/nemotron-7b",
    reward_models={
        "semantic": "./reward_models/semantic",
        "structural": "./reward_models/structural",
        "format": "./reward_models/format",
        "meta": "./reward_models/meta"
    },
    conflict_detector="./models/conflict_detector"
)

# Train with hierarchical reward combination
ppo_trainer.train(
    dataset=instruction_dataset,
    num_steps=10000,
    batch_size=512,
    learning_rate=1e-6
)
```

### Inference

```python
from constraint_decomposition import ConstraintDecompositionModel

model = ConstraintDecompositionModel.from_pretrained("epaunova/cd-nemotron-7b")

response = model.generate(
    prompt="Explain quantum entanglement in exactly 3 sentences for a high school student.",
    max_length=256
)
print(response)
```

## 📁 Repository Structure

```
constraint-decomposition/
├── configs/
│   ├── reward_training.yaml      # Reward model training config
│   ├── ppo_training.yaml         # PPO training config
│   └── evaluation.yaml           # Evaluation config
├── src/
│   ├── constraint_decomposition/
│   │   ├── __init__.py
│   │   ├── reward_models.py      # Decomposed reward model classes
│   │   ├── hierarchical.py       # Hierarchical combination logic
│   │   ├── conflict_detector.py  # Conflict detection module
│   │   ├── ppo_trainer.py        # PPO training with decomposed rewards
│   │   └── utils.py
│   └── evaluation/
│       ├── ifeval.py             # IFEval benchmark evaluation
│       ├── generalization.py     # GSM8K, HumanEval, MT-Bench
│       └── ablation.py           # Ablation study scripts
├── scripts/
│   ├── train_reward_models.sh    # Train all 4 reward models
│   ├── train_ppo.sh              # PPO training script
│   ├── evaluate.sh               # Full evaluation pipeline
│   └── run_ablation.sh           # Ablation experiments
├── data/
│   ├── preference_data/          # Aspect-specific preference datasets
│   └── evaluation/               # Evaluation datasets
├── docs/
│   ├── NVIDIA_Technical_Report.pdf
│   ├── method_diagram.png
│   └── ablation_plot.png
├── notebooks/
│   ├── reward_analysis.ipynb     # Reward model analysis
│   ├── conflict_detection.ipynb  # Conflict detector evaluation
│   └── results_visualization.ipynb
├── tests/
│   ├── test_reward_models.py
│   ├── test_hierarchical.py
│   └── test_conflict_detector.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🔬 Experiments

### Reproduce Main Results

```bash
# Step 1: Train decomposed reward models (4x parallel)
./scripts/train_reward_models.sh

# Step 2: Train PPO with hierarchical rewards
./scripts/train_ppo.sh --config configs/ppo_training.yaml

# Step 3: Evaluate on all benchmarks
./scripts/evaluate.sh --model ./checkpoints/final
```

### Ablation Studies

```bash
# Run all ablation experiments
./scripts/run_ablation.sh

# Individual ablations
python -m evaluation.ablation --config decomposition_only
python -m evaluation.ablation --config hierarchical_only
python -m evaluation.ablation --config full_system
```

## 📊 Detailed Results

### IFEval Breakdown by Constraint Type

| Constraint Type | Standard RLHF | Ours | Δ |
|-----------------|---------------|------|---|
| Format (length, structure) | 38.2% | 71.4% | +33.2 |
| Content (semantic) | 45.1% | 76.2% | +31.1 |
| Style (tone, audience) | 39.8% | 72.1% | +32.3 |
| Multi-constraint | 31.5% | 69.8% | +38.3 |

### Reward Model Validation Accuracy

| Component | Validation Accuracy | Dataset Size |
|-----------|---------------------|--------------|
| R_semantic | 84.2% | 50K pairs |
| R_structural | 89.1% | 50K pairs |
| R_format | 92.3% | 50K pairs |
| R_meta | 87.4% | 50K pairs |

## 🛠️ Training Configuration

### Hardware Requirements

- **Reward Model Training**: 4× A100 40GB (per model)
- **PPO Training**: 8× A100 80GB
- **Inference**: 1× A100 40GB

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (reward) | 1e-5 |
| Learning rate (PPO) | 1e-6 |
| Batch size | 512 |
| PPO epochs | 4 |
| KL penalty | 0.02 |
| Training steps | 10,000 |

## 📝 Citation

```bibtex
@techreport{paunova2025constraint,
  title={Constraint Decomposition for Multi-Objective RLHF in Large Language Models},
  author={Paunova, Eva},
  institution={NVIDIA Research},
  year={2025},
  month={December},
  address={Zurich, Switzerland}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NVIDIA Research team for infrastructure and support
- ETH Zurich collaborators for valuable feedback
- The open-source community for foundational tools (PyTorch, Transformers, TRL)

## 📧 Contact

**Eva Paunova** - NVIDIA Research, Zurich

- Email: e.hpaunova@gmail.com
- LinkedIn: [linkedin.com/in/epaunova](https://linkedin.com/in/epaunova)
- GitHub: [github.com/epaunova](https://github.com/epaunova)

---

<p align="center">
  <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d.png" width="150">
</p>
