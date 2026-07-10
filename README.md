# Fractional-order Flow Matching (FoFM)

PyTorch implementation of **Fractional-order Flow Matching (FoFM)**.

FoFM extends standard Flow Matching by replacing the integer-order probability path with a Caputo fractional ODE. For fractional order `α`:

```
xt = x0 + (x1 - x0) * t^α
ut = Γ(1 + α) * (x1 - x0)
```

When `α = 1.0`, FoFM reduces to standard Conditional Flow Matching. Sampling for `α ≠ 1.0` uses a discrete Caputo fractional integrator.

## Repository Structure

```text
FoFM/
├── CIFAR-10-FoFM/        # CIFAR-10 unconditional generation
│   ├── train_cifar10.py
│   ├── train_cifar10_ddp.py
│   ├── sample_fm.py
│   ├── compute_fid.py
│   └── utils_cifar.py
├── minist/               # Conditional MNIST generation
│   └── Caputo_mi_FID_Condition_5M.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

### CIFAR-10

```bash
cd FoFM/CIFAR-10-FoFM
python train_cifar10.py --model "fm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --alpha 0.95 --N_steps 100
python compute_fid.py --model "fm" --step 400000 --integration_method euler
```

### MNIST

```bash
cd FoFM/minist
python Caputo_mi_FID_Condition_5M.py
```

See the source files for available flags and hyper-parameters.

## Acknowledgements

The CIFAR-10 training pipeline is built on top of [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) by Tong et al. We thank the authors for the excellent codebase and the `torchcfm` library.

## Citation

<!-- If you use this code, please cite:

```bibtex
@article{your_paper_key,
  title={Fractional-order Flow Matching},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
``` -->
