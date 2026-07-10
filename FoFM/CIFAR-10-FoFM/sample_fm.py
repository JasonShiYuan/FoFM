"""Sample images from a trained CIFAR-10 Flow Matching (FM) checkpoint.

Example:
    python sample_fm.py --weights fm_cifar10_weights_step_400000.pt \
                        --num_samples 64 --output fm_samples.png
"""

import os

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import sample_caputo

FLAGS = flags.FLAGS

flags.DEFINE_string("weights", "fm_cifar10_weights_step_400000.pt",
                    "Path to the checkpoint .pt file.")
flags.DEFINE_string("output", "fm_samples.png", "Output image grid path.")
flags.DEFINE_integer("num_samples", 64, "Number of images to generate.")
flags.DEFINE_integer("nrow", 8, "Number of images per row in the saved grid.")
flags.DEFINE_integer("integration_steps", 100,
                     "Number of Euler steps for the ODE solver.")
flags.DEFINE_string("solver", "euler", "torchdyn ODE solver name.")
flags.DEFINE_bool("use_ema", True,
                  "Use the EMA weights (recommended) instead of the raw network.")
flags.DEFINE_float("alpha", -1.0, help="fractional order alpha; use checkpoint value if negative")


def load_model(weights_path, use_ema=True):
    """Build the UNet and load the checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["ema_model"] if use_ema else checkpoint["net_model"]
    alpha = checkpoint.get("alpha", 1.0)
    N_steps = checkpoint.get("N_steps", 100)

    # Handle possible `module.` prefix from DataParallel/DistributedDataParallel.
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith("module.") else k] = v
        net.load_state_dict(new_state_dict)

    net.eval()
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')} on {device}")
    return net, device, alpha, N_steps


def sample(net, device, num_samples=64, solver="euler", integration_steps=100, alpha=1.0, N_steps=100):
    """Run the ODE from t=0 to t=1 starting from Gaussian noise."""
    x = torch.randn(num_samples, 3, 32, 32, device=device)

    with torch.no_grad():
        if alpha == 1.0:
            node = NeuralODE(net, solver=solver, sensitivity="adjoint")
            t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
            images = traj[-1]
        else:
            images = sample_caputo(net, x, alpha=alpha, N=N_steps, device=device)

    # Last timestep, clip to [-1, 1], then map to [0, 1].
    images = images.view(-1, 3, 32, 32).clip(-1, 1) / 2 + 0.5
    return images


def main(argv):
    net, device, checkpoint_alpha, checkpoint_N_steps = load_model(FLAGS.weights, use_ema=FLAGS.use_ema)
    alpha = FLAGS.alpha if FLAGS.alpha > 0 else checkpoint_alpha
    N_steps = checkpoint_N_steps
    images = sample(
        net,
        device,
        num_samples=FLAGS.num_samples,
        solver=FLAGS.solver,
        integration_steps=FLAGS.integration_steps,
        alpha=alpha,
        N_steps=N_steps,
    )
    save_image(images, FLAGS.output, nrow=FLAGS.nrow)
    print(f"Saved {FLAGS.num_samples} generated images to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
