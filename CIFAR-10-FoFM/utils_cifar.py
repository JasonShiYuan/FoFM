import copy
import math
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def sample_fractional_flow(x0, x1, alpha, N, device):
    """Sample (t, xt, ut) for fractional-order flow matching.

    Uses the Caputo-style discrete formulation from the reference code:
        xt = x0 + (x1 - x0) * t^alpha
        ut = Gamma(1 + alpha) * (x1 - x0)

    Parameters
    ----------
    x0, x1 : torch.Tensor
        Source and target samples of shape (B, C, H, W).
    alpha : float
        Fractional order. alpha == 1.0 recovers standard linear flow.
    N : int
        Number of discrete time steps.
    device : torch.device

    Returns
    -------
    t, xt, ut : torch.Tensor
        Time (B,), location (B, C, H, W), and target velocity (B, C, H, W).
    """
    B = x0.size(0)
    h = 1.0 / N
    k = torch.randint(0, N, (B,), device=device)
    t = k.float() * h
    t_expanded = t.view(B, 1, 1, 1)
    xt = x0 + (x1 - x0) * (t_expanded ** alpha)
    ut = math.gamma(1 + alpha) * (x1 - x0)
    return t, xt, ut


def sample_caputo(model, x0, alpha, N, device):
    """Discrete Caputo fractional integration sampler.

    Adapts the reference sample_caputo to work with UNetModelWrapper,
    which keeps images in (B, 3, 32, 32) format and expects model(t, x).

    Parameters
    ----------
    model : nn.Module
        Velocity network. Called as model(t, x) with x of shape (B, 3, 32, 32).
    x0 : torch.Tensor
        Gaussian noise of shape (B, 3, 32, 32).
    alpha : float
        Fractional order.
    N : int
        Number of integration steps.
    device : torch.device

    Returns
    -------
    torch.Tensor
        Generated samples of shape (B, 3, 32, 32).
    """
    model.eval()
    h = 1.0 / N
    B = x0.shape[0]
    scale = (h ** alpha) / math.gamma(1 + alpha)

    original_shape = x0.shape
    x0_flat = x0.view(B, -1)
    vs = []

    with torch.no_grad():
        for k in range(N):
            if k == 0:
                x = x0_flat.clone()
            else:
                diffs = torch.arange(k, 0, -1, device=device).float()
                w_raw = diffs ** alpha
                w = torch.zeros(k, device=device)
                if k > 1:
                    w[:-1] = w_raw[:-1] - w_raw[1:]
                w[-1] = w_raw[-1]

                v_stack = torch.stack(vs)
                weighted_sum = (w.view(-1, 1, 1) * v_stack).sum(dim=0)
                x = x0_flat + scale * weighted_sum

            t = torch.full((B,), k * h, device=device)
            x_spatial = x.view(original_shape)
            v_k = model(t, x_spatial).view(B, -1)
            vs.append(v_k)

        diffs = torch.arange(N, 0, -1, device=device).float()
        w_raw = diffs ** alpha
        w = torch.zeros(N, device=device)
        if N > 1:
            w[:-1] = w_raw[:-1] - w_raw[1:]
        w[-1] = w_raw[-1]

        v_stack = torch.stack(vs)
        weighted_sum = (w.view(-1, 1, 1) * v_stack).sum(dim=0)
        x_N = x0_flat + scale * weighted_sum

    return x_N.view(original_shape)


def generate_samples(model, parallel, savedir, step, net_="normal", alpha=1.0, N=100):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    alpha: float
        Fractional order. alpha == 1.0 uses standard NeuralODE; otherwise uses Caputo sampling.
    N: int
        Number of integration/discretization steps.
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    with torch.no_grad():
        x0 = torch.randn(64, 3, 32, 32, device=device)
        if alpha == 1.0:
            node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
            traj = node_.trajectory(
                x0,
                t_span=torch.linspace(0, 1, N + 1, device=device),
            )
            traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        else:
            traj = sample_caputo(model_, x0, alpha=alpha, N=N, device=device).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x