import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from math import gamma as gamma_func
import os
import scipy.linalg

# ====================== 1. MNIST data loaders ======================
def get_mnist_loaders(batch_size=1024):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0)
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=2)

def get_mnist_test_loader(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0)
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ====================== 2. Sinusoidal time embedding ======================
def time_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# ====================== 3. Conditional lightweight UNet (~4.95M params, all channels are multiples of 8) ======================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, inject_t=False, t_ch=256):
        super().__init__()
        self.inject_t = inject_t
        self.conv1 = nn.Conv2d(in_ch + (t_ch if inject_t else 0), out_ch,
                               3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride),
            nn.GroupNorm(8, out_ch)
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x, t_emb=None):
        identity = x
        if self.inject_t and t_emb is not None:
            x = torch.cat([x, t_emb], dim=1)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + self.shortcut(identity))


class ConditionalLightUNet(nn.Module):
    def __init__(self, t_dim=128, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, 256), nn.SiLU(), nn.Linear(256, 256)
        )
        self.class_embed = nn.Embedding(num_classes, 256)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(256, 512))

        # All channel counts are multiples of 8 to satisfy GroupNorm(8, ...) requirements
        self.enc1 = ResBlock(1, 64)
        self.enc2 = ResBlock(64, 152, stride=2)      # 64 -> 152  (152 % 8 == 0)
        self.enc3 = ResBlock(152, 176, stride=2)     # 152 -> 176 (176 % 8 == 0)

        # Wider bottleneck with an extra layer
        self.bottleneck1 = ResBlock(176, 176, inject_t=True, t_ch=256)
        self.bottleneck2 = ResBlock(176, 176)
        self.bottleneck3 = ResBlock(176, 176)        # extra bottleneck layer

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(176, 176, 4, stride=2, padding=1),
            nn.GroupNorm(8, 176), nn.SiLU()
        )
        self.dec1 = ResBlock(328, 176)               # 176 (up1) + 152 (enc2) = 328 -> 176

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(176, 88, 4, stride=2, padding=1),
            nn.GroupNorm(8, 88), nn.SiLU()
        )
        self.dec2 = ResBlock(152, 64)                # 88 (up2) + 64 (enc1) = 152 -> 64

        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t, y=None):
        B = x.size(0)
        if x.dim() == 2:
            x = x.view(B, 1, 28, 28)

        t_emb = time_embedding(t, dim=128)
        t_emb = self.t_mlp(t_emb)  # [B, 256]

        if y is None:
            raise ValueError("Conditional generation requires class labels y")
        c_emb = self.class_embed(y)  # [B, 256]

        film_params = self.film(c_emb)          # [B, 512]
        gate, shift = film_params.chunk(2, dim=-1)
        cond = t_emb * (1.0 + gate) + shift     # [B, 256]
        cond_spatial = cond.view(B, 256, 1, 1)

        e1 = self.enc1(x)          # [B, 64, 28, 28]
        e2 = self.enc2(e1)         # [B, 152, 14, 14]
        e3 = self.enc3(e2)         # [B, 176,  7,  7]
        b = self.bottleneck1(e3, cond_spatial.expand(B, 256, 7, 7))
        b = self.bottleneck2(b)
        b = self.bottleneck3(b)   # [B, 176,  7,  7]
        d1 = self.up1(b)          # [B, 176, 14, 14]
        d1 = torch.cat([d1, e2], dim=1)  # 176 + 152 = 328
        d1 = self.dec1(d1)        # [B, 176, 14, 14]
        d2 = self.up2(d1)         # [B,  88, 28, 28]
        d2 = torch.cat([d2, e1], dim=1)  # 88 + 64 = 152
        d2 = self.dec2(d2)        # [B,  64, 28, 28]
        return self.out(d2).view(B, 784)


# ====================== 4. EMA ======================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.clone().detach()

    def apply_shadow(self, model):
        self.backup = {name: param.clone().detach()
                       for name, param in model.named_parameters() if param.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# ====================== 5. FID evaluation module ======================
class MNISTFeatureNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, feat_dim)
        self._feat_dim = feat_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


def train_mnist_feature_extractor(device='cuda', feat_dim=128, epochs=16,
                                   save_path='mnist_fid_feature_net.pt'):
    if os.path.exists(save_path):
        print(f"[FID] Loading pretrained feature extractor: {save_path}")
        net = MNISTFeatureNet(feat_dim).to(device)
        net.load_state_dict(torch.load(save_path, map_location=device))
        net.eval()
        return net

    print(f"[FID] Training feature extractor on real MNIST ...")
    train_loader = get_mnist_loaders(batch_size=256)
    net = MNISTFeatureNet(feat_dim).to(device)
    clf = nn.Linear(feat_dim, 10).to(device)

    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        net.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feat = net(x)
            logits = clf(feat)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"  Epoch {ep+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {correct/total:.4f}")

    torch.save(net.state_dict(), save_path)
    print(f"[FID] Saved to {save_path}")
    net.eval()
    return net


@torch.no_grad()
def extract_features(feature_net, images, batch_size=512, device='cuda'):
    feature_net.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        feat = feature_net(batch)
        feats.append(feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(real_feats, gen_feats):
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)

# ====================== 6. Training ======================
def train(model, train_loader, alpha=1.0, N=100, epochs=500,
          lr=1e-3, weight_decay=1e-4, device='cuda'):
    model = model.to(device)
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_epochs = max(1, int(epochs * 0.05))
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / float(warmup_epochs)
        progress = (ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    h = 1.0 / N
    v_scale = gamma_func(1 + alpha)

    model.train()
    for ep in range(epochs):
        losses = []
        for x_batch, y_batch in train_loader:
            x1 = x_batch.view(x_batch.size(0), -1).to(device)
            y = y_batch.to(device)
            B = x1.size(0)
            x0 = torch.randn_like(x1)

            k = torch.randint(0, N, (B,), device=device)
            t = k.float() * h

            x_k = x0 + (x1 - x0) * (t[:, None] ** alpha)
            v_target = v_scale * (x1 - x0)

            v_pred = model(x_k, t, y)
            loss = ((v_pred - v_target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

        scheduler.step()
        ema.update(model)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1:3d}/{epochs} | Loss: {np.mean(losses):.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return model, ema

# ====================== 7. Conditional Caputo sampler =======================
@torch.no_grad()
def sample_caputo(model, x0, alpha, N=100, device='cuda', y=None):
    model.eval()
    h = 1.0 / N
    B = x0.shape[0]
    scale = (h ** alpha) / gamma_func(1 + alpha)

    if y is None:
        raise ValueError("Conditional generation requires class labels y (0-9)")
    if isinstance(y, int):
        y = torch.full((B,), y, dtype=torch.long, device=device)
    elif isinstance(y, (list, tuple, np.ndarray)):
        y = torch.tensor(y, dtype=torch.long, device=device)
    elif y.dim() == 0:
        y = y.unsqueeze(0).expand(B)

    vs = []

    for k in range(N):
        if k == 0:
            x = x0.clone()
        else:
            diffs = torch.arange(k, 0, -1, device=device).float()
            w_raw = diffs ** alpha
            w = torch.zeros(k, device=device)
            if k > 1:
                w[:-1] = w_raw[:-1] - w_raw[1:]
            w[-1] = w_raw[-1]

            v_stack = torch.stack(vs)
            weighted_sum = (w.view(-1, 1, 1) * v_stack).sum(dim=0)
            x = x0 + scale * weighted_sum

        t = torch.full((B,), k * h, device=device)
        v_k = model(x, t, y)
        vs.append(v_k)

    diffs = torch.arange(N, 0, -1, device=device).float()
    w_raw = diffs ** alpha
    w = torch.zeros(N, device=device)
    if N > 1:
        w[:-1] = w_raw[:-1] - w_raw[1:]
    w[-1] = w_raw[-1]

    v_stack = torch.stack(vs)
    weighted_sum = (w.view(-1, 1, 1) * v_stack).sum(dim=0)
    x_N = x0 + scale * weighted_sum

    return x_N

# ====================== 8. Visualization ======================
@torch.no_grad()
def visualize_all_digits(model, ema, alpha, N=100, samples_per_digit=8,
                         device='cuda', save_path=None):
    model.eval()
    ema.apply_shadow(model)
    try:
        all_images = []
        for digit in range(10):
            x0 = torch.randn(samples_per_digit, 784, device=device)
            y = torch.full((samples_per_digit,), digit, dtype=torch.long, device=device)
            x_gen = sample_caputo(model, x0, alpha, N=N, device=device, y=y)
            x_gen = torch.clamp((x_gen + 1.0) / 2.0, 0.0, 1.0)
            all_images.append(x_gen.cpu().numpy())

        fig, axes = plt.subplots(10, samples_per_digit, figsize=(2 * samples_per_digit, 20))
        for digit in range(10):
            for i in range(samples_per_digit):
                ax = axes[digit, i]
                ax.imshow(all_images[digit][i].reshape(28, 28), cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_ylabel(str(digit), fontsize=20, rotation=0, labelpad=20, va='center')

        plt.suptitle(f"Conditional Generation | α={alpha:.2f}, N={N}\n"
                     f"Rows: digits 0-9, Columns: random samples", fontsize=16)
        plt.tight_layout()

        if save_path is None:
            save_path = f"cond_caputo_alpha_{alpha:.2f}_N{N}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        print(f"Saved conditional visualization to {save_path}")
    finally:
        ema.restore(model)

# ====================== 9. Main program ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    N_STEPS = 100
    EPOCHS = 50
    BATCH_SIZE = 512
    N_FID_PER_DIGIT = 500
    FID_BATCH_SIZE = 256

    # ALPHAS = [1.1, 1.0, 0.98, 0.96, 0.94]
    # ALPHAS = [1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
    ALPHAS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

    feat_net = train_mnist_feature_extractor(device=device, feat_dim=128)
    feat_net.eval()

    print("[FID] Extracting features from real MNIST (per-digit)...")
    test_loader = get_mnist_test_loader(batch_size=FID_BATCH_SIZE)
    real_feats_by_digit = {d: [] for d in range(10)}
    for x, y in test_loader:
        x = x.to(device)
        feat = feat_net(x).detach().cpu().numpy()
        for d in range(10):
            mask = (y == d).numpy()
            if mask.any():
                real_feats_by_digit[d].append(feat[mask])

    real_feats_all = {}
    for d in range(10):
        if len(real_feats_by_digit[d]) > 0:
            real_feats_all[d] = np.concatenate(real_feats_by_digit[d], axis=0)
        else:
            real_feats_all[d] = np.zeros((1, 128), dtype=np.float32)
        print(f"  Digit {d}: {real_feats_all[d].shape[0]} real samples")

    fid_results = {}

    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f">>> Conditional Caputo | α = {alpha:.2f}  (N={N_STEPS})")
        print(f"{'='*60}")

        train_loader = get_mnist_loaders(batch_size=BATCH_SIZE)

        model = ConditionalLightUNet(t_dim=128, num_classes=10)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {n_params:.2f}M")

        model, ema = train(model, train_loader, alpha=alpha, N=N_STEPS,
                           epochs=EPOCHS, device=device)

        ckpt_path = f"cond_caputopower_ckpt_alpha_{alpha:.2f}_N{N_STEPS}.pt"
        torch.save({
            'model': model.state_dict(),
            'ema_shadow': ema.shadow,
            'alpha': alpha,
            'N': N_STEPS,
            'epochs': EPOCHS,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        print(f"\n>>> Generating {N_FID_PER_DIGIT} samples per digit for FID...")
        ema.apply_shadow(model)
        gen_feats_by_digit = {d: [] for d in range(10)}
        try:
            with torch.no_grad():
                for digit in range(10):
                    n_batches = (N_FID_PER_DIGIT + 255) // 256
                    digit_imgs = []
                    for _ in range(n_batches):
                        bs = min(256, N_FID_PER_DIGIT - len(digit_imgs) * 256)
                        if bs <= 0:
                            break
                        x0 = torch.randn(bs, 784, device=device)
                        y = torch.full((bs,), digit, dtype=torch.long, device=device)
                        x_gen = sample_caputo(model, x0, alpha=alpha, N=N_STEPS, device=device, y=y)
                        digit_imgs.append(x_gen.cpu())
                    digit_imgs = torch.cat(digit_imgs, dim=0)[:N_FID_PER_DIGIT]
                    gen_feats = extract_features(feat_net, digit_imgs, device=device)
                    gen_feats_by_digit[digit] = gen_feats
                    print(f"  Digit {digit}: generated {gen_feats.shape[0]} samples")
        finally:
            ema.restore(model)

        fids = {}
        for d in range(10):
            fids[d] = calculate_fid(real_feats_all[d], gen_feats_by_digit[d])
        avg_fid = np.mean(list(fids.values()))
        fid_results[alpha] = (fids, avg_fid)

        print(f"\n{'='*60}")
        print(f"[FID RESULT] α={alpha:.2f}")
        for d in range(10):
            print(f"  Digit {d}: FID = {fids[d]:.2f}")
        print(f"  Average FID = {avg_fid:.2f}")
        print(f"{'='*60}")

        print(f"\n>>> Conditional visualization (α={alpha:.2f})")
        visualize_all_digits(model, ema, alpha=alpha, N=N_STEPS,
                             samples_per_digit=8, device=device)

    print(f"\n{'='*60}")
    print("FID Summary (lower is better):")
    for alpha, (fids, avg_fid) in sorted(fid_results.items()):
        print(f"\nα = {alpha:.2f} | Avg FID = {avg_fid:.2f}")
        for d in range(10):
            print(f"  Digit {d}: {fids[d]:.2f}")
    print(f"{'='*60}")
    print("Training finished.")