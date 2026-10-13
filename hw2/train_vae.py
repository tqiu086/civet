"""Train FC-VAE on MNIST, standard or CIVET (IBP-certified decoder error).

usage: python train_vae.py {standard|civet} NAME [--epochs N] [--eps 0.1]
Re-implements src/train.py's loop because that one cannot run (model.ibp does
not exist, robust()/unrobust() skip q_logvar/projection, and it never switches
back to robust mode for the CIVET term).
"""
import argparse
import os
import time

from common import *  # noqa: F401,F403
import common
import torch
import torch.nn.functional as F
from torch import optim

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["standard", "civet"])
p.add_argument("name")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eps", type=float, default=0.1)
p.add_argument("--deltas", type=float, nargs="+", default=[0.35, 0.20, 0.05])
p.add_argument("--std_epochs", type=int, default=2)
p.add_argument("--ramp_epochs", type=int, default=5)
p.add_argument("--lam", type=float, default=1.0, help="weight of CIVET MSE term")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "cuda"
X = torch.stack([x for x, _ in common.mnist(True)]).to(dev)  # N,1,28,28
N = X.shape[0]
model = FCVAE(10, 28, 1, 512, ibp=True).to(dev)
opt = optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
deltas = sorted(a.deltas, reverse=True)
weights = [1 - deltas[0]] + [d0 - d1 for d0, d1 in zip(deltas[:-1], deltas[1:])]
steps_per_epoch = N // a.bs
std_it, ramp_it = a.std_epochs * steps_per_epoch, a.ramp_epochs * steps_per_epoch


def civet_outputs(x, eps):
    """IBP latent -> CIVET support box (per delta) -> IBP decoder bounds."""
    xu, xl = torch.clamp(x + eps, 0, 1), torch.clamp(x - eps, 0, 1)
    common.set_robust(model, True)
    mean, logvar = model.q(model.encoder(torch.cat([xu, xl], 0)), 2 * x.shape[0])
    B = x.shape[0]
    muu, mul = mean[:B], mean[B:]
    sig_u = (0.5 * logvar[:B]).exp()  # upper bound of sigma (upper half!)
    outs = []
    for d in deltas:
        with torch.no_grad():
            ratio = common.civet_xi(muu - mul, sig_u, d) / sig_u
        xi = sig_u * ratio  # grad flows through sigma only
        z = torch.cat([muu + xi, mul - xi], 0)
        outs.append(model.decoder(model.project(z)).view(2 * B, 1, 28, 28))
    common.set_robust(model, False)
    return outs


it = 0
t0 = time.time()
for ep in range(a.epochs):
    perm = torch.randperm(N, device=dev)
    tot = tot_c = 0.0
    for b in range(steps_per_epoch):
        x = X[perm[b * a.bs:(b + 1) * a.bs]]
        common.set_robust(model, False)
        (mean, logvar), xr = model(x)
        rec = F.binary_cross_entropy(xr, x)
        kl = model.kl_divergence_loss(mean, logvar) / x.shape[0] / 784
        loss = rec + kl
        civ = torch.zeros((), device=dev)
        if a.mode == "civet" and it >= std_it:
            eps = a.eps * min(1.0, (it - std_it) / ramp_it)
            if eps > 0:
                for w, out in zip(weights, civet_outputs(x, eps)):
                    xru, xrl = out[:x.shape[0]], out[x.shape[0]:]
                    worst = torch.where((xru - x).abs() > (xrl - x).abs(), xru, xrl)
                    civ = civ + w * F.mse_loss(worst, x)
                loss = loss + a.lam * civ
        opt.zero_grad()
        loss.backward()
        opt.step()
        it += 1
        tot += loss.item()
        tot_c += civ.item()
    print(f"ep {ep+1}/{a.epochs} loss {tot/steps_per_epoch:.4f} civet {tot_c/steps_per_epoch:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(common.MODEL_DIR, exist_ok=True)
    torch.save({"state": model.state_dict(), "epoch": ep, "args": vars(a)},
               os.path.join(common.MODEL_DIR, a.name))
