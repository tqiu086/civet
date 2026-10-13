import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

sys.modules["utils"] = types.ModuleType("utils")

import torch 
import torchvision 
import torchvision.transforms as T 
from torch.distributions import Normal 

from ibp_modules import RobustModule 
from model import FCVAE 

DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
RESULT_DIR = os.path.join(HERE, "results")


def set_robust(model, flag):
    for m in model.modules():
        if isinstance(m, RobustModule):
            m.r = bool(flag)


def mnist(train, device="cpu"):
    ds = torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True,
                                    transform=T.ToTensor())
    return ds


def load_model(name, device="cuda"):
    m = FCVAE(10, 28, 1, 512, ibp=True)
    ck = torch.load(os.path.join(MODEL_DIR, name), map_location="cpu")
    m.load_state_dict(ck["state"])
    m.to(device).eval()
    set_robust(m, False)
    return m


def civet_xi(width, sigma_u, delta, iters=30):
    n = Normal(torch.zeros((), dtype=torch.float64, device=width.device),
               torch.ones((), dtype=torch.float64, device=width.device))
    w = width.double()
    s = sigma_u.double()
    lo = torch.zeros_like(w)
    hi = s * 12.0 
    target = 1.0 - delta
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = n.cdf(mid / s) - n.cdf(-(w + mid) / s)
        ok = p >= target
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi.to(width.dtype)


def sym_c(delta):
    return float(Normal(0.0, 1.0).icdf(torch.tensor(1 - delta / 2)))
