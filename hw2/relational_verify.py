import argparse
import json
import math
import os
import time

from common import *  # noqa
import common
import torch
import torch.nn.functional as F

dev = "cuda"
D = torch.float64


class S:
    """Affine lower/upper expressions  Lw v + Lb <= f(v) <= Uw v + Ub  (rows = neurons)."""
    def __init__(s, Lw, Lb, Uw, Ub):
        s.Lw, s.Lb, s.Uw, s.Ub = Lw, Lb, Uw, Ub

    def sel(s, mask, other):
        m = mask[:, None]
        return S(torch.where(m, s.Lw, other.Lw), torch.where(mask, s.Lb, other.Lb),
                 torch.where(m, s.Uw, other.Uw), torch.where(mask, s.Ub, other.Ub))


def zeros_like_S(q):
    return S(torch.zeros_like(q.Lw), torch.zeros_like(q.Lb), torch.zeros_like(q.Uw), torch.zeros_like(q.Ub))


def neg(q):
    return S(-q.Uw, -q.Ub, -q.Lw, -q.Lb)


def sub(a, b):  # a - b
    return S(a.Lw - b.Uw, a.Lb - b.Ub, a.Uw - b.Lw, a.Ub - b.Lb)


def conc(q, vlo, vhi):
    lo = q.Lw.clamp(min=0) @ vlo + q.Lw.clamp(max=0) @ vhi + q.Lb
    up = q.Uw.clamp(min=0) @ vhi + q.Uw.clamp(max=0) @ vlo + q.Ub
    return lo, up


def conc_L(Lw, Lb, vlo, vhi):
    return Lw.clamp(min=0) @ vlo + Lw.clamp(max=0) @ vhi + Lb


def conc_U(Uw, Ub, vlo, vhi):
    return Uw.clamp(min=0) @ vhi + Uw.clamp(max=0) @ vlo + Ub


def affine(q, W, b=None):
    Wp, Wn = W.clamp(min=0), W.clamp(max=0)
    Lb = Wp @ q.Lb + Wn @ q.Ub
    Ub = Wp @ q.Ub + Wn @ q.Lb
    if b is not None:
        Lb, Ub = Lb + b, Ub + b
    return S(Wp @ q.Lw + Wn @ q.Uw, Lb, Wp @ q.Uw + Wn @ q.Lw, Ub)


def refine(b1, b2, bd):
    """Intersect concrete bounds using q1 = q2 + d."""
    l1, u1, l2, u2, dl, du = b1[0], b1[1], b2[0], b2[1], bd[0], bd[1]
    for _ in range(2):
        l1, u1 = torch.maximum(l1, l2 + dl), torch.minimum(u1, u2 + du)
        l2, u2 = torch.maximum(l2, l1 - du), torch.minimum(u2, u1 - dl)
        dl, du = torch.maximum(dl, l1 - u2), torch.minimum(du, u1 - l2)
    return (l1, u1), (l2, u2), (dl, du)


def relu_ind(q, l, u):
    act, inact = l >= 0, u <= 0
    amb = ~(act | inact)
    s = torch.where(amb, u / (u - l).clamp(min=1e-30), act.to(D))
    lam = torch.where(amb, (u > -l).to(D), act.to(D))
    Ub = s * q.Ub - torch.where(amb, s * l, torch.zeros_like(l))
    return S(lam[:, None] * q.Lw, lam * q.Lb, s[:, None] * q.Uw, Ub)


def relu_diff(q1, q2, qd, h1, h2, b1, b2, bd, vlo, vhi):
    """Symbolic bounds on relu(a1)-relu(a2) given tracker d=a1-a2 (qd)."""
    (l1, u1), (l2, u2), (dl, du) = b1, b2, bd
    C1, C2 = (l1 >= 0) & (l2 >= 0), (u1 <= 0) & (u2 <= 0)
    C3, C4 = (l1 >= 0) & (u2 <= 0), (u1 <= 0) & (l2 >= 0)
    z = zeros_like_S(qd)
    exL = z.sel(C1, z) 
    exL = qd.sel(C1, q1.sel(C3, neg(q2).sel(C4, z)))
    exact = C1 | C2 | C3 | C4
    t = dl / (du - dl).clamp(min=1e-30)
    sd = du / (du - dl).clamp(min=1e-30)
    amb_d = (dl < 0) & (du > 0)
    gL_Lw = torch.where((du <= 0)[:, None], qd.Lw, torch.where(amb_d[:, None], (-t)[:, None] * qd.Lw, torch.zeros_like(qd.Lw)))
    gL_Lb = torch.where(du <= 0, qd.Lb, torch.where(amb_d, (-t) * qd.Lb + t * du, torch.zeros_like(dl)))
    gU_Uw = torch.where((dl >= 0)[:, None], qd.Uw, torch.where(amb_d[:, None], sd[:, None] * qd.Uw, torch.zeros_like(qd.Uw)))
    gU_Ub = torch.where(dl >= 0, qd.Ub, torch.where(amb_d, sd * qd.Ub - sd * dl, torch.zeros_like(dl)))
    bs = sub(h1, h2)
    cands_L = [(bs.Lw, bs.Lb), (gL_Lw, gL_Lb), (exL.Lw, exL.Lb)]
    cands_U = [(bs.Uw, bs.Ub), (gU_Uw, gU_Ub), (exL.Uw, exL.Ub)]
    best_Lw, best_Lb = cands_L[0]
    best_v = conc_L(best_Lw, best_Lb, vlo, vhi)
    for i, (w, b) in enumerate(cands_L[1:], 1):
        v = conc_L(w, b, vlo, vhi)
        ok = (v > best_v) & (exact if i == 2 else torch.ones_like(exact))
        best_Lw, best_Lb, best_v = torch.where(ok[:, None], w, best_Lw), torch.where(ok, b, best_Lb), torch.where(ok, v, best_v)
    best_Uw, best_Ub = cands_U[0]
    best_u = conc_U(best_Uw, best_Ub, vlo, vhi)
    for i, (w, b) in enumerate(cands_U[1:], 1):
        v = conc_U(w, b, vlo, vhi)
        ok = (v < best_u) & (exact if i == 2 else torch.ones_like(exact))
        best_Uw, best_Ub, best_u = torch.where(ok[:, None], w, best_Uw), torch.where(ok, b, best_Ub), torch.where(ok, v, best_u)
    return S(best_Lw, best_Lb, best_Uw, best_Ub)


def relu_triple(q1, q2, qd, b1, b2, bd, vlo, vhi, use_diff):
    if use_diff:
        b1, b2, bd = refine(b1, b2, bd)
    h1, h2 = relu_ind(q1, *b1), relu_ind(q2, *b2)
    if use_diff:
        hd = relu_diff(q1, q2, qd, h1, h2, b1, b2, bd, vlo, vhi)
    else:
        hd = sub(h1, h2)
    hb1, hb2 = (b1[0].clamp(min=0), b1[1].clamp(min=0)), (b2[0].clamp(min=0), b2[1].clamp(min=0))
    hbd = (hb1[0] - hb2[1], hb1[1] - hb2[0])
    if use_diff:
        c = conc(hd, vlo, vhi)
        hbd = (torch.maximum(hbd[0], c[0]), torch.minimum(hbd[1], c[1]))
        hb1, hb2, hbd = refine(hb1, hb2, hbd)
    return h1, h2, hd, (hb1, hb2, hbd)


def iaff(b, W, bias=None):
    m, r = (b[0] + b[1]) / 2, (b[1] - b[0]) / 2
    mm = W @ m + (bias if bias is not None else 0)
    rr = W.abs() @ r
    return mm - rr, mm + rr


def meet(a, b):
    return torch.maximum(a[0], b[0]), torch.minimum(a[1], b[1])


def sig_np(p):
    return torch.sigmoid(p)


def dsig(p):
    s = torch.sigmoid(p)
    return s * (1 - s)





def weights(model):
    g = lambda m: (m.weight.detach().to(D), m.bias.detach().to(D))
    W1, b1 = g(model.encoder[1])
    W2, b2 = g(model.encoder[3])
    Wm, bm = g(model.q_mean)
    Wl, bl = g(model.q_logvar)
    Wp, bp = g(model.projection)
    Wd0, bd0 = g(model.decoder[0])
    Wd1, bd1 = g(model.decoder[2])
    return dict(W1=W1, b1=b1,
                Wh=torch.cat([Wm @ W2, Wl @ W2], 0), bh=torch.cat([Wm @ b2 + bm, Wl @ b2 + bl], 0),
                A=Wd0 @ Wp, bA=Wd0 @ bp + bd0, Wd1=Wd1, bd1=bd1)





def encoder_pair(w, x1, x2, eps, use_diff):
    """x1,x2: flat float64 [784]. returns mu bounds for both, d-mu bounds, sigma upper."""
    n = x1.numel()
    vlo, vhi = -eps * torch.ones(n, dtype=D, device=dev), eps * torch.ones(n, dtype=D, device=dev)
    W1 = w["W1"]
    q1 = S(W1, W1 @ x1 + w["b1"], W1, W1 @ x1 + w["b1"])
    q2 = S(W1, W1 @ x2 + w["b1"], W1, W1 @ x2 + w["b1"])
    zc = torch.zeros_like(W1)
    qd = S(zc, W1 @ (x1 - x2), zc, W1 @ (x1 - x2))
    b1, b2, bd = conc(q1, vlo, vhi), conc(q2, vlo, vhi), conc(qd, vlo, vhi)
    h1, h2, hd, hb = relu_triple(q1, q2, qd, b1, b2, bd, vlo, vhi, use_diff)
    H1, H2, Hd = affine(h1, w["Wh"], w["bh"]), affine(h2, w["Wh"], w["bh"]), affine(hd, w["Wh"], None)
    c1, c2, cd = (meet(conc(H1, vlo, vhi), iaff(hb[0], w["Wh"], w["bh"])),
                  meet(conc(H2, vlo, vhi), iaff(hb[1], w["Wh"], w["bh"])),
                  meet(conc(Hd, vlo, vhi), iaff(hb[2], w["Wh"])))
    if use_diff:
        c1, c2, cd = refine(c1, c2, cd)
    d = w["Wh"].shape[0] // 2
    mu = [(c1[0][:d], c1[1][:d]), (c2[0][:d], c2[1][:d])]
    dmu = (cd[0][:d], cd[1][:d])
    sig_u = [(0.5 * c1[1][d:]).exp(), (0.5 * c2[1][d:]).exp()]
    return mu, dmu, sig_u


def encoder_ibp(w, x, eps):
    l, u = x - eps, x + eps

    def aff(l, u, W, b):
        m, r = (l + u) / 2, (u - l) / 2
        mm = W @ m + b
        rr = W.abs() @ r
        return mm - rr, mm + rr
    l, u = aff(l, u, w["W1"], w["b1"])
    l, u = l.clamp(min=0), u.clamp(min=0)
    # head matrix already folds encoder[3]
    l, u = aff(l, u, w["Wh"], w["bh"])
    d = w["Wh"].shape[0] // 2
    return (l[:d], u[:d]), (0.5 * u[d:]).exp()





def decoder_pair(w, z1b, z2b, dzb, mode):
    """mode 'diff': variables (z1, z1-z2) + tracker ; mode 'indep': variables (z1, z2), no tracker."""
    A, bA = w["A"], w["bA"]
    m = A.shape[1]
    Z = torch.zeros_like(A)
    if mode == "diff":
        vlo, vhi = torch.cat([z1b[0], dzb[0]]), torch.cat([z1b[1], dzb[1]])
        W1v, W2v, Wdv = torch.cat([A, Z], 1), torch.cat([A, -A], 1), torch.cat([Z, A], 1)
        use_diff = True
    else:
        vlo, vhi = torch.cat([z1b[0], z2b[0]]), torch.cat([z1b[1], z2b[1]])
        W1v, W2v, Wdv = torch.cat([A, Z], 1), torch.cat([Z, A], 1), torch.cat([A, -A], 1)
        use_diff = False
    q1 = S(W1v, bA, W1v, bA)
    q2 = S(W2v, bA, W2v, bA)
    qd = S(Wdv, torch.zeros_like(bA), Wdv, torch.zeros_like(bA))
    b1, b2, bd = conc(q1, vlo, vhi), conc(q2, vlo, vhi), conc(qd, vlo, vhi)
    h1, h2, hd, hb = relu_triple(q1, q2, qd, b1, b2, bd, vlo, vhi, use_diff)
    P1, P2, Pd = affine(h1, w["Wd1"], w["bd1"]), affine(h2, w["Wd1"], w["bd1"]), affine(hd, w["Wd1"], None)
    p1, p2, pd = (meet(conc(P1, vlo, vhi), iaff(hb[0], w["Wd1"], w["bd1"])),
                  meet(conc(P2, vlo, vhi), iaff(hb[1], w["Wd1"], w["bd1"])),
                  meet(conc(Pd, vlo, vhi), iaff(hb[2], w["Wd1"])))
    if use_diff:
        p1, p2, pd = refine(p1, p2, pd)
    o1 = (sig_np(p1[0]), sig_np(p1[1]))
    o2 = (sig_np(p2[0]), sig_np(p2[1]))
    od = (o1[0] - o2[1], o1[1] - o2[0])
    if use_diff:
        lo_all, hi_all = torch.minimum(p1[0], p2[0]), torch.maximum(p1[1], p2[1])
        smin = torch.minimum(dsig(lo_all), dsig(hi_all))
        dl, du = pd
        odl = dl * torch.where(dl < 0, torch.full_like(dl, 0.25), smin)
        odu = du * torch.where(du > 0, torch.full_like(du, 0.25), smin)
        od = (torch.maximum(od[0], odl), torch.minimum(od[1], odu))
        o1, o2, od = refine(o1, o2, od)
    return o1, o2, od


def decoder_ibp(w, zlo, zhi):
    def aff(l, u, W, b):
        m, r = (l + u) / 2, (u - l) / 2
        mm = W @ m + b
        rr = W.abs() @ r
        return mm - rr, mm + rr
    l, u = aff(zlo, zhi, w["A"], w["bA"])
    l, u = l.clamp(min=0), u.clamp(min=0)
    l, u = aff(l, u, w["Wd1"], w["bd1"])
    return sig_np(l), sig_np(u)


def mse_ub(o, x):
    return ((torch.maximum((o[0] - x) ** 2, (o[1] - x) ** 2)).mean()).item()


def _corner_max(a, b, f):
    """max of f over the 4 corners of box a x box b (f convex => exact max over the box)."""
    best = None
    for p in (a[0], a[1]):
        for q in (b[0], b[1]):
            v = f(p, q)
            best = v if best is None else torch.maximum(best, v)
    return best


def joint_ub(o1, o2, od, x1, x2, eps, noisy):
    pad = eps if noisy else 0.0
    r1 = (o1[0] - x1 - pad, o1[1] - x1 + pad)
    r2 = (o2[0] - x2 - pad, o2[1] - x2 + pad)
    g = x1 - x2
    rd = (od[0] - g, od[1] - g)
    rd = (torch.maximum(rd[0], r1[0] - r2[1]), torch.minimum(rd[1], r1[1] - r2[0]))
    m1 = torch.maximum(r1[0] ** 2, r1[1] ** 2)
    m2 = torch.maximum(r2[0] ** 2, r2[1] ** 2)
    ind = m1 + m2
    via1 = _corner_max(r1, rd, lambda a, d: a ** 2 + (a - d) ** 2)  # r2 = r1 - rd
    via2 = _corner_max(r2, rd, lambda b, d: b ** 2 + (b + d) ** 2)  # r1 = r2 + rd
    rel = torch.minimum(ind, torch.minimum(via1, via2))
    return dict(mse=[m1.mean().item(), m2.mean().item()],
                joint_indep=(ind.mean() / 2).item(), joint_rel=(rel.mean() / 2).item())





def certify_pair(w, x1, x2, eps, delta_total=0.05, c_scale=1.0):
    d = w["Wh"].shape[0] // 2
    d_dim = 1 - (1 - delta_total) ** (1.0 / (2 * d)) 
    c = common.sym_c(d_dim) * c_scale
    out = {}
    res = []
    for x in (x1, x2):
        (ml, mu_), su = encoder_ibp(w, x, eps)
        xi = common.civet_xi(mu_ - ml, su, d_dim).to(D)
        o = decoder_ibp(w, ml - xi, mu_ + xi)
        res.append(mse_ub(o, x))
    out["A_ibp_civetbox"] = res
    mu, dmu, su = encoder_pair(w, x1, x2, eps, use_diff=False)
    xi = [common.civet_xi(m[1] - m[0], s, d_dim).to(D) for m, s in zip(mu, su)]
    zb = [(m[0] - x_, m[1] + x_) for m, x_ in zip(mu, xi)]
    o1, o2, _ = decoder_pair(w, zb[0], zb[1], None, "indep")
    out["B_dp_civetbox"] = [mse_ub(o1, x1), mse_ub(o2, x2)]
    zs = [(m[0] - c * s, m[1] + c * s) for m, s in zip(mu, su)]
    o1, o2, _ = decoder_pair(w, zs[0], zs[1], None, "indep")
    out["C_dp_symbox"] = [mse_ub(o1, x1), mse_ub(o2, x2)]
    dd = torch.maximum((o1[0] - o2[1]).abs(), (o1[1] - o2[0]).abs())
    out["C_mean_absdiff_ub"] = float(dd.mean())
    Co1, Co2 = o1, o2
    mu, dmu, su = encoder_pair(w, x1, x2, eps, use_diff=True)
    z1 = (mu[0][0] - c * su[0], mu[0][1] + c * su[0])
    z2 = (mu[1][0] - c * su[1], mu[1][1] + c * su[1])
    dz = (dmu[0] - c * (su[0] + su[1]), dmu[1] + c * (su[0] + su[1]))
    dz = (torch.maximum(dz[0], z1[0] - z2[1]), torch.minimum(dz[1], z1[1] - z2[0]))
    o1, o2, od = decoder_pair(w, z1, z2, dz, "diff")
    i1, i2, _ = decoder_pair(w, z1, z2, None, "indep")
    o1 = (torch.maximum(o1[0], i1[0]), torch.minimum(o1[1], i1[1]))
    o2 = (torch.maximum(o2[0], i2[0]), torch.minimum(o2[1], i2[1]))
    o1, o2, od = refine(o1, o2, od)
    out["D_relational"] = [mse_ub(o1, x1), mse_ub(o2, x2)]
    for tag, noisy in (("clean", False), ("noisy", True)):
        Cd = (Co1[0] - Co2[1], Co1[1] - Co2[0])
        out[f"J_{tag}_C"] = joint_ub(Co1, Co2, Cd, x1, x2, eps, noisy)
        out[f"J_{tag}_D"] = joint_ub(o1, o2, od, x1, x2, eps, noisy)
    out["D_maxdiff_out"] = float(torch.maximum(od[0].abs(), od[1].abs()).max())
    out["D_mean_absdiff_ub"] = float(torch.maximum(od[0].abs(), od[1].abs()).mean())
    out["c"], out["delta_dim"] = c, d_dim
    out["sig_u_mean"] = [float(su[0].mean()), float(su[1].mean())]
    out["mu_width_mean"] = [float((mu[0][1] - mu[0][0]).mean()), float((mu[1][1] - mu[1][0]).mean())]
    out["dmu_width_mean"] = float((dmu[1] - dmu[0]).mean())
    return out




def attack(model, xs, eps, c, steps=150, R=16, joint=False, seed=0, noisy=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.stack(xs).float().view(len(xs), 1, 28, 28)
    n = x.shape[0]
    d = model.z_size

    def err(v, e): 
        res = []
        for i in range(n):
            vi = v if joint else v[i]
            xin = x[i:i + 1] + vi
            h = model.encoder(xin)
            mean, logvar = model.q_mean(h), model.q_logvar(h)
            z = mean + (0.5 * logvar).exp() * e[i]
            out = model.decoder(model.project(z)).view(-1, 1, 28, 28)
            tgt = x[i:i + 1] + vi if noisy else x[i:i + 1]
            res.append(((out - tgt) ** 2).flatten(1).mean(1))
        return torch.stack(res)

    nv = (R, 1, 28, 28) if joint else (n, R, 1, 28, 28)
    v = (torch.rand(nv, generator=g, device=dev) * 2 - 1) * eps
    e = (torch.rand(n, R, d, generator=g, device=dev) * 2 - 1) * c
    best = torch.zeros(n, device=dev)
    for t in range(steps):
        v.requires_grad_(True)
        e.requires_grad_(True)
        E = err(v, e)
        obj = E.sum() if joint is False else E.sum(0).sum()
        gv, ge = torch.autograd.grad(obj, [v, e])
        with torch.no_grad():
            if joint:
                cur = E.sum(0)
                k = cur.argmax()
                if cur[k] > best.sum():
                    best = E[:, k].detach().clone()
            else:
                best = torch.maximum(best, E.max(1).values)
            a = 1.0 - t / steps * 0.9
            v = (v + a * eps / 5 * gv.sign()).clamp(-eps, eps)
            e = (e + a * c / 5 * ge.sign()).clamp(-c, c)
    with torch.no_grad():
        E = err(v, e)
        if joint:
            cur = E.sum(0)
            k = cur.argmax()
            if cur[k] > best.sum():
                best = E[:, k].clone()
        else:
            best = torch.maximum(best, E.max(1).values)
    return best.tolist()





def make_pairs(X, Y, kind, n, seed):
    rng = torch.Generator().manual_seed(seed)
    pairs = []
    while len(pairs) < n:
        i, j = torch.randint(0, len(Y), (2,), generator=rng).tolist()
        if i == j:
            continue
        if kind == "same" and Y[i] != Y[j]:
            continue
        if kind == "diff" and Y[i] == Y[j]:
            continue
        if kind == "near": 
            cand = (Y == Y[i]).nonzero().flatten()
            cand = cand[cand != i][torch.randperm(len(cand) - 1, generator=rng)[:50].to(dev)]
            j = int(cand[(X[cand] - X[i]).norm(dim=1).argmin()])
        pairs.append((i, j))
    return pairs


def run(model, w, X, pairs, eps, do_attack=True, c_scale=1.0, verbose=True):
    rows = []
    for (i, j) in pairs:
        x1, x2 = X[i].to(D), X[j].to(D)
        r = certify_pair(w, x1, x2, eps, c_scale=c_scale)
        r["pair"], r["eps"], r["l2dist"] = (i, j), eps, float((x1 - x2).norm())
        if do_attack:
            xs = [X[i], X[j]]
            for tag, noisy in (("clean", False), ("noisy", True)):
                each = attack(model, xs, eps, r["c"], noisy=noisy)
                jt = attack(model, xs, eps, r["c"], joint=True, noisy=noisy)
                r[f"atk_{tag}_each"], r[f"atk_{tag}_joint"] = each, jt
                ja = max(sum(each) / 2 * 0, sum(jt) / 2)
                r[f"atk_{tag}_joint_avg"] = ja
                ok = (ja <= r[f"J_{tag}_D"]["joint_rel"] + 1e-9 and
                      all(each[n] <= r[f"J_{tag}_D"]["mse"][n] + 1e-9 for n in (0, 1)))
                r[f"sound_{tag}"] = ok
        rows.append(r)
        if verbose:
            print(f"eps={eps} pair {i},{j} l2={r['l2dist']:.1f} "
                  f"clean: sumIndep={r['J_clean_C']['joint_indep']:.4f} rel={r['J_clean_D']['joint_rel']:.4f}"
                  + (f" atk={r['atk_clean_joint_avg']:.4f} ok={r['sound_clean']}" if do_attack else "")
                  + f" | noisy: indep={r['J_noisy_C']['joint_indep']:.4f} rel={r['J_noisy_D']['joint_rel']:.4f}"
                  + (f" atk={r['atk_noisy_joint_avg']:.4f} ok={r['sound_noisy']}" if do_attack else ""), flush=True)
    return rows


def summarize(rows, eps):
    med = lambda f: float(torch.tensor([f(r) for r in rows]).median())
    d = dict(eps=eps,
             nominal=None,
             per_input_ibp=med(lambda r: sum(r["A_ibp_civetbox"]) / 2),
             per_input_rel=med(lambda r: sum(r["D_relational"]) / 2))
    for tag in ("clean", "noisy"):
        d[f"{tag}_joint_indep"] = med(lambda r: r[f"J_{tag}_C"]["joint_indep"])
        d[f"{tag}_joint_rel"] = med(lambda r: r[f"J_{tag}_D"]["joint_rel"])
        if f"atk_{tag}_joint_avg" in rows[0]:
            d[f"{tag}_atk_joint"] = med(lambda r: r[f"atk_{tag}_joint_avg"])
            d[f"{tag}_atk_each_avg"] = med(lambda r: sum(r[f"atk_{tag}_each"]) / 2)
            d[f"{tag}_sound"] = all(r[f"sound_{tag}"] for r in rows)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1])
    ap.add_argument("--npairs", type=int, default=30)
    ap.add_argument("--kind", choices=["same", "diff", "near"], default="same")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_attack", action="store_true")
    ap.add_argument("--c_scale", type=float, default=1.0, help="DEBUG: scale noise radius (NOT a 95% bound)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model = load_model(a.model)
    w = weights(model)
    ds = common.mnist(False)
    X = torch.stack([x for x, _ in ds]).view(-1, 784).to(dev)
    Y = torch.tensor([y for _, y in ds]).to(dev)
    pairs = make_pairs(X, Y, a.kind, a.npairs, a.seed)
    summ, allrows = [], []
    for eps in a.eps:
        rows = run(model, w, X, pairs, eps, not a.no_attack, a.c_scale)
        summ.append(summarize(rows, eps))
        allrows += rows
    print("==== median over", len(pairs), a.kind, "pairs;", a.model, "====")
    for d in summ:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}))
    if a.out:
        os.makedirs(common.RESULT_DIR, exist_ok=True)
        json.dump(dict(summary=summ, rows=allrows), open(os.path.join(common.RESULT_DIR, a.out), "w"), indent=1)
