"""
Конвеєр моделі звіра: скульптинг → сітка → LOD0/LOD1 → ваги → кольори хутра → WHD1 (+ base64).
Вид описують модулі wolf.py, fox.py, hare.py, boar.py: функції bones(), sculpt(vol), paint(...).
"""
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mesh import (adjacency, clean, decimate, largest_component, lod_subset, marching, pack_whd1,  # noqa: E402
                  reorder, sdf_ao, skin_weights, vertex_normals)
from sdf import Vol  # noqa: E402

OUT = os.environ.get('BEAST_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))


# ─── Допоміжні функції фарбування ───

def sm(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def lin(hexv):
    """sRGB hex → лінійний колір (3,)."""
    c = np.array([(hexv >> 16) & 255, (hexv >> 8) & 255, hexv & 255], float) / 255.0
    return c ** 2.2


def mix(a, b, t):
    t = np.asarray(t, float)
    if t.ndim == 1:
        t = t[:, None]
    return a + (b - a) * t


def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


def vnoise(P, freq, seed=0):
    q = np.asarray(P, float) * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    u = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm(P, freq, octaves=3, seed=0):
    s = 0
    a = 1
    tot = 0
    for o in range(octaves):
        s = s + a * vnoise(P, freq * (2.03 ** o), seed + o * 17)
        tot += a
        a *= 0.5
    return s / tot


def stretch_noise(P, freq, axis_scale, seed=0):
    """Видовжений шум (пасма шерсті): масштаб уздовж осей axis_scale = (sx, sy, sz)."""
    return vnoise(np.asarray(P) * np.asarray(axis_scale)[None, :], freq, seed)


# ─── Збирання ───

class Species:
    name = 'beast'
    h = 0.003
    bounds = ((-0.2, 0.0, -0.8), (0.2, 1.0, 0.9))
    tris0 = 14000
    tris1 = 2800
    weight_iters = 24
    extra_header = {}

    def bones(self):
        raise NotImplementedError

    @property
    def bone_names(self):
        return [b[0] for b in self.bones()]

    def sculpt(self, vol):
        raise NotImplementedError

    def displace(self, v, n):
        return None

    def weight_setup(self, v, bones):
        raise NotImplementedError

    def paint(self, v, n, lab, ao, bones):
        raise NotImplementedError


def warp(v, c, sigma, a):
    d = v - np.asarray(c, float)[None, :]
    r = np.linalg.norm(d, axis=1)
    return v + d * (a * np.exp(-(r / sigma) ** 2))[:, None]


def unwarp(v, c, sigma, a):
    c = np.asarray(c, float)
    d = v - c[None, :]
    rp = np.linalg.norm(d, axis=1)
    r = rp.copy()
    for _ in range(30):
        e = np.exp(-(r / sigma) ** 2)
        fval = r * (1 + a * e) - rp
        fd = 1 + a * e * (1 - 2 * r * r / sigma ** 2)
        r = r - fval / fd
    k = np.where(rp > 1e-12, r / np.maximum(rp, 1e-12), 1.0)
    return c[None, :] + d * k[:, None]


def build(sp, out_dir=OUT, verbose=True):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = sp.bounds
    vol = Vol(lo, hi, sp.h)
    sp.sculpt(vol)
    if verbose:
        print('%s: grid %s, sculpt %.1fs' % (sp.name, tuple(vol.n), time.time() - t0))
    v, f, _ = marching(vol)
    v, f = largest_component(v, f)
    if verbose:
        print('  mc: %d verts %d tris (%.1fs)' % (len(v), len(f), time.time() - t0))
    n = vertex_normals(v, f)
    dsp = sp.displace(v, n)
    if dsp is not None:
        v = v + n * dsp[:, None]
    # Збільшувальна «лупа» для голови: спрощувач бачить голову вдвічі більшою й лишає їй більше трикутників.
    focus = getattr(sp, 'focus', None)
    if focus is not None:
        v = warp(v, *focus)
    # Спрощувач працює з порогами похибки в абсолютних одиницях: у сантиметрах (×100) він
    # зберігає більше трикутників на вигинах (голова, лапи), а не розподіляє їх за площею.
    v0, f0 = decimate(v * 100.0, f, sp.tris0)
    v0, f0 = clean(v0 / 100.0, f0)
    if focus is not None:
        v0 = unwarp(v0, *focus)
    v0, f0 = largest_component(v0, f0)
    v1, f1 = decimate(v0 * 100.0, f0, sp.tris1, halfedge=True)
    f1 = lod_subset(v0, v1 / 100.0, f1)
    n0 = vertex_normals(v0, f0)
    if verbose:
        print('  lod0 %d verts %d tris, lod1 %d tris (%.1fs)' % (len(v0), len(f0), len(f1), time.time() - t0))
    bones = sp.bones()
    names = [b[0] for b in bones]
    segs, masks, iters = sp.weight_setup(v0, {b[0]: np.asarray(b[2], float) for b in bones})
    joints, weights, lab = skin_weights(v0, f0, names, [segs[nm] for nm in names], [masks.get(nm) for nm in names],
                                        iters=iters if iters else sp.weight_iters)
    ao = sdf_ao(vol, v0, n0)
    # Згладжене затінення (ділянки, а не окремі вершини).
    M = adjacency(len(v0), f0)
    for _ in range(3):
        ao = 0.5 * ao + 0.5 * (M @ ao)
    rgb, rough = sp.paint(v0, n0, lab, ao, {b[0]: np.asarray(b[2], float) for b in bones})
    srgb = np.clip(rgb, 0, 1) ** (1 / 2.2)
    rgba = np.concatenate([np.round(srgb * 255), np.round(np.clip(rough, 0, 1)[:, None] * 255)], axis=1).astype(np.uint8)
    order, (L0, L1) = reorder(v0, [f0, f1])
    hdr = {'kind': sp.name, 'bones': [{'name': b[0], 'parent': b[1], 'pos': [round(float(x), 4) for x in b[2]]} for b in bones]}
    hdr.update(sp.extra_header)
    blob = pack_whd1(hdr, v0[order], n0[order], rgba[order], joints[order], weights[order], [L0, L1])
    with open(os.path.join(out_dir, sp.name + '.whd'), 'wb') as fh:
        fh.write(blob)
    b64 = base64.b64encode(blob).decode('ascii')
    with open(os.path.join(out_dir, sp.name + '.b64'), 'w') as fh:
        fh.write(b64)
    stats = {'name': sp.name, 'verts': int(len(v0)), 'tris0': int(len(L0)), 'tris1': int(len(L1)), 'bytes': len(blob), 'b64': len(b64),
             'seconds': round(time.time() - t0, 1), 'bmin': v0.min(axis=0).round(3).tolist(), 'bmax': v0.max(axis=0).round(3).tolist()}
    with open(os.path.join(out_dir, sp.name + '.json'), 'w') as fh:
        json.dump(stats, fh)
    if verbose:
        print('  ', stats)
    # Мітки кісток — для перевірки ваг на стенді.
    np.save(os.path.join(out_dir, sp.name + '-lab.npy'), lab[order])
    return stats


def write_bundle(names, path=os.path.join(OUT, 'beast-data.js')):
    parts = []
    for nm in names:
        b64 = open(os.path.join(OUT, nm + '.b64')).read()
        parts.append("  %s: '%s'," % (nm, b64))
    with open(path, 'w') as fh:
        fh.write('window.BEAST_DATA = {\n' + '\n'.join(parts) + '\n};\n')
