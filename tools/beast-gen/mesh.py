"""
Сітка з поля відстаней: marching cubes → спрощення (fqms) → рівні деталізації → нормалі,
ваги скіну (мітки найближчих кісток + дифузія по графу сітки), затінення западин, пакування WHD1.
"""
import json
import os
import struct
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from skimage import measure

# Спрощувач Fast-Quadric-Mesh-Simplification (MIT) з латкою fqms-halfedge.patch; шлях — у змінній FQMS.
FQMS = os.environ.get('FQMS', 'simplify')


def marching(vol, level=0.0):
    v, f, n, _ = measure.marching_cubes(vol.d, level=level, spacing=(vol.h,) * 3, allow_degenerate=False)
    v = v + vol.bmin[None, :]
    # Обхід граней skimage для SDF (від'ємне — всередині) уже дає нормалі назовні.
    return v.astype(np.float64), f.astype(np.int64), n


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def write_obj(path, v, f):
    with open(path, 'w') as fh:
        fh.write(''.join('v %.7f %.7f %.7f\n' % tuple(p) for p in v))
        fh.write(''.join('f %d %d %d\n' % (a + 1, b + 1, c + 1) for a, b, c in f))


def read_obj(path):
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                fs.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(vs, np.float64), np.array(fs, np.int64)


def decimate(v, f, target, halfedge=False, aggr=7.0):
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, 'in.obj')
        b = os.path.join(td, 'out.obj')
        write_obj(a, v, f)
        env = dict(os.environ)
        if halfedge:
            env['FQMS_HALFEDGE'] = '1'
        ratio = min(1.0, target / len(f))
        r = subprocess.run([FQMS, a, b, '%.6f' % ratio, str(aggr)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        return read_obj(b)


def clean(v, f):
    """Прибрати вироджені трикутники й невикористані вершини."""
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[area > 1e-12]
    used = np.unique(f)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def largest_component(v, f):
    n = len(v)
    A = sp.coo_matrix((np.ones(len(f) * 3), (f[:, [0, 1, 2]].ravel(), f[:, [1, 2, 0]].ravel())), shape=(n, n))
    ncomp, lab = csgraph.connected_components(A, directed=False)
    if ncomp == 1:
        return v, f
    sizes = np.bincount(lab)
    keep = sizes.argmax()
    fk = f[lab[f[:, 0]] == keep]
    return clean(v, fk)


def components(v, f):
    n = len(v)
    A = sp.coo_matrix((np.ones(len(f) * 3), (f[:, [0, 1, 2]].ravel(), f[:, [1, 2, 0]].ravel())), shape=(n, n))
    return csgraph.connected_components(A, directed=False)


def lod_subset(v0, v1, f1):
    """Вершини LOD1 (спрощення половинними ребрами) — підмножина вершин LOD0: зіставити індекси."""
    tree = cKDTree(v0)
    d, idx = tree.query(v1)
    if d.max() > 1e-4:
        print('  lod_subset: max snap %.5f m' % d.max())
    f = idx[f1]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return f[ok]


def adjacency(n, f):
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(1.0 / np.maximum(deg, 1))
    return D @ A


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    c = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(P - c, axis=1), t


def skin_weights(v, f, bones, segs, masks, iters=24, alpha=0.5, rigid=None):
    """
    bones — список імен; segs[i] = (a, b, r) — відрізок кістки й радіус впливу;
    masks[i] — булевий масив допустимих вершин (або None).
    Початкові мітки — найближча кістка (відстань/радіус), далі дифузія ваг по графу сітки.
    rigid — словник {кістка: ітерацій дифузії} для жорсткіших ділянок (менше розмиття).
    """
    n = len(v)
    nb = len(bones)
    score = np.full((n, nb), np.inf)
    for i, (a, b, r) in enumerate(segs):
        d, _ = seg_dist(v, np.asarray(a, float), np.asarray(b, float))
        s = d / r
        if masks[i] is not None:
            s = np.where(masks[i], s, np.inf)
        score[:, i] = s
    lab = score.argmin(axis=1)
    W = np.zeros((n, nb))
    W[np.arange(n), lab] = 1.0
    M = adjacency(n, f)
    for _ in range(iters):
        W = (1 - alpha) * W + alpha * (M @ W)
    # Не даємо ваги кісткам, які «замасковані» для вершини (дифузія могла їх занести).
    for i in range(nb):
        if masks[i] is not None:
            W[~masks[i], i] = 0
    W[W < 0.02] = 0
    order = np.argsort(-W, axis=1)[:, :4]
    top = np.take_along_axis(W, order, axis=1)
    top = top / np.maximum(top.sum(axis=1, keepdims=True), 1e-9)
    q = np.floor(top * 255 + 0.5).astype(np.int64)
    # Сума рівно 255: залишок — найбільшій вазі.
    q[:, 0] += 255 - q.sum(axis=1)
    q = np.clip(q, 0, 255)
    order = np.where(q > 0, order, 0)
    return order.astype(np.uint8), q.astype(np.uint8), lab


def sdf_ao(vol, v, n, dists=(0.006, 0.014, 0.028, 0.05), strength=1.0):
    occ = np.zeros(len(v))
    wsum = 0
    for i, d in enumerate(dists):
        s = vol.sample(v + n * d)
        w = 1.0 / (1 + i)
        occ += w * np.clip((d - s) / d, 0, 1)
        wsum += w
    return np.clip(1 - strength * occ / wsum, 0, 1)


def morton_order(v):
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    q = ((v - lo) / (hi - lo + 1e-9) * 1023).astype(np.int64)

    def spread(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    return np.argsort(code, kind='stable')


def reorder(v, lods):
    """Вершини — у порядку першого використання (кривою Мортона), трикутники — за індексами."""
    order = morton_order(v)
    remap = np.empty(len(v), np.int64)
    remap[order] = np.arange(len(v))
    out = []
    for f in lods:
        f = remap[f]
        # Обертаємо трикутник так, щоб найменший індекс був першим (обхід зберігається).
        r = f.argmin(axis=1)
        f = np.stack([f[np.arange(len(f)), r], f[np.arange(len(f)), (r + 1) % 3], f[np.arange(len(f)), (r + 2) % 3]], axis=1)
        f = f[np.lexsort((f[:, 2], f[:, 1], f[:, 0]))]
        out.append(f)
    return order, out


def oct_encode(n):
    n = n / (np.abs(n).sum(axis=1, keepdims=True) + 1e-12)
    x, y, z = n[:, 0].copy(), n[:, 1].copy(), n[:, 2]
    neg = z < 0
    ox = x.copy()
    x[neg] = (1 - np.abs(y[neg])) * np.where(ox[neg] >= 0, 1, -1)
    y[neg] = (1 - np.abs(ox[neg])) * np.where(y[neg] >= 0, 1, -1)
    return np.clip(np.round(np.stack([x, y], axis=1) * 127), -127, 127).astype(np.int8)


def varint_indices(idx):
    out = bytearray()
    prev = 0
    for i in idx.ravel().tolist():
        d = i - prev
        prev = i
        zz = (d << 1) if d >= 0 else ((-d) << 1) - 1
        while True:
            b = zz & 0x7F
            zz >>= 7
            if zz:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def pad4(b):
    return b + b'\x00' * ((4 - len(b) % 4) % 4)


def pack_whd1(header, v, n, rgba, joints, weights, lods):
    nverts = len(v)
    bmin = v.min(axis=0)
    bmax = v.max(axis=0)
    header = dict(header)
    header.update({'v': 1, 'n': nverts, 'bmin': [round(float(x), 6) for x in bmin], 'bmax': [round(float(x), 6) for x in bmax],
                   'lods': [int(len(f)) for f in lods], 'colorGamma': 2.2})
    hj = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    q = np.round((v - bmin) / (bmax - bmin) * 65535).clip(0, 65535).astype('<u2')
    parts = [struct.pack('<II', 0x31444857, len(hj)), pad4(hj), pad4(q.tobytes()), pad4(oct_encode(n).tobytes()),
             rgba.astype(np.uint8).tobytes(), joints.astype(np.uint8).tobytes(), weights.astype(np.uint8).tobytes()]
    for f in lods:
        b = varint_indices(f)
        parts.append(struct.pack('<I', len(b)))
        parts.append(pad4(b))
    return b''.join(parts)
