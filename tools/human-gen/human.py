"""
Люди селища: одне тіло в одязі (куртка, штани, черевики) зі скелетом із 22 кісток і набір насадок, що
кріпляться до кістки голови (волосся, кепка, капелюх, шапка, борода, вуса). Колір не «запікається»:
у вершинах зберігається номер ділянки (шкіра, губи, очі, брови, куртка, комір і манжети, штани, черевики,
підошви, пасок, перед і спина торса — під жилет чи фартух, стегна спереду — під фартух, волосся, борода,
капелюх, стрічка, криси), затінення (AO) і шум тканини; гра фарбує кожну людину власною палітрою.

Формат — той самий WHD1, що й у звірів (вершини, нормалі, RGBA = ділянка/AO/шум/шорсткість, кістки й ваги,
два рівні деталізації).
"""
import base64
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('BEAST_GEN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'beast-gen')))
from mesh import (adjacency, clean, decimate, largest_component, lod_subset, marching, pack_whd1,  # noqa: E402
                  reorder, sdf_ao, skin_weights, vertex_normals)
from sdf import Ellipsoid, HalfSpace, RoundCone, Smooth, Vol, rot_x, rot_z  # noqa: E402

OUT = os.environ.get('HUMAN_OUT', os.path.join(HERE, 'out'))

REG = {
    'skin': 0, 'lips': 1, 'sclera': 2, 'iris': 3, 'pupil': 4, 'brows': 5, 'jacket': 6, 'jacket_dark': 7,
    'trousers': 8, 'trousers_dark': 9, 'boots': 10, 'soles': 11, 'belt': 12, 'torso_front': 13, 'torso_back': 14,
    'hair': 15, 'beard': 16, 'hat': 17, 'hat_band': 18, 'hat_brim': 19, 'thigh_front': 20, 'nails': 21,
}

# Скелет у позі «A» (руки трохи відведені): ім'я, батько, положення суглоба (м). Обличчям до +z, ліва сторона — +x.
BONES = [
    ('root', None, (0.0, 0.94, 0.0)),
    ('spine1', 'root', (0.0, 1.04, 0.0)),
    ('spine2', 'spine1', (0.0, 1.18, 0.0)),
    ('chest', 'spine2', (0.0, 1.33, 0.0)),
    ('neck', 'chest', (0.0, 1.49, -0.008)),
    ('head', 'neck', (0.0, 1.60, 0.0)),
    ('clavL', 'chest', (0.045, 1.445, -0.005)),
    ('upperarmL', 'clavL', (0.195, 1.415, -0.012)),
    ('forearmL', 'upperarmL', (0.285, 1.15, -0.03)),
    ('handL', 'forearmL', (0.345, 0.895, -0.005)),
    ('clavR', 'chest', (-0.045, 1.445, -0.005)),
    ('upperarmR', 'clavR', (-0.195, 1.415, -0.012)),
    ('forearmR', 'upperarmR', (-0.285, 1.15, -0.03)),
    ('handR', 'forearmR', (-0.345, 0.895, -0.005)),
    ('thighL', 'root', (0.092, 0.915, 0.0)),
    ('shinL', 'thighL', (0.105, 0.505, 0.012)),
    ('footL', 'shinL', (0.112, 0.085, -0.02)),
    ('toeL', 'footL', (0.114, 0.03, 0.135)),
    ('thighR', 'root', (-0.092, 0.915, 0.0)),
    ('shinR', 'thighR', (-0.105, 0.505, 0.012)),
    ('footR', 'shinR', (-0.112, 0.085, -0.02)),
    ('toeR', 'footR', (-0.114, 0.03, 0.135)),
]
BPOS = {b[0]: np.array(b[2], float) for b in BONES}
NAMES = [b[0] for b in BONES]
HEAD = np.array(BONES[NAMES.index('head')][2], float)


def sm(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


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


def mirror(c):
    return (-c[0], c[1], c[2])


# ─── Тіло ───

def sculpt_body(vol):
    """Чоловік ~1,80 м у мисливській куртці, штанях і черевиках (поза «A»)."""
    k = 0.045
    # Торс: груди, плечовий пояс, живіт, таз, сідниці; куртка трохи «об'ємніша» за тіло.
    torso = [
        Ellipsoid((0, 1.325, 0.006), (0.178, 0.16, 0.118)),
        Ellipsoid((0, 1.405, -0.006), (0.205, 0.075, 0.104)),
        Ellipsoid((0, 1.13, 0.014), (0.158, 0.16, 0.114)),
        Ellipsoid((0, 0.955, -0.004), (0.168, 0.115, 0.118)),
    ]
    for p in torso:
        vol.add(p, k=k)
    vol.add(Ellipsoid((0.072, 0.905, -0.062), (0.086, 0.09, 0.07)), k=0.04, mirror=True)
    # Шия й комір куртки (кільце з капсул навколо шиї).
    vol.add(RoundCone((0, 1.44, -0.012), (0, 1.585, 0.004), 0.06, 0.053), k=0.03)
    ring = []
    for i in range(10):
        a0 = i / 10 * np.pi * 2
        a1 = (i + 1) / 10 * np.pi * 2
        p0 = (np.sin(a0) * 0.072, 1.47 + 0.012 * np.cos(a0), np.cos(a0) * 0.07 - 0.012)
        p1 = (np.sin(a1) * 0.072, 1.47 + 0.012 * np.cos(a1), np.cos(a1) * 0.07 - 0.012)
        ring.append(RoundCone(p0, p1, 0.02, 0.02))
    vol.add(Smooth(ring, 0.01), k=0.02)
    # Голова: череп, обличчя й щелепа, підборіддя, вилиці, надбрівні дуги, ніс, губи, вуха, очі.
    vol.add(Ellipsoid((0, 1.70, -0.006), (0.077, 0.093, 0.098)), k=0.03)
    vol.add(Ellipsoid((0, 1.636, 0.026), (0.066, 0.075, 0.074)), k=0.03)
    vol.add(Ellipsoid((0, 1.588, 0.062), (0.03, 0.024, 0.024)), k=0.02)
    vol.add(Ellipsoid((0.041, 1.664, 0.056), (0.03, 0.021, 0.03)), k=0.018, mirror=True)
    vol.add(Ellipsoid((0, 1.703, 0.073), (0.06, 0.016, 0.024)), k=0.016)
    vol.add(RoundCone((0, 1.69, 0.086), (0, 1.646, 0.109), 0.0105, 0.0135), k=0.012)
    vol.add(Ellipsoid((0.0135, 1.644, 0.098), (0.0095, 0.008, 0.009)), k=0.008, mirror=True)
    vol.add(Ellipsoid((0, 1.622, 0.088), (0.022, 0.0068, 0.0105)), k=0.006)
    vol.add(Ellipsoid((0, 1.6115, 0.086), (0.0195, 0.0078, 0.0105)), k=0.006)
    vol.sub(Ellipsoid((0, 1.6168, 0.094), (0.02, 0.0018, 0.006)), k=0.002)
    vol.sub(Ellipsoid((0.032, 1.679, 0.083), (0.017, 0.0135, 0.015)), k=0.008, mirror=True)
    vol.add(Ellipsoid((0.032, 1.677, 0.0745), (0.0125, 0.0125, 0.0125)), k=0.0)
    vol.add(Ellipsoid((-0.032, 1.677, 0.0745), (0.0125, 0.0125, 0.0125)), k=0.0)
    vol.add(Ellipsoid((0.078, 1.668, -0.004), (0.011, 0.03, 0.018), rot_z(-0.12)), k=0.008, mirror=True)
    # Руки (рукави куртки, манжети, зап'ястя, кисті з великим пальцем).
    for s in (1, -1):
        sh = np.array((0.195 * s, 1.415, -0.012))
        el = np.array((0.285 * s, 1.15, -0.03))
        wr = np.array((0.345 * s, 0.895, -0.005))
        vol.add(Ellipsoid((0.188 * s, 1.43, -0.01), (0.066, 0.07, 0.072)), k=0.035)
        vol.add(RoundCone(sh, el, 0.058, 0.047), k=0.02)
        vol.add(RoundCone(el, wr + (wr - el) * -0.06, 0.047, 0.04), k=0.012)
        vol.add(RoundCone(wr + (wr - el) * -0.1, wr + (wr - el) * -0.01, 0.043, 0.043), k=0.004)
        d = (wr - el) / np.linalg.norm(wr - el)
        wrist_end = wr + d * 0.03
        vol.add(RoundCone(wr, wrist_end, 0.026, 0.025), k=0.008)
        palm_c = wr + d * 0.085
        a2 = np.array((0, 0, 1.0)) - d * d[2]
        a2 /= np.linalg.norm(a2)
        a3 = np.cross(d, a2)
        vol.add(Ellipsoid(palm_c, (0.044, 0.05, 0.019), np.stack([a2, d, a3], axis=1)), k=0.012)
        base = wr + d * 0.13
        for j, zo in enumerate((0.03, 0.01, -0.01, -0.028)):
            L = (0.075, 0.085, 0.08, 0.065)[j]
            a = base + np.array((0, 0, zo))
            b = a + d * L * 0.62 + np.array((-0.012 * s, 0, 0.006))
            c = b + d * L * 0.38 + np.array((-0.012 * s, 0, 0.004))
            vol.add(RoundCone(a, b, 0.0098, 0.0088), k=0.006)
            vol.add(RoundCone(b, c, 0.0088, 0.0078), k=0.004)
        t0 = wr + d * 0.05 + np.array((-0.006 * s, 0, 0.033))
        t1 = t0 + d * 0.04 + np.array((-0.01 * s, 0, 0.022))
        t2 = t1 + d * 0.03 + np.array((-0.008 * s, 0, 0.008))
        vol.add(RoundCone(t0, t1, 0.0125, 0.011), k=0.008)
        vol.add(RoundCone(t1, t2, 0.011, 0.0095), k=0.004)
    # Ноги: стегна, коліна, литки, халяви черевиків, стопи з плоскою підошвою.
    for s in (1, -1):
        hip = np.array((0.092 * s, 0.915, 0.0))
        knee = np.array((0.105 * s, 0.505, 0.012))
        ank = np.array((0.11 * s, 0.13, -0.008))
        vol.add(RoundCone(hip, knee, 0.088, 0.062), k=0.03)
        vol.add(Ellipsoid(knee + np.array((0, 0, 0.012)), (0.058, 0.06, 0.058)), k=0.02)
        vol.add(RoundCone(knee, ank, 0.06, 0.046), k=0.02)
        vol.add(Ellipsoid((0.107 * s, 0.37, -0.022), (0.052, 0.1, 0.052)), k=0.03)
        vol.add(RoundCone((0.111 * s, 0.27, -0.006), (0.112 * s, 0.075, -0.004), 0.059, 0.057), k=0.012)
        vol.add(RoundCone((0.112 * s, 0.065, -0.045), (0.113 * s, 0.05, 0.17), 0.052, 0.043), k=0.02)
        vol.add(Ellipsoid((0.112 * s, 0.018, 0.058), (0.057, 0.022, 0.152)), k=0.012)
    # Плоска підошва.
    vol.sub(HalfSpace((0, 0.0, 0), (0, 1, 0), (-0.3, -0.1, -0.2), (0.3, 0.0, 0.3)), k=0.0)
    vol.sub(HalfSpace((0, -0.003, 0), (0, -1, 0), (-0.3, -0.1, -0.2), (0.3, -0.003, 0.3)), k=0.0)


def body_regions(v, n):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)
    r = np.full(len(v), REG['jacket'], np.int32)
    # Голова й шия — шкіра (вище коміра), кисті — шкіра (за манжетами).
    head = (y > 1.50) & (ax < 0.12)
    r[head] = REG['skin']
    collar = (y > 1.445) & (y <= 1.505) & (ax < 0.1)
    r[collar] = REG['jacket_dark']
    # Кисті: далі за манжету вздовж руки.
    for s in (1, -1):
        el = np.array((0.285 * s, 1.15, -0.03))
        wr = np.array((0.345 * s, 0.895, -0.005))
        d = (wr - el) / np.linalg.norm(wr - el)
        t = (v - el) @ d
        side = (x * s > 0.2)
        hand = side & (t > np.linalg.norm(wr - el) + 0.004)
        cuff = side & (t > np.linalg.norm(wr - el) - 0.03) & ~hand
        r[cuff] = REG['jacket_dark']
        r[hand] = REG['skin']
    # Торс: перед і спина (під жилет або фартух), блискавка, кишені, пасок.
    torso = (y > 0.985) & (y < 1.445) & (ax < 0.19)
    r[torso & (z > -0.005)] = REG['torso_front']
    r[torso & (z <= -0.005)] = REG['torso_back']
    zip_ = torso & (ax < 0.008) & (z > 0.05)
    r[zip_] = REG['jacket_dark']
    for s in (1, -1):
        flap = torso & (np.abs(x - 0.085 * s) < 0.045) & (np.abs(y - 1.355) < 0.012) & (z > 0.06)
        r[flap] = REG['jacket_dark']
        hp = (np.abs(x - 0.11 * s) < 0.05) & (np.abs(y - 1.075) < 0.01) & (z > 0.06) & (y > 0.985)
        r[hp] = REG['jacket_dark']
    # Низ куртки, пасок, штани, наколінки, черевики, підошви.
    belt = (y > 0.94) & (y <= 0.985) & (ax < 0.2)
    r[belt] = REG['belt']
    legs = (y <= 0.94)
    r[legs] = REG['trousers']
    thigh_front = legs & (y > 0.53) & (z > 0.0)
    r[thigh_front] = REG['thigh_front']
    knee = legs & (y > 0.44) & (y < 0.56) & (z > 0.035)
    r[knee] = REG['trousers_dark']
    boots = (y < 0.255)
    r[boots] = REG['boots']
    soles = (y < 0.03)
    r[soles] = REG['soles']
    # Обличчя: губи, очі (білок, райдужка, зіниця), брови.
    lips = (np.abs(x) < 0.023) & (y > 1.604) & (y < 1.629) & (z > 0.08)
    r[lips] = REG['lips']
    for s in (1, -1):
        ec = np.array((0.032 * s, 1.677, 0.0745))
        de = np.linalg.norm(v - ec, axis=1)
        eye = de < 0.0135
        r[eye] = REG['sclera']
        dirz = (v - ec) / np.maximum(de[:, None], 1e-9)
        gaze = np.array((0.04 * s * -1, -0.02, 1.0))
        gaze /= np.linalg.norm(gaze)
        cosg = dirz @ gaze
        r[eye & (cosg > 0.82)] = REG['iris']
        r[eye & (cosg > 0.95)] = REG['pupil']
        brow = (np.abs(x - 0.033 * s) < 0.022) & (y > 1.693) & (y < 1.708) & (z > 0.068) & (np.abs(x) > 0.008)
        r[brow] = REG['brows']
    return r


def body_weights(v, f):
    B = BPOS
    segs = {
        'root': (B['root'] + (0, -0.06, 0), B['root'] + (0, 0.05, 0), 0.16),
        'spine1': (B['spine1'], B['spine2'], 0.15),
        'spine2': (B['spine2'], B['chest'], 0.16),
        'chest': (B['chest'], B['chest'] + (0, 0.12, 0), 0.18),
        'neck': (B['neck'], B['head'], 0.07),
        'head': (B['head'], B['head'] + (0, 0.2, 0.02), 0.11),
    }
    for s, L in ((1, 'L'), (-1, 'R')):
        segs['clav' + L] = (B['clav' + L], B['upperarm' + L], 0.09)
        segs['upperarm' + L] = (B['upperarm' + L], B['forearm' + L], 0.07)
        segs['forearm' + L] = (B['forearm' + L], B['hand' + L], 0.055)
        d = (B['hand' + L] - B['forearm' + L])
        d = d / np.linalg.norm(d)
        segs['hand' + L] = (B['hand' + L], B['hand' + L] + d * 0.19, 0.06)
        segs['thigh' + L] = (B['thigh' + L], B['shin' + L], 0.1)
        segs['shin' + L] = (B['shin' + L], B['foot' + L], 0.07)
        segs['foot' + L] = (B['foot' + L], B['toe' + L], 0.07)
        segs['toe' + L] = (B['toe' + L], B['toe' + L] + (0, 0, 0.06), 0.06)
    x, y = v[:, 0], v[:, 1]
    masks = {}
    for s, L in ((1, 'L'), (-1, 'R')):
        arm = (x * s > 0.13) & (y > 0.6) & (y < 1.5)
        masks['clav' + L] = (x * s > 0.02) & (y > 1.33)
        masks['upperarm' + L] = arm & (x * s > 0.15)
        masks['forearm' + L] = arm & (x * s > 0.2) & (y < 1.25)
        masks['hand' + L] = arm & (x * s > 0.27) & (y < 0.98)
        leg = (x * s > -0.01) & (y < 1.0)
        masks['thigh' + L] = leg
        masks['shin' + L] = leg & (y < 0.62)
        masks['foot' + L] = leg & (y < 0.2)
        masks['toe' + L] = leg & (y < 0.12) & (v[:, 2] > 0.07)
    masks['head'] = y > 1.55
    masks['neck'] = (y > 1.42) & (y < 1.66)
    masks['chest'] = (y > 1.18) & (np.abs(x) < 0.26)
    masks['spine2'] = (y > 1.0) & (y < 1.42) & (np.abs(x) < 0.24)
    masks['spine1'] = (y > 0.86) & (y < 1.3) & (np.abs(x) < 0.22)
    masks['root'] = (y > 0.72) & (y < 1.12)
    return [segs[nm] for nm in NAMES], [masks.get(nm) for nm in NAMES]


# ─── Насадки на голову (жорстко на кістці голови) ───

class Field:
    """Поле відстаней голови (для «оболонок» волосся й бороди)."""

    def __init__(self):
        self.vol = Vol((-0.12, 1.5, -0.14), (0.12, 1.85, 0.16), 0.003)
        v = self.vol
        v.add(Ellipsoid((0, 1.70, -0.006), (0.077, 0.093, 0.098)), k=0.03)
        v.add(Ellipsoid((0, 1.636, 0.026), (0.066, 0.075, 0.074)), k=0.03)
        v.add(Ellipsoid((0, 1.588, 0.062), (0.03, 0.024, 0.024)), k=0.02)
        v.add(Ellipsoid((0.041, 1.664, 0.056), (0.03, 0.021, 0.03)), k=0.018, mirror=True)


def accessory(name, build, h=0.0025, bounds=((-0.18, 1.5, -0.2), (0.18, 1.9, 0.24)), tris=(1400, 420)):
    vol = Vol(bounds[0], bounds[1], h)
    regions = build(vol)
    v, f, _ = marching(vol)
    v, f = largest_component(v, f) if name not in ('beard_full', 'beard_short') else (v, f)
    v0, f0 = decimate(v * 100.0, f, tris[0])
    v0, f0 = clean(v0 / 100.0, f0)
    v1, f1 = decimate(v0 * 100.0, f0, tris[1], halfedge=True)
    f1 = lod_subset(v0, v1 / 100.0, f1)
    n0 = vertex_normals(v0, f0)
    reg = regions(v0, n0)
    ao = sdf_ao(vol, v0, n0, dists=(0.004, 0.009, 0.018))
    return v0, f0, f1, n0, reg, ao


def build_accessories():
    F = Field()

    def shell(vol, region_prim, t0, t1, k=0.004):
        """Оболонка над шкірою голови: від t0 до t1 над поверхнею, у межах region_prim."""
        X = vol.ax[0][:, None, None]
        Y = vol.ax[1][None, :, None]
        Z = vol.ax[2][None, None, :]
        P = np.stack(np.broadcast_arrays(X, Y, Z), axis=-1).reshape(-1, 3)
        dh = F.vol.sample(P).reshape(vol.d.shape)
        outer = dh - t1
        inner = t0 - dh
        d = np.maximum(outer, inner)
        dr = region_prim(X, Y, Z)
        vol.d = np.maximum(d, dr).astype(np.float32)

    out = {}

    def hair_short(vol):
        def region(X, Y, Z):
            line = 1.6925 + 0.656 * Z - Y
            ears = np.maximum(np.abs(X) - 0.056, 1.692 - Y)
            return np.maximum(line, -ears)
        shell(vol, region, 0.002, 0.011)
        return lambda v, n: np.full(len(v), REG['hair'], np.int32)

    def cap(vol):
        vol.add(Ellipsoid((0, 1.725, -0.005), (0.088, 0.079, 0.107)))
        vol.sub(HalfSpace((0, 1.694, 0), (0, 1, 0), (-0.2, 1.5, -0.2), (0.2, 1.694, 0.2)))
        brim = Ellipsoid((0, 1.70, 0.075), (0.074, 0.006, 0.072), rot_x(-0.14))
        vol.add(Smooth([brim, HalfSpace((0, 0, 0.035), (0, 0, -1), (-0.2, 1.6, 0.035), (0.2, 1.8, 0.2))], 0.0, 'inter'), k=0.004)
        vol.sub(Ellipsoid((0, 1.70, -0.006), (0.076, 0.092, 0.097)), k=0.0)
        return lambda v, n: np.where((v[:, 2] > 0.06) & (v[:, 1] < 1.715), REG['hat_brim'], np.where(v[:, 1] < 1.708, REG['hat_band'], REG['hat']))

    def felt(vol):
        crown = Ellipsoid((0, 1.735, -0.005), (0.09, 0.1, 0.108))
        vol.add(crown)
        vol.sub(HalfSpace((0, 1.705, 0), (0, 1, 0), (-0.2, 1.5, -0.2), (0.2, 1.705, 0.2)))
        vol.sub(Ellipsoid((0, 1.84, -0.004), (0.012, 0.03, 0.07)), k=0.01)
        brim = Ellipsoid((0, 1.712, -0.002), (0.15, 0.007, 0.155))
        vol.add(brim, k=0.006)
        vol.add(Ellipsoid((0, 1.716, 0.0), (0.142, 0.012, 0.012), rot_x(0.0)), k=0.004)
        vol.sub(Ellipsoid((0, 1.70, -0.006), (0.076, 0.092, 0.097)), k=0.0)
        return lambda v, n: np.where(np.hypot(v[:, 0] / 0.09, v[:, 2] / 0.1) > 1.02, REG['hat_brim'], np.where((v[:, 1] < 1.738) & (v[:, 1] > 1.712), REG['hat_band'], REG['hat']))

    def beanie(vol):
        vol.add(Ellipsoid((0, 1.725, -0.009), (0.096, 0.09, 0.11)))
        vol.sub(HalfSpace((0, 1.672, 0), (0, 1, 0), (-0.2, 1.5, -0.2), (0.2, 1.672, 0.2)))
        vol.add(Smooth([Ellipsoid((0, 1.70, -0.009), (0.1, 0.028, 0.116)), HalfSpace((0, 1.674, 0), (0, -1, 0), (-0.2, 1.6, -0.2), (0.2, 1.75, 0.2))], 0.0, 'inter'), k=0.004)
        vol.sub(Ellipsoid((0, 1.70, -0.006), (0.076, 0.092, 0.097)), k=0.0)
        return lambda v, n: np.where(v[:, 1] < 1.722, REG['hat_band'], REG['hat'])

    def beard(t, mask_y):
        def bld(vol):
            def region(X, Y, Z):
                below = Y - mask_y
                front = -0.012 - Z
                sides = np.abs(X) - 0.074
                mouth = 0.012 - np.sqrt((X / 0.024) ** 2 + ((Y - 1.617) / 0.0115) ** 2 + 0 * Z) * 0.012
                return np.maximum(np.maximum(below, front), np.maximum(sides, mouth))
            shell(vol, region, 0.0, t)
            return lambda v, n: np.full(len(v), REG['beard'], np.int32)
        return bld

    def mustache(vol):
        vol.add(Ellipsoid((0, 1.629, 0.091), (0.026, 0.0065, 0.009), rot_x(0.2)))
        vol.add(Ellipsoid((0.022, 1.622, 0.086), (0.01, 0.009, 0.007)), mirror=True)
        return lambda v, n: np.full(len(v), REG['beard'], np.int32)

    specs = {
        'hair_short': (hair_short, (1600, 500)),
        'cap': (cap, (1300, 420)),
        'felt': (felt, (1700, 520)),
        'beanie': (beanie, (1300, 420)),
        'beard_full': (beard(0.016, 1.655), (1300, 420)),
        'beard_short': (beard(0.006, 1.645), (1000, 340)),
        'mustache': (mustache, (300, 120)),
    }
    for name, (fn, tris) in specs.items():
        t0 = time.time()
        res = accessory(name, fn, tris=tris)
        out[name] = res
        print('  %s: %d verts, %d/%d tris (%.1fs)' % (name, len(res[0]), len(res[1]), len(res[2]), time.time() - t0))
    return out


def pack(name, v, f0, f1, n, reg, ao, joints, weights, extra=None):
    noise = vnoise(v, 180.0, 5) * 0.6 + vnoise(v, 520.0, 9) * 0.4
    rough = np.full(len(v), 0.86)
    rgba = np.stack([reg.astype(float), np.round(np.clip(ao, 0, 1) * 255), np.round(np.clip(noise, 0, 1) * 255), np.round(rough * 255)], axis=1)
    order, (L0, L1) = reorder(v, [f0, f1])
    hdr = {'kind': name, 'regions': REG, 'bones': [{'name': b[0], 'parent': b[1], 'pos': [round(float(x), 4) for x in b[2]]} for b in BONES]}
    if extra:
        hdr.update(extra)
    blob = pack_whd1(hdr, v[order], n[order], rgba[order].astype(np.uint8), joints[order], weights[order], [L0, L1])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.whd'), 'wb') as fh:
        fh.write(blob)
    with open(os.path.join(OUT, name + '.b64'), 'w') as fh:
        fh.write(base64.b64encode(blob).decode('ascii'))
    return {'name': name, 'verts': int(len(v)), 'tris0': int(len(L0)), 'tris1': int(len(L1)), 'bytes': len(blob)}


def build_body():
    t0 = time.time()
    vol = Vol((-0.46, -0.01, -0.2), (0.46, 1.82, 0.26), 0.0036)
    sculpt_body(vol)
    print('body: grid %s, sculpt %.1fs' % (tuple(vol.n), time.time() - t0))
    v, f, _ = marching(vol)
    v, f = largest_component(v, f)
    print('  mc %d verts %d tris (%.1fs)' % (len(v), len(f), time.time() - t0))
    # «Лупа» на голову й кисті: більше трикутників на обличчі.
    from beast import unwarp, warp
    foci = [((0, 1.66, 0.04), 0.09, 0.9)]
    for c, sg, a in foci:
        v = warp(v, c, sg, a)
    v0, f0 = decimate(v * 100.0, f, 10500)
    v0, f0 = clean(v0 / 100.0, f0)
    for c, sg, a in reversed(foci):
        v0 = unwarp(v0, c, sg, a)
    v0, f0 = largest_component(v0, f0)
    v1, f1 = decimate(v0 * 100.0, f0, 2400, halfedge=True)
    f1 = lod_subset(v0, v1 / 100.0, f1)
    n0 = vertex_normals(v0, f0)
    print('  lod0 %d verts %d tris, lod1 %d tris (%.1fs)' % (len(v0), len(f0), len(f1), time.time() - t0))
    segs, masks = body_weights(v0, f0)
    joints, weights, lab = skin_weights(v0, f0, NAMES, segs, masks, iters=14)
    ao = sdf_ao(vol, v0, n0)
    M = adjacency(len(v0), f0)
    for _ in range(3):
        ao = 0.5 * ao + 0.5 * (M @ ao)
    reg = body_regions(v0, n0)
    st = pack('human_body', v0, f0, f1, n0, reg, ao, joints, weights, {'height': round(float(v0[:, 1].max()), 3)})
    np.save(os.path.join(OUT, 'human_body-lab.npy'), lab)
    print('  ', st, '%.1fs' % (time.time() - t0))
    return st


def build_all():
    stats = [build_body()]
    acc = build_accessories()
    hi = NAMES.index('head')
    for name, (v0, f0, f1, n0, reg, ao) in acc.items():
        joints = np.zeros((len(v0), 4), np.uint8)
        joints[:, 0] = hi
        weights = np.zeros((len(v0), 4), np.uint8)
        weights[:, 0] = 255
        stats.append(pack('human_' + name, v0, f0, f1, n0, reg, ao, joints, weights))
    with open(os.path.join(OUT, 'stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def emit_js(dst):
    names = ['human_body', 'human_hair_short', 'human_cap', 'human_felt', 'human_beanie', 'human_beard_full', 'human_beard_short', 'human_mustache']
    parts = []
    for nm in names:
        b64 = open(os.path.join(OUT, nm + '.b64')).read()
        parts.append("  %s: '%s'," % (nm.replace('human_', ''), b64))
    with open(dst, 'w') as fh:
        fh.write('// Моделі людей селища (згенеровано tools/human-gen/human.py): тіло й насадки на голову, WHD1 у base64.\n')
        fh.write('const HUMAN_MODEL_DATA = {\n' + '\n'.join(parts) + '\n};\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'emit':
        emit_js(sys.argv[2])
    else:
        build_all()
