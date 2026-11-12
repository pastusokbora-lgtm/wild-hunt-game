"""Спільні анатомічні деталі й фарбування для звірів."""
import numpy as np

from beast import lin, mix, sm
from sdf import Ellipsoid, RoundCone, frame_from


def along(a, b, up=(0, 0, 1)):
    """Рамка еліпсоїда з віссю y уздовж відрізка a→b (x — бокова вісь)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    R = frame_from(d, up)
    return np.stack([R[:, 0], R[:, 2], -R[:, 1]], axis=1)


def mirror_bones(B, order):
    """Кістки правого боку — дзеркальні копії лівих (x → −x)."""
    out = []
    for nm in order:
        if nm in B:
            par, p = B[nm]
            out.append((nm, par, tuple(p)))
        else:
            src = nm[:-1] + 'L'
            par, p = B[src]
            if par and par[-1] == 'L':
                par = par[:-1] + 'R'
            out.append((nm, par, (-p[0], p[1], p[2])))
    return out


BONE_ORDER = ['root', 'pelvis', 'chest', 'neck1', 'neck2', 'head', 'tail1', 'tail2',
              'scapFL', 'humFL', 'radFL', 'canFL', 'pasFL', 'femBL', 'tibBL', 'metBL', 'pasBL',
              'scapFR', 'humFR', 'radFR', 'canFR', 'pasFR', 'femBR', 'tibBR', 'metBR', 'pasBR',
              'earL', 'earR', 'tail3', 'jaw']


def canid_paw(vol, c, s, toes=((0.011, 0.04, 0.018), (0.028, 0.03, 0.016)), claws=True):
    """Лапа псових (центр c на рівні підошви, права сторона +x; дзеркалиться): п'ясткова частина, 4 пальці, кігті."""
    c = np.asarray(c, float)
    vol.add(Ellipsoid(c + [0, 0.028 * s, 0.0], (0.033 * s, 0.027 * s, 0.046 * s)), k=0.02 * s, mirror=True)
    out = []
    for dx0, dz, dy in toes:
        for dx in (-dx0, dx0):
            vol.add(Ellipsoid(c + np.array([dx, dy, dz]) * s, (0.0135 * s, 0.016 * s, 0.019 * s)), k=0.008 * s, mirror=True)
            if claws:
                a = c + np.array([dx * 1.05, 0.013, dz + 0.016]) * s
                b = c + np.array([dx * 1.1, 0.003, dz + 0.027]) * s
                vol.add(RoundCone(a, b, 0.0045 * s, 0.002 * s), k=0.004 * s, mirror=True)
                out.append(c + np.array([dx * 1.08, 0.008, dz + 0.022]) * s)
    return out


def claw_mask(v, centers, r):
    m = np.zeros(len(v))
    for cc in centers:
        for sg in (1, -1):
            p = np.array([cc[0] * sg, cc[1], cc[2]])
            m = np.maximum(m, sm(r, r * 0.45, np.linalg.norm(v - p, axis=1)))
    return m


def paint_eyes(v, col, rough, center, r, gaze, iris_hex, iris_dark_hex, pupil=0.95, sclera_hex=0x1c1712, rim_hex=0x1e1a16, rim_w=0.006):
    """Очі: райдужка, зіниця, темна облямівка повік. center/gaze — для лівого ока (−x), дзеркалиться."""
    for sg in (-1, 1):
        c = np.array([abs(center[0]) * sg, center[1], center[2]])
        d = np.linalg.norm(v - c, axis=1)
        eye = sm(r * 1.17, r * 1.02, d)
        g = np.array([abs(gaze[0]) * sg, gaze[1], gaze[2]], float)
        g /= np.linalg.norm(g)
        cosg = ((v - c) @ g) / np.maximum(d, 1e-6)
        iris = mix(lin(iris_hex), lin(iris_dark_hex), sm(0.93, 0.74, cosg))
        ecol = mix(iris, lin(0x080706), sm(pupil, pupil + 0.025, cosg))
        ecol = mix(lin(sclera_hex), ecol, sm(0.55, 0.74, cosg))
        col = mix(col, ecol, eye)
        rim = sm(r + rim_w, r + rim_w * 0.3, d) * (1 - eye)
        col = mix(col, lin(rim_hex), rim * 0.85)
        rough = np.where(eye > 0.5, 0.4, rough)
    return col, rough


def tail_region(v, pts, r):
    """Вершини хвоста: ближче за r до ламаної хвоста й не перед його основою."""
    pts = [np.asarray(p, float) for p in pts]
    best = np.full(len(v), np.inf)
    seg = np.zeros(len(v), int)
    t0 = None
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        tr = ((v - a) @ ab) / (ab @ ab)
        if i == 0:
            t0 = tr
        t = np.clip(tr, 0, 1)
        d = np.linalg.norm(v - (a[None, :] + t[:, None] * ab[None, :]), axis=1)
        upd = d < best
        best = np.where(upd, d, best)
        seg = np.where(upd, i, seg)
    allow = (best < r * 1.15) & ~((seg == 0) & (t0 < -0.12))
    core = (best < r) & ~((seg == 0) & (t0 < 0.2))
    return allow, core


def quad_weights(v, B, s, nose, chin, tail_tip, ear_tip, toe_f, toe_b, mouth_y, jaw_z0, jaw_ymin, ear_ymin, tail_zmax,
                 tail_xmax, low_y, pelvis_end, iters, jaw_xmax=None, tail_r=None):
    """
    Ваги скіну чотириногого звіра (кістки — як у BONE_ORDER). s — масштаб радіусів впливу
    відносно вовка; маски не дають кісткам «перескакувати» на інший бік тіла, хвіст чи щелепу.
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    L, R = x < 0.004 * s, x > -0.004 * s
    jaw_m = (z > jaw_z0) & (y < mouth_y(z)) & (y > jaw_ymin) & (np.abs(x) < (jaw_xmax if jaw_xmax else 0.05 * s))
    A = lambda p: np.asarray(p, float)  # noqa: E731
    segs = {
        'root': (B['root'], B['chest'] * 0.5 + B['root'] * 0.5, 0.16 * s),
        'pelvis': (B['pelvis'], A(pelvis_end), 0.15 * s),
        'chest': (B['chest'], B['neck1'], 0.17 * s),
        'neck1': (B['neck1'], B['neck2'], 0.12 * s),
        'neck2': (B['neck2'], B['head'], 0.1 * s),
        'head': (B['head'] + [0, 0, -0.02 * s], A(nose), 0.075 * s),
        'tail1': (B['tail1'], B['tail2'], 0.06 * s),
        'tail2': (B['tail2'], B['tail3'], 0.07 * s),
        'tail3': (B['tail3'], A(tail_tip), 0.07 * s),
        'earL': (B['earL'], A(ear_tip), 0.025 * s),
        'earR': (B['earR'], A(ear_tip) * [-1, 1, 1], 0.025 * s),
    }
    if 'jaw' in B:
        segs['jaw'] = (B['jaw'], A(chin), 0.03 * s)
    for sd, sg in (('L', 1), ('R', -1)):
        m = np.array([sg, 1, 1])
        segs['scapF' + sd] = (B['scapF' + sd], B['humF' + sd], 0.07 * s)
        segs['humF' + sd] = (B['humF' + sd], B['radF' + sd], 0.06 * s)
        segs['radF' + sd] = (B['radF' + sd], B['canF' + sd], 0.034 * s)
        segs['canF' + sd] = (B['canF' + sd], B['pasF' + sd], 0.025 * s)
        segs['pasF' + sd] = (B['pasF' + sd], A(toe_f) * m, 0.03 * s)
        segs['femB' + sd] = (B['femB' + sd], B['tibB' + sd], 0.08 * s)
        segs['tibB' + sd] = (B['tibB' + sd], B['metB' + sd], 0.045 * s)
        segs['metB' + sd] = (B['metB' + sd], B['pasB' + sd], 0.024 * s)
        segs['pasB' + sd] = (B['pasB' + sd], A(toe_b) * m, 0.03 * s)
    masks = {}
    for nm in segs:
        if nm[-1] in 'LR':
            masks[nm] = L if nm[-1] == 'L' else R
    masks['earL'] = L & (y > ear_ymin)
    masks['earR'] = R & (y > ear_ymin)
    if 'jaw' in segs:
        masks['jaw'] = jaw_m
        masks['head'] = ~jaw_m
    if tail_r:
        reg, core = tail_region(v, [B['tail1'], B['tail2'], B['tail3'], A(tail_tip)], tail_r)
    else:
        reg = core = (z < tail_zmax) & (np.abs(x) < tail_xmax)
    for t in ('tail1', 'tail2', 'tail3'):
        masks[t] = reg
    low = y < low_y
    for nm in ('root', 'pelvis', 'chest', 'neck1', 'neck2', 'head'):
        masks[nm] = masks.get(nm, np.ones(len(v), bool)) & ~low
    if tail_r:
        for nm in list(segs):
            if not nm.startswith('tail'):
                masks[nm] = masks.get(nm, np.ones(len(v), bool)) & ~core
    return segs, masks, iters
