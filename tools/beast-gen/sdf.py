"""
Скульптинг полями відстаней (SDF) на регулярній сітці: примітиви обчислюються лише в межах
власної обмежувальної коробки (з запасом), об'єднання — гладкий мінімум.
"""
import numpy as np


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def frame_from(fwd, up_hint=(0, 1, 0)):
    """Ортонормована рамка: стовпці — (x, y, z), де z = fwd, y — найближчий до up_hint."""
    z = _norm(fwd)
    up = np.asarray(up_hint, np.float64)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((1, 0, 0), z)
    x = _norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def _ell_dist(qx, qy, qz, rx, ry, rz):
    """Наближена відстань до еліпсоїда (iq): k0*(k0-1)/k1."""
    k0 = np.sqrt((qx / rx) ** 2 + (qy / ry) ** 2 + (qz / rz) ** 2)
    k1 = np.sqrt((qx / (rx * rx)) ** 2 + (qy / (ry * ry)) ** 2 + (qz / (rz * rz)) ** 2)
    return np.where(k1 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -min(np.min(rx), np.min(ry), np.min(rz)))


class Prim:
    def aabb(self):
        raise NotImplementedError

    def dist(self, X, Y, Z):
        raise NotImplementedError


class Ellipsoid(Prim):
    def __init__(self, c, r, R=None):
        self.c = np.asarray(c, np.float64)
        self.r = np.asarray(r, np.float64)
        self.R = np.eye(3) if R is None else np.asarray(R, np.float64)

    def aabb(self):
        ext = np.sqrt(((self.R * self.r[None, :]) ** 2).sum(axis=1))
        return self.c - ext, self.c + ext

    def dist(self, X, Y, Z):
        px, py, pz = X - self.c[0], Y - self.c[1], Z - self.c[2]
        R = self.R
        qx = px * R[0, 0] + py * R[1, 0] + pz * R[2, 0]
        qy = px * R[0, 1] + py * R[1, 1] + pz * R[2, 1]
        qz = px * R[0, 2] + py * R[1, 2] + pz * R[2, 2]
        return _ell_dist(qx, qy, qz, *self.r)


class RoundCone(Prim):
    """Точна відстань до «круглого конуса» між сферами (a, ra) і (b, rb) — формула iq."""

    def __init__(self, a, b, ra, rb):
        self.a = np.asarray(a, np.float64)
        self.b = np.asarray(b, np.float64)
        self.ra = float(ra)
        self.rb = float(rb)

    def aabb(self):
        r = max(self.ra, self.rb)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def dist(self, X, Y, Z):
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = float(ba @ ba)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pax, pay, paz = X - a[0], Y - a[1], Z - a[2]
        y = pax * ba[0] + pay * ba[1] + paz * ba[2]
        z = y - l2
        x2 = (pax * l2 - ba[0] * y) ** 2 + (pay * l2 - ba[1] * y) ** 2 + (paz * l2 - ba[2] * y) ** 2
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d1 = np.sqrt(x2 + z2) * il2 - r2
        d2 = np.sqrt(x2 + y2) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


class Tube(Prim):
    """
    Трубка вздовж ламаної з еліптичними перерізами (ru — уздовж «бокової» осі, rv — уздовж
    «верхньої»). Кожен відрізок — капсула з еліптичним перерізом; трубка — їх гладке об'єднання.
    lateral — напрям бокової осі (зазвичай x); up обчислюється як перпендикуляр.
    """

    def __init__(self, pts, secs, lateral=(1, 0, 0), k=0.0):
        self.p = np.asarray(pts, np.float64)
        s = np.asarray(secs, np.float64)
        if s.ndim == 1:
            s = np.stack([s, s], axis=1)
        self.s = s
        self.lat0 = np.asarray(lateral, np.float64)
        self.k = k

    def aabb(self):
        r = self.s.max()
        return self.p.min(axis=0) - r, self.p.max(axis=0) + r

    def dist(self, X, Y, Z):
        out = None
        for i in range(len(self.p) - 1):
            A, B = self.p[i], self.p[i + 1]
            T = B - A
            L = np.linalg.norm(T)
            T = T / L
            lat = self.lat0 - T * (self.lat0 @ T)
            if np.linalg.norm(lat) < 1e-6:
                lat = np.cross(T, (0, 0, 1))
            lat = _norm(lat)
            up = np.cross(T, lat)
            pax, pay, paz = X - A[0], Y - A[1], Z - A[2]
            tt = (pax * T[0] + pay * T[1] + paz * T[2]) / L
            t = np.clip(tt, 0.0, 1.0)
            dx = pax - T[0] * t * L
            dy = pay - T[1] * t * L
            dz = paz - T[2] * t * L
            u = dx * lat[0] + dy * lat[1] + dz * lat[2]
            v = dx * up[0] + dy * up[1] + dz * up[2]
            w = dx * T[0] + dy * T[1] + dz * T[2]
            ru = self.s[i, 0] + (self.s[i + 1, 0] - self.s[i, 0]) * t
            rv = self.s[i, 1] + (self.s[i + 1, 1] - self.s[i, 1]) * t
            rw = np.minimum(ru, rv)
            d = _ell_dist(u, v, w, ru, rv, rw)
            # Гладке поєднання сусідніх відрізків «роздуває» місце стику (ковбаски), тому тут — майже жорсткий мінімум.
            out = d if out is None else smin(out, d, min(self.k, 0.003))
        return out


class Smooth(Prim):
    """Гладке поєднання кількох примітивів в одну форму (для вкладених операцій)."""

    def __init__(self, prims, k, mode='union'):
        self.prims = prims
        self.k = k
        self.mode = mode

    def aabb(self):
        los, his = zip(*[p.aabb() for p in self.prims])
        if self.mode == 'inter':
            return np.max(los, axis=0), np.min(his, axis=0)
        return np.min(los, axis=0), np.max(his, axis=0)

    def dist(self, X, Y, Z):
        d = self.prims[0].dist(X, Y, Z)
        for p in self.prims[1:]:
            e = p.dist(X, Y, Z)
            if self.mode == 'union':
                d = smin(d, e, self.k)
            elif self.mode == 'inter':
                d = smax(d, e, self.k)
            else:
                d = smax(d, -e, self.k)
        return d


class HalfSpace(Prim):
    """Півпростір n·(p − p0) ≤ 0 (від'ємне — всередині); коробка задається явно."""

    def __init__(self, p0, n, lo, hi):
        self.p0 = np.asarray(p0, np.float64)
        self.n = _norm(n)
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)

    def aabb(self):
        return self.lo, self.hi

    def dist(self, X, Y, Z):
        n = self.n
        return (X - self.p0[0]) * n[0] + (Y - self.p0[1]) * n[1] + (Z - self.p0[2]) * n[2]


class Vol:
    def __init__(self, bmin, bmax, h, pad=0.05):
        self.h = float(h)
        self.bmin = np.asarray(bmin, np.float64)
        self.n = (np.ceil((np.asarray(bmax, np.float64) - self.bmin) / h).astype(int) + 1)
        self.d = np.full(tuple(self.n), 1.0, np.float32)
        self.ax = [self.bmin[i] + np.arange(self.n[i]) * self.h for i in range(3)]
        self.pad = pad

    def box(self, lo, hi):
        i0 = np.maximum(np.floor((np.asarray(lo) - self.bmin) / self.h).astype(int), 0)
        i1 = np.minimum(np.ceil((np.asarray(hi) - self.bmin) / self.h).astype(int) + 1, self.n)
        if np.any(i1 <= i0):
            return None
        sl = tuple(slice(int(i0[k]), int(i1[k])) for k in range(3))
        X = self.ax[0][sl[0]][:, None, None]
        Y = self.ax[1][sl[1]][None, :, None]
        Z = self.ax[2][sl[2]][None, None, :]
        return sl, X, Y, Z

    def apply(self, prim, mode='union', k=0.0, mirror=False):
        prims = [prim]
        if mirror:
            prims.append(Mirror(prim))
        for p in prims:
            lo, hi = p.aabb()
            pad = k + self.pad
            r = self.box(np.asarray(lo) - pad, np.asarray(hi) + pad)
            if r is None:
                continue
            sl, X, Y, Z = r
            # Обчислення шарами вздовж x — обмежує пам'ять для великих коробок.
            nx = X.shape[0]
            step = max(1, int(4e6 // max(1, Y.shape[1] * Z.shape[2])))
            for s0 in range(0, nx, step):
                s1 = min(nx, s0 + step)
                sub = (slice(sl[0].start + s0, sl[0].start + s1), sl[1], sl[2])
                d = p.dist(X[s0:s1], Y, Z).astype(np.float32)
                cur = self.d[sub]
                if mode == 'union':
                    self.d[sub] = smin(cur, d, k)
                elif mode == 'sub':
                    self.d[sub] = smax(cur, -d, k)
                elif mode == 'inter':
                    self.d[sub] = smax(cur, d, k)

    def add(self, prim, k=0.0, mirror=False):
        self.apply(prim, 'union', k, mirror)

    def sub(self, prim, k=0.0, mirror=False):
        self.apply(prim, 'sub', k, mirror)

    def sample(self, P):
        """Трилінійна вибірка поля в точках P (N×3)."""
        from scipy.ndimage import map_coordinates
        idx = ((np.asarray(P) - self.bmin) / self.h).T
        return map_coordinates(self.d, idx, order=1, mode='nearest')


class Mirror(Prim):
    """Дзеркальна копія примітива відносно площини x = 0."""

    def __init__(self, p):
        self.p = p

    def aabb(self):
        lo, hi = self.p.aabb()
        return np.array([-hi[0], lo[1], lo[2]]), np.array([-lo[0], hi[1], hi[2]])

    def dist(self, X, Y, Z):
        return self.p.dist(-X, Y, Z)
