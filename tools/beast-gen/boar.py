"""Дикий кабан (Sus scrofa), дорослий самець ≈ 110 кг: висота в холці ≈ 0,85 м, тулуб ≈ 1,45 м із головою."""
import numpy as np

from anat import BONE_ORDER, along, mirror_bones, paint_eyes, quad_weights
from beast import Species, fbm, lin, mix, sm, stretch_noise, vnoise
from sdf import Ellipsoid as E, HalfSpace, Tube as T, frame_from


class Boar(Species):
    name = 'boar'
    h = 0.0035
    bounds = ((-0.22, -0.004, -0.67), (0.22, 0.95, 0.95))
    tris0 = 13000
    tris1 = 2600
    weight_iters = 22
    focus = ((0, 0.5, 0.72), 0.14, 1.3)

    B = {
        'root': (None, (0, 0.62, 0.0)),
        'pelvis': ('root', (0, 0.62, -0.35)),
        'chest': ('root', (0, 0.62, 0.28)),
        'neck1': ('chest', (0, 0.64, 0.42)),
        'neck2': ('neck1', (0, 0.625, 0.5)),
        'head': ('neck2', (0, 0.605, 0.56)),
        'jaw': ('head', (0, 0.5, 0.62)),
        'tail1': ('pelvis', (0, 0.665, -0.535)),
        'tail2': ('tail1', (0, 0.6, -0.585)),
        'tail3': ('tail2', (0, 0.5, -0.605)),
        'scapFL': ('chest', (-0.1, 0.78, 0.3)),
        'humFL': ('scapFL', (-0.13, 0.52, 0.36)),
        'radFL': ('humFL', (-0.122, 0.37, 0.26)),
        'canFL': ('radFL', (-0.104, 0.17, 0.28)),
        'pasFL': ('canFL', (-0.098, 0.072, 0.292)),
        'femBL': ('pelvis', (-0.1, 0.56, -0.38)),
        'tibBL': ('femBL', (-0.112, 0.4, -0.3)),
        'metBL': ('tibBL', (-0.102, 0.2, -0.44)),
        'pasBL': ('metBL', (-0.097, 0.072, -0.425)),
        'earL': ('head', (-0.066, 0.682, 0.552)),
    }
    TAIL_TIP = (0, 0.41, -0.612)
    NOSE = (0, 0.352, 0.882)
    CHIN = (0, 0.33, 0.8)
    TOE_F = (-0.097, 0.0, 0.338)
    TOE_B = (-0.097, 0.0, -0.375)
    EAR_TIP = (-0.126, 0.81, 0.5)
    EYE = (0.07, 0.585, 0.652)
    HOOF_F = (0.097, 0.0, 0.312)
    HOOF_B = (0.097, 0.0, -0.4)

    def __init__(self):
        hb = self.B['head'][1]
        self.extra_header = {
            'headPitch': round(float(np.arctan2(hb[1] - self.NOSE[1], self.NOSE[2] - hb[2])), 4),
            'nose': list(self.NOSE),
            'footFwd': [0.045, 0.045],
            'tailTip': list(self.TAIL_TIP),
            'eye': list(self.EYE),
            'tusk': [0.045, 0.43, 0.76],
        }
        self.hooves = []

    def bones(self):
        return mirror_bones(self.B, BONE_ORDER)

    def mouth_y(self, z):
        return 0.478 - np.clip((z - 0.655) / 0.17, 0, 1.3) * 0.122

    def _hoof(self, vol, c):
        c = np.asarray(c, float)
        for dx in (-0.0135, 0.0135):
            vol.add(E(c + [dx, 0.024, 0.012], (0.0118, 0.036, 0.017), along(c + [dx, 0.056, -0.022], c + [dx, 0.0, 0.042])), k=0.006, mirror=True)
            vol.add(E(c + [dx * 1.5, 0.052, -0.036], (0.0072, 0.012, 0.009)), k=0.006, mirror=True)
            self.hooves.append(c + [dx, 0.022, 0.012])

    def sculpt(self, vol):
        add, sub = vol.add, vol.sub
        P = lambda nm: np.asarray(self.B[nm][1], float)  # noqa: E731
        M = np.array([-1, 1, 1])
        # Масивна передня частина, клиноподібний тулуб, вужчий зад.
        add(E((0, 0.6, 0.17), (0.168, 0.245, 0.3)))
        add(E((0, 0.72, 0.22), (0.13, 0.125, 0.2)), k=0.08)
        add(E((0, 0.6, -0.12), (0.152, 0.205, 0.27)), k=0.1)
        add(E((0, 0.6, -0.36), (0.13, 0.15, 0.175)), k=0.08)
        add(E((0.072, 0.53, -0.375), (0.07, 0.14, 0.11), along(P('femBL') * M, P('tibBL') * M)), k=0.06, mirror=True)
        add(E((0, 0.455, 0.02), (0.128, 0.08, 0.3)), k=0.08)
        add(E((0.1, 0.6, 0.29), (0.07, 0.17, 0.12), along(P('scapFL') * M, P('humFL') * M)), k=0.06, mirror=True)
        add(T([(0, 0.79, 0.36), (0, 0.8, 0.22), (0, 0.79, 0.05), (0, 0.77, -0.15), (0, 0.745, -0.35), (0, 0.69, -0.5)],
              [(0.1, 0.06), (0.11, 0.065), (0.11, 0.062), (0.1, 0.06), (0.095, 0.056), (0.06, 0.04)], k=0.05), k=0.08)
        # Коротка товста шия, що переходить у голову.
        add(E((0, 0.6, 0.42), (0.148, 0.2, 0.15)), k=0.08)
        # Голова-клин: прямий профіль від лоба до п'ятачка, масивні щелепні м'язи, нижня щелепа.
        from sdf import RoundCone
        add(RoundCone((0, 0.565, 0.57), (0, 0.375, 0.845), 0.112, 0.04), k=0.06)
        hd = along((0, 0.66, 0.5), (0, 0.36, 0.88))
        add(E((0, 0.655, 0.6), (0.072, 0.11, 0.05), hd), k=0.05)
        add(E((0.064, 0.49, 0.6), (0.058, 0.085, 0.1), hd), k=0.05, mirror=True)
        add(E((0.052, 0.6, 0.64), (0.034, 0.045, 0.03), hd), k=0.03, mirror=True)
        disc = frame_from((0, -0.62, 0.79))
        add(E((0, 0.36, 0.866), (0.043, 0.042, 0.018), disc), k=0.012)
        sub(E((0.016, 0.352, 0.882), (0.0085, 0.011, 0.012), disc), k=0.004, mirror=True)
        add(T([(0, 0.45, 0.58), (0, 0.39, 0.7), (0, 0.338, 0.79)], [(0.06, 0.045), (0.042, 0.03), (0.026, 0.018)], k=0.02), k=0.03)
        sub(T([(0.058, 0.478, 0.655), (0.045, 0.43, 0.73), (0.031, 0.375, 0.81)], [(0.006, 0.0045), (0.006, 0.004), (0.005, 0.0035)]), k=0.008, mirror=True)
        ex, ey, ez = self.EYE
        sub(E((ex, ey, ez), (0.0135, 0.0125, 0.013)), k=0.005, mirror=True)
        add(E((ex, ey, ez), (0.0105, 0.0105, 0.0105)), k=0.002, mirror=True)
        add(E((ex - 0.006, ey + 0.016, ez - 0.004), (0.018, 0.008, 0.016)), k=0.01, mirror=True)
        # Вуха: стоячі, трикутні, щетинисті.
        eb, et = P('earL') * M, np.asarray(self.EAR_TIP) * M
        add(T([eb, eb + (et - eb) * 0.5, et], [(0.05, 0.015), (0.033, 0.012), (0.006, 0.005)], lateral=(1, 0.2, 0.55)), k=0.022, mirror=True)
        sub(E(eb + (et - eb) * 0.45 + np.array([0.004, 0, 0.009]), (0.022, 0.04, 0.007), along(eb, et)), k=0.005, mirror=True)
        # Передні ноги: короткі, тонкі; ратиці з двох пальців і задні «рудиментарні» пальці.
        add(E((0.122, 0.375, 0.24), (0.045, 0.052, 0.046)), k=0.035, mirror=True)
        add(T([(0.122, 0.385, 0.262), (0.112, 0.265, 0.27), (0.104, 0.182, 0.28)], [(0.046, 0.056), (0.033, 0.039), (0.026, 0.029)], k=0.02), k=0.035, mirror=True)
        add(E((0.104, 0.17, 0.28), (0.025, 0.028, 0.028)), k=0.015, mirror=True)
        add(T([(0.104, 0.17, 0.28), (0.098, 0.07, 0.296)], [(0.023, 0.025), (0.022, 0.024)]), k=0.015, mirror=True)
        self._hoof(vol, self.HOOF_F)
        # Задні ноги.
        add(T([(0.112, 0.41, -0.31), (0.106, 0.29, -0.38), (0.102, 0.205, -0.43)], [(0.046, 0.065), (0.031, 0.042), (0.022, 0.026)], k=0.02), k=0.04, mirror=True)
        add(E((0.102, 0.2, -0.448), (0.022, 0.03, 0.03)), k=0.015, mirror=True)
        add(T([(0.102, 0.2, -0.44), (0.097, 0.07, -0.427)], [(0.023, 0.025), (0.022, 0.024)]), k=0.015, mirror=True)
        self._hoof(vol, self.HOOF_B)
        # Хвіст: тонкий, з китичкою.
        add(T([(0, 0.68, -0.52), (0, 0.61, -0.585), (0, 0.51, -0.605), (0, 0.43, -0.612), (0, 0.405, -0.613)],
              [(0.022, 0.022), (0.014, 0.014), (0.012, 0.012), (0.021, 0.024), (0.012, 0.012)], k=0.01), k=0.03)
        # Грива зі щетини вздовж хребта.
        add(T([(0, 0.79, 0.44), (0, 0.85, 0.3), (0, 0.86, 0.14), (0, 0.84, -0.05), (0, 0.81, -0.2)],
              [(0.028, 0.035), (0.034, 0.048), (0.03, 0.045), (0.02, 0.03), (0.006, 0.008)], k=0.03), k=0.05)
        vol.apply(HalfSpace((0, 0, 0), (0, -1, 0), (-0.3, -0.01, -0.8), (0.3, 0.02, 1.0)), 'inter', 0.0)

    def displace(self, v, n):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        mane = sm(0.8, 0.86, y) * sm(-0.22, -0.05, z) * sm(0.48, 0.4, z) * sm(0.06, 0.02, np.abs(x))
        body = sm(0.36, 0.42, y) * sm(0.84, 0.64, z) * (1 - mane)
        cheek = sm(0.56, 0.6, z) * sm(0.7, 0.64, z) * sm(0.45, 0.5, y) * sm(0.6, 0.55, y)
        amp = 0.0016 + 0.0016 * body + 0.009 * mane + 0.003 * cheek
        bristle = stretch_noise(v, 70.0, (1.0, 0.35, 0.5), 3) * 0.6 + vnoise(v, 160.0, 5) * 0.4
        return amp * (bristle - 0.45) * 2.0

    def weight_setup(self, v, B):
        return quad_weights(v, B, 1.0, self.NOSE, self.CHIN, self.TAIL_TIP, self.EAR_TIP, self.TOE_F, self.TOE_B,
                            self.mouth_y, 0.59, 0.29, 0.66, -0.515, 0.05, 0.3, (0, 0.6, -0.53), self.weight_iters, jaw_xmax=0.07, tail_r=0.04)

    def paint(self, v, n, lab, ao, B):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        ax = np.abs(x)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        names = self.bone_names
        bl = np.array([names[i] for i in lab])
        isin = lambda *a: np.isin(bl, a).astype(float)  # noqa: E731
        leg = isin('radFL', 'radFR', 'canFL', 'canFR', 'pasFL', 'pasFR', 'tibBL', 'tibBR', 'metBL', 'metBR', 'pasBL', 'pasBR')
        istail = isin('tail1', 'tail2', 'tail3')
        isear = isin('earL', 'earR')
        head = ((z > 0.5) & (y < 0.72)).astype(float) * (1 - isear)
        g1 = fbm(v, 5.0, 2, 1)
        g2 = vnoise(v, 70.0, 7)
        tips = stretch_noise(v, 28.0, (1.0, 0.4, 0.6), 11)
        # Темна щетина з світлішими (сіро-палевими) кінчиками на боках.
        col = mix(lin(0x4d4238), lin(0x3c342c), g1)
        col = mix(col, lin(0x75654f), sm(0.45, 0.8, tips) * sm(0.6, -0.2, ny) * 0.55)
        mane = sm(0.78, 0.84, y) * sm(0.5, 0.42, z) * sm(-0.25, -0.1, z)
        col = mix(col, lin(0x221d19), np.clip(mane + sm(0.3, 0.9, ny) * 0.3, 0, 1))
        belly = sm(-0.3, -0.8, ny) * sm(0.5, 0.4, y)
        col = mix(col, lin(0x4b4037), belly * 0.5)
        col = mix(col, lin(0x2b2520), leg * 0.85)
        # Голова: сіруваті щоки й рило, темний лоб, п'ятачок.
        snout = head * sm(0.68, 0.76, z)
        col = mix(col, lin(0x655849), snout * 0.7)
        jowl = head * sm(0.55, 0.6, z) * sm(0.72, 0.66, z) * sm(0.58, 0.48, y) * sm(0.3, 0.6, np.abs(nx))
        col = mix(col, lin(0x857766), jowl * 0.55)
        col = mix(col, lin(0x2a2420), head * sm(0.2, 0.8, ny) * sm(0.7, 0.62, z) * 0.6)
        dsk = frame_from((0, -0.62, 0.79))[:, 2]
        dd = (v - [0, 0.36, 0.866]) @ dsk
        disc = head * sm(0.0, 0.012, dd) * sm(0.05, 0.04, np.linalg.norm((v - [0, 0.36, 0.866]) - np.outer(dd, dsk), axis=1))
        col = mix(col, lin(0x4a3a35), disc)
        nostril = disc * sm(0.016, 0.009, np.linalg.norm(np.stack([ax - 0.016, y - 0.352, z - 0.882], 1), axis=1))
        col = mix(col, lin(0x1a1414), nostril * 0.9)
        lipline = head * sm(0.62, 0.66, z) * np.exp(-((y - self.mouth_y(z)) / 0.007) ** 2)
        col = mix(col, lin(0x1e1916), lipline * 0.8)
        rough = np.full(len(v), 0.9)
        col, rough = paint_eyes(v, col, rough, self.EYE, 0.0105, (0.85, 0.15, 0.5), 0x4a2e1a, 0x2a1a10, rim_hex=0x1a1513, rim_w=0.005)
        # Вуха: темні, всередині світліша щетина.
        earin = isear * sm(0.0, 0.5, np.sign(x) * nx * 0.4 + nz * 0.7)
        col = mix(col, lin(0x2a241f), isear)
        col = mix(col, lin(0x5c5045), earin * 0.6)
        col = mix(col, lin(0x1a1714), istail)
        # Ратиці — темно-сірі, гладкіші.
        hoof = np.zeros(len(v))
        for hc in self.hooves:
            for sg in (1, -1):
                p = np.array([hc[0] * sg, hc[1], hc[2]])
                hoof = np.maximum(hoof, sm(0.034, 0.026, np.linalg.norm((v - p) * [1, 0.8, 1], axis=1)))
        hoof = hoof * sm(0.07, 0.05, y)
        col = mix(col, lin(0x1d1a18), hoof)
        col = col * (0.55 + 0.45 * ao)[:, None] * (0.92 + 0.16 * g2)[:, None]
        rough = np.where(disc > 0.5, 0.5, rough)
        rough = np.where(hoof > 0.5, 0.6, rough)
        return col, rough
