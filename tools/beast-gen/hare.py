"""Заєць-русак (Lepus europaeus), дорослий ≈ 4 кг: тулуб ≈ 0,6 м, вуха ≈ 0,12 м, задня ступня ≈ 0,15 м."""
import numpy as np

from anat import BONE_ORDER, along, mirror_bones, quad_weights
from beast import Species, fbm, lin, mix, sm, stretch_noise, vnoise
from sdf import Ellipsoid as E, HalfSpace, Tube as T


class Hare(Species):
    name = 'hare'
    h = 0.0015
    bounds = ((-0.08, -0.003, -0.29), (0.08, 0.425, 0.33))
    tris0 = 9000
    tris1 = 1800
    weight_iters = 16
    focus = ((0, 0.255, 0.255), 0.05, 1.3)

    B = {
        'root': (None, (0, 0.2, 0.0)),
        'pelvis': ('root', (0, 0.22, -0.125)),
        'chest': ('root', (0, 0.19, 0.1)),
        'neck1': ('chest', (0, 0.212, 0.168)),
        'neck2': ('neck1', (0, 0.234, 0.198)),
        'head': ('neck2', (0, 0.256, 0.224)),
        'jaw': ('head', (0, 0.236, 0.244)),
        'tail1': ('pelvis', (0, 0.222, -0.212)),
        'tail2': ('tail1', (0, 0.232, -0.234)),
        'tail3': ('tail2', (0, 0.232, -0.25)),
        'scapFL': ('chest', (-0.03, 0.236, 0.106)),
        'humFL': ('scapFL', (-0.04, 0.17, 0.132)),
        'radFL': ('humFL', (-0.039, 0.114, 0.086)),
        'canFL': ('radFL', (-0.035, 0.034, 0.106)),
        'pasFL': ('canFL', (-0.034, 0.011, 0.121)),
        'femBL': ('pelvis', (-0.041, 0.2, -0.14)),
        'tibBL': ('femBL', (-0.05, 0.122, -0.07)),
        'metBL': ('tibBL', (-0.045, 0.052, -0.176)),
        'pasBL': ('metBL', (-0.042, 0.012, -0.074)),
        'earL': ('head', (-0.016, 0.284, 0.22)),
    }
    TAIL_TIP = (0, 0.226, -0.262)
    NOSE = (0, 0.243, 0.304)
    CHIN = (0, 0.226, 0.288)
    TOE_F = (-0.034, 0.0, 0.152)
    TOE_B = (-0.042, 0.0, -0.032)
    EAR_TIP = (-0.031, 0.405, 0.183)
    EYE = (0.0245, 0.262, 0.25)

    def __init__(self):
        hb = self.B['head'][1]
        self.extra_header = {
            'headPitch': round(float(np.arctan2(hb[1] - self.NOSE[1], self.NOSE[2] - hb[2])), 4),
            'nose': list(self.NOSE),
            'footFwd': [0.031, 0.042],
            'tailTip': list(self.TAIL_TIP),
            'eye': list(self.EYE),
        }

    def bones(self):
        return mirror_bones(self.B, BONE_ORDER)

    def mouth_y(self, z):
        return 0.2335 + 0.0 * z

    def sculpt(self, vol):
        add, sub = vol.add, vol.sub
        P = lambda nm: np.asarray(self.B[nm][1], float)  # noqa: E731
        M = np.array([-1, 1, 1])
        # Тулуб «яйцем»: вузькі груди, вигнута спина, потужні стегна.
        add(E((0, 0.185, 0.068), (0.049, 0.064, 0.085)))
        add(E((0, 0.205, -0.04), (0.058, 0.074, 0.11), along((0, 0.25, -0.13), (0, 0.17, 0.06))), k=0.03)
        add(T([(0, 0.214, 0.14), (0, 0.236, 0.07), (0, 0.255, -0.02), (0, 0.264, -0.1), (0, 0.254, -0.165), (0, 0.228, -0.206)],
              [(0.032, 0.03), (0.04, 0.036), (0.048, 0.042), (0.052, 0.045), (0.05, 0.042), (0.034, 0.028)], k=0.02), k=0.03)
        add(E((0.042, 0.168, -0.118), (0.036, 0.068, 0.072), along(P('femBL') * M, P('tibBL') * M)), k=0.03, mirror=True)
        add(E((0, 0.198, -0.168), (0.052, 0.058, 0.058)), k=0.03)
        add(E((0.03, 0.182, 0.11), (0.021, 0.048, 0.033), along(P('scapFL') * M, P('humFL') * M)), k=0.022, mirror=True)
        add(E((0, 0.162, 0.118), (0.034, 0.042, 0.036)), k=0.025)
        # Коротка товста шия.
        add(T([(0, 0.188, 0.108), (0, 0.218, 0.168), (0, 0.244, 0.21)], [(0.037, 0.05), (0.033, 0.042), (0.029, 0.034)], k=0.015), k=0.022)
        # Голова: округлий череп, щоки, тупа морда з роздвоєною губою, великі очі з боків.
        add(E((0, 0.261, 0.235), (0.029, 0.03, 0.04)), k=0.018)
        add(E((0.021, 0.246, 0.243), (0.018, 0.024, 0.03)), k=0.015, mirror=True)
        add(T([(0, 0.256, 0.258), (0, 0.249, 0.283), (0, 0.242, 0.299)], [(0.0205, 0.022), (0.017, 0.018), (0.012, 0.0125)], k=0.01), k=0.016)
        add(E((0.006, 0.236, 0.296), (0.0085, 0.008, 0.0075)), k=0.004, mirror=True)        # «подушечки» верхньої губи
        sub(E((0, 0.232, 0.304), (0.0016, 0.007, 0.006)), k=0.0015)                         # розщеплена губа
        sub(E((0.0035, 0.2455, 0.3045), (0.0022, 0.0013, 0.0028)), k=0.001, mirror=True)  # ніздрі
        add(T([(0, 0.232, 0.25), (0, 0.228, 0.282)], [(0.0135, 0.0078), (0.0078, 0.0052)], k=0.004), k=0.008)
        ex, ey, ez = self.EYE
        sub(E((ex, ey, ez), (0.0098, 0.0094, 0.0098)), k=0.0025, mirror=True)
        add(E((ex, ey, ez), (0.0086, 0.0086, 0.0086)), k=0.001, mirror=True)
        # Довгі вуха («ложечки»), увігнуті назовні-вперед.
        eb, et = P('earL') * M, np.asarray(self.EAR_TIP) * M
        lat = (1, 0, -0.35)
        add(T([eb, eb + (et - eb) * 0.18, eb + (et - eb) * 0.55, eb + (et - eb) * 0.88, et],
              [(0.0105, 0.0065), (0.0158, 0.0052), (0.0175, 0.0042), (0.011, 0.0036), (0.0025, 0.0022)], lateral=lat), k=0.006, mirror=True)
        sub(E(eb + (et - eb) * 0.52 + np.array([0.0028, 0, 0.0032]), (0.0125, 0.046, 0.0032), along(eb, et)), k=0.0018, mirror=True)
        # Передні ноги: довгі тонкі.
        add(E((0.041, 0.142, 0.108), (0.018, 0.034, 0.026), along((0.04, 0.17, 0.132), (0.039, 0.114, 0.086))), k=0.018, mirror=True)
        add(T([(0.039, 0.118, 0.088), (0.037, 0.072, 0.097), (0.035, 0.038, 0.105)], [(0.0108, 0.0138), (0.0084, 0.0102), (0.0074, 0.008)], k=0.006), k=0.012, mirror=True)
        add(T([(0.035, 0.038, 0.105), (0.034, 0.016, 0.118), (0.034, 0.008, 0.146)], [(0.0078, 0.0085), (0.009, 0.0085), (0.0072, 0.0058)], k=0.006), k=0.006, mirror=True)
        # Задні ноги: гомілка, п'ятка, довга опушена ступня.
        add(T([(0.05, 0.124, -0.074), (0.048, 0.088, -0.128), (0.045, 0.058, -0.168)], [(0.016, 0.026), (0.0125, 0.019), (0.0098, 0.0135)], k=0.008), k=0.016, mirror=True)
        add(E((0.045, 0.048, -0.181), (0.0105, 0.0135, 0.0125)), k=0.008, mirror=True)
        add(T([(0.045, 0.042, -0.18), (0.043, 0.019, -0.112), (0.042, 0.011, -0.05), (0.042, 0.008, -0.032)],
              [(0.0118, 0.0125), (0.0128, 0.0108), (0.0118, 0.0088), (0.009, 0.0068)], k=0.006), k=0.01, mirror=True)
        # Хвіст: короткий, загнутий догори.
        add(T([(0, 0.22, -0.205), (0, 0.232, -0.228), (0, 0.233, -0.248), (0, 0.226, -0.26)], [(0.016, 0.017), (0.021, 0.022), (0.019, 0.02), (0.008, 0.008)], k=0.01), k=0.016)
        vol.apply(HalfSpace((0, 0, 0), (0, -1, 0), (-0.2, -0.01, -0.4), (0.2, 0.01, 0.4)), 'inter', 0.0)

    def displace(self, v, n):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        body = sm(0.28, 0.2, z)
        ear = sm(0.28, 0.3, y)
        amp = 0.0003 + 0.0002 * body - 0.0002 * ear
        clump = stretch_noise(v, 150.0, (1.0, 0.5, 0.4), 3) * 0.6 + vnoise(v, 320.0, 5) * 0.4
        return amp * (clump - 0.45) * 2.0

    def weight_setup(self, v, B):
        return quad_weights(v, B, 0.35, self.NOSE, self.CHIN, self.TAIL_TIP, self.EAR_TIP, self.TOE_F, self.TOE_B,
                            self.mouth_y, 0.24, 0.215, 0.282, -0.2, 0.04, 0.085, (0, 0.21, -0.2), self.weight_iters, jaw_xmax=0.016, tail_r=0.032)

    def paint(self, v, n, lab, ao, B):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        ax = np.abs(x)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        names = self.bone_names
        bl = np.array([names[i] for i in lab])
        isin = lambda *a: np.isin(bl, a).astype(float)  # noqa: E731
        legF = isin('radFL', 'radFR', 'canFL', 'canFR', 'pasFL', 'pasFR', 'humFL', 'humFR')
        legB = isin('tibBL', 'tibBR', 'metBL', 'metBR', 'pasBL', 'pasBR')
        istail = isin('tail1', 'tail2', 'tail3')
        isear = isin('earL', 'earR')
        head = ((z > 0.205) & (y > 0.215)).astype(float) * (1 - isear)
        g1 = fbm(v, 9.0, 2, 1)
        g2 = vnoise(v, 160.0, 7)
        fleck = vnoise(v, 45.0, 13)
        # Агуті: жовтувато-бура з чорними цятками на спині, рудуватіша на боках.
        col = mix(lin(0x8d6c47), lin(0x7a5c3c), g1)
        back = sm(0.1, 0.8, ny) * (1 - head) * (1 - isear)
        col = mix(col, mix(lin(0x67503a), lin(0x2f261f), sm(0.5, 0.78, fleck)), back * 0.7)
        col = mix(col, lin(0xb08d62), sm(0.35, 0.15, fleck) * back * 0.35)
        flank = sm(0.35, -0.2, ny) * (1 - head)
        col = mix(col, lin(0xa47b4c), flank * 0.4)
        # Рудуваті груди, плечі й ноги.
        rust = np.clip(sm(0.07, 0.13, z) * sm(0.24, 0.15, y) * (1 - head) + legF * 0.8 + legB * sm(0.3, -0.3, ny) * 0.6, 0, 1)
        col = mix(col, lin(0xa8703f), rust * 0.6)
        # Білий живіт і внутрішні сторони ніг.
        under = sm(-0.35, -0.8, ny) * sm(0.2, 0.15, y) * sm(0.17, 0.1, z) * (1 - legB) * (1 - legF) * sm(0.09, 0.12, y)
        inner = (legF + legB) * sm(0.2, 0.7, -np.sign(x) * nx)
        col = mix(col, lin(0xe8e2d7), np.clip(under + inner * 0.7, 0, 1) * 0.9)
        # Ступні знизу бурувато-сірі (опушені).
        sole = legB * sm(0.012, 0.004, y)
        col = mix(col, lin(0x7a6551), sole * 0.6)
        foot = isin('metBL', 'metBR', 'pasBL', 'pasBR')
        col = mix(col, lin(0x9c7b56), foot * 0.5)
        # Голова: світле кільце довкола ока, сірувато-бурі щоки, світла морда, темна ніздря.
        eyec = np.array([0, self.EYE[1], self.EYE[2]])
        de = np.linalg.norm((v - eyec) * [0, 1, 1] + np.stack([ax - self.EYE[0], np.zeros(len(v)), np.zeros(len(v))], 1), axis=1)
        ring = head * sm(0.016, 0.0115, de) * sm(0.0085, 0.0105, de)
        col = mix(col, lin(0x8b7258), head * 0.5)
        col = mix(col, lin(0xa3896a), head * sm(0.27, 0.29, z) * 0.6)
        col = mix(col, lin(0xdccfb7), ring * 0.85)
        lips = head * sm(0.28, 0.295, z) * sm(0.243, 0.232, y)
        col = mix(col, lin(0xd9cdb8), lips * 0.8)
        jawc = head * (bl == 'jaw')
        col = mix(col, lin(0xe4dccd), jawc * 0.8)
        nose = head * sm(0.008, 0.004, np.linalg.norm((v - [0, 0.2455, 0.304]) * [1.0, 1.4, 1.0], axis=1))
        col = mix(col, lin(0x5a4038), nose * 0.85)
        rough = np.full(len(v), 0.88)
        col, rough = self._eyes(v, col, rough)
        # Вуха: сірувато-бурі з чорними кінчиками й світлою облямівкою.
        et = np.stack([np.where(x < 0, self.EAR_TIP[0], -self.EAR_TIP[0]), np.full(len(v), self.EAR_TIP[1]), np.full(len(v), self.EAR_TIP[2])], 1)
        dt = np.linalg.norm(v - et, axis=1)
        col = mix(col, mix(lin(0x8f7d66), lin(0x7c6a55), g2), isear)
        inner_e = isear * sm(0.1, 0.6, np.sign(x) * nx * 0.7 + nz * 0.5)
        col = mix(col, lin(0xbcae98), inner_e * 0.5)
        col = mix(col, lin(0x191513), isear * sm(0.03, 0.02, dt))
        # Хвіст: чорний зверху, білий знизу.
        top = istail * sm(-0.2, 0.4, ny - 0.3 * nz)
        col = mix(col, lin(0xf0ece4), istail * (1 - top))
        col = mix(col, lin(0x1d1916), top)
        col = col * (0.55 + 0.45 * ao)[:, None] * (0.94 + 0.12 * g2)[:, None]
        rough = np.where(nose > 0.5, 0.6, rough)
        return col, rough

    def _eyes(self, v, col, rough):
        r = 0.0086
        for sg in (-1, 1):
            c = np.array([abs(self.EYE[0]) * sg, self.EYE[1], self.EYE[2]])
            d = np.linalg.norm(v - c, axis=1)
            eye = sm(r * 1.16, r * 1.02, d)
            g = np.array([0.93 * sg, 0.12, 0.34])
            g /= np.linalg.norm(g)
            cosg = ((v - c) @ g) / np.maximum(d, 1e-6)
            iris = mix(lin(0xc28b36), lin(0x7a5520), sm(0.9, 0.6, cosg))
            ecol = mix(iris, lin(0x080706), sm(0.88, 0.92, cosg))
            ecol = mix(lin(0x2b1f14), ecol, sm(0.35, 0.55, cosg))
            col = mix(col, ecol, eye)
            rough = np.where(eye > 0.5, 0.4, rough)
        return col, rough
