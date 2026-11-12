"""Сірий вовк (Canis lupus), дорослий самець ≈ 40 кг: висота в холці ≈ 0,78 м, тулуб ≈ 1,15 м, хвіст ≈ 0,45 м."""
import numpy as np

from anat import BONE_ORDER, along, canid_paw, claw_mask, mirror_bones, paint_eyes, quad_weights
from beast import Species, fbm, lin, mix, sm, stretch_noise, vnoise
from sdf import Ellipsoid as E, HalfSpace, RoundCone as RC, Tube as T


class Wolf(Species):
    name = 'wolf'
    h = 0.003
    bounds = ((-0.17, -0.004, -0.69), (0.17, 1.0, 0.84))
    tris0 = 13000
    tris1 = 2600
    weight_iters = 20
    focus = ((0, 0.81, 0.665), 0.1, 1.5)

    B = {
        'root': (None, (0, 0.64, 0.0)),
        'pelvis': ('root', (0, 0.66, -0.28)),
        'chest': ('root', (0, 0.62, 0.22)),
        'neck1': ('chest', (0, 0.66, 0.38)),
        'neck2': ('neck1', (0, 0.75, 0.48)),
        'head': ('neck2', (0, 0.83, 0.56)),
        'jaw': ('head', (0, 0.785, 0.605)),
        'tail1': ('pelvis', (0, 0.655, -0.44)),
        'tail2': ('tail1', (0, 0.585, -0.525)),
        'tail3': ('tail2', (0, 0.455, -0.585)),
        'scapFL': ('chest', (-0.075, 0.72, 0.29)),
        'humFL': ('scapFL', (-0.093, 0.54, 0.36)),
        'radFL': ('humFL', (-0.094, 0.405, 0.24)),
        'canFL': ('radFL', (-0.088, 0.152, 0.255)),
        'pasFL': ('canFL', (-0.086, 0.042, 0.272)),
        'femBL': ('pelvis', (-0.083, 0.6, -0.32)),
        'tibBL': ('femBL', (-0.092, 0.42, -0.235)),
        'metBL': ('tibBL', (-0.084, 0.178, -0.402)),
        'pasBL': ('metBL', (-0.081, 0.04, -0.387)),
        'earL': ('head', (-0.047, 0.885, 0.572)),
    }
    TAIL_TIP = (0, 0.25, -0.618)
    NOSE = (0, 0.776, 0.802)
    TOE_F = (-0.086, 0.0, 0.335)
    TOE_B = (-0.081, 0.0, -0.33)
    EAR_TIP = (-0.068, 0.972, 0.558)
    EYE = (0.0385, 0.833, 0.659)
    PAW_F = (0.086, 0.0, 0.29)
    PAW_B = (0.081, 0.0, -0.368)

    def __init__(self):
        hb = self.B['head'][1]
        self.extra_header = {
            'headPitch': round(float(np.arctan2(hb[1] - self.NOSE[1], self.NOSE[2] - hb[2])), 4),
            'nose': list(self.NOSE),
            'footFwd': [0.062, 0.055],
            'tailTip': list(self.TAIL_TIP),
            'eye': list(self.EYE),
        }
        self.claws = []

    def bones(self):
        return mirror_bones(self.B, BONE_ORDER)

    def mouth_y(self, z):
        return 0.763 - 0.012 * np.clip((z - 0.64) / 0.13, 0, 1)

    # ─── Форма ───
    def sculpt(self, vol):
        add, sub = vol.add, vol.sub
        P = lambda nm: np.asarray(self.B[nm][1], float)  # noqa: E731
        M = np.array([-1, 1, 1])
        # Тулуб: грудна клітка з кілем грудини, лопатки, лінія спини, передгруддя, живіт, поперек, круп, стегна.
        add(E((0, 0.6, 0.15), (0.1, 0.182, 0.235)))
        add(E((0, 0.6, 0.23), (0.098, 0.17, 0.17)), k=0.05)
        add(E((0, 0.478, 0.215), (0.07, 0.058, 0.12)), k=0.06)
        add(E((0.062, 0.6, 0.292), (0.045, 0.092, 0.078), along(P('scapFL') * M, P('humFL') * M)), k=0.05, mirror=True)
        add(T([(0, 0.712, 0.37), (0, 0.718, 0.22), (0, 0.708, 0.02), (0, 0.703, -0.18), (0, 0.69, -0.34), (0, 0.665, -0.42)],
              [(0.066, 0.07), (0.075, 0.07), (0.072, 0.068), (0.072, 0.068), (0.076, 0.066), (0.05, 0.045)], k=0.04), k=0.06)
        add(E((0.084, 0.468, 0.30), (0.05, 0.088, 0.068), along((0.093, 0.54, 0.36), (0.094, 0.41, 0.24))), k=0.045, mirror=True)
        add(E((0, 0.53, 0.355), (0.082, 0.105, 0.088)), k=0.06)
        add(E((0, 0.63, -0.05), (0.088, 0.12, 0.2), along((0, 0.73, -0.24), (0, 0.5, 0.12))), k=0.08)
        add(E((0, 0.67, -0.13), (0.076, 0.08, 0.18)), k=0.07)
        add(E((0, 0.664, -0.315), (0.084, 0.078, 0.13)), k=0.07)
        add(E((0.07, 0.548, -0.305), (0.056, 0.13, 0.102), along((0.083, 0.62, -0.33), (0.092, 0.42, -0.235))), k=0.06, mirror=True)
        add(E((0.05, 0.585, -0.385), (0.051, 0.1, 0.07)), k=0.05, mirror=True)
        # Шия з гривою й шерстю на горлі.
        add(T([(0, 0.62, 0.28), (0, 0.7, 0.4), (0, 0.775, 0.495), (0, 0.815, 0.552)],
              [(0.09, 0.14), (0.083, 0.122), (0.07, 0.095), (0.06, 0.076)], k=0.03), k=0.06)
        add(T([(0, 0.735, 0.3), (0, 0.765, 0.4), (0, 0.8, 0.48), (0, 0.835, 0.54)], [(0.055, 0.04), (0.058, 0.042), (0.05, 0.036), (0.04, 0.03)], k=0.03), k=0.05)
        add(E((0, 0.65, 0.47), (0.078, 0.095, 0.076)), k=0.06)
        # Голова: череп, лоб, вилиці, «бакенбарди», морда, ніс, нижня щелепа, лінія рота.
        add(E((0, 0.836, 0.584), (0.063, 0.061, 0.078)), k=0.045)
        add(E((0, 0.846, 0.626), (0.05, 0.04, 0.046)), k=0.03)
        add(E((0.046, 0.808, 0.618), (0.031, 0.03, 0.052)), k=0.03, mirror=True)
        add(E((0.061, 0.774, 0.578), (0.04, 0.075, 0.058)), k=0.035, mirror=True)
        add(T([(0, 0.805, 0.635), (0, 0.792, 0.71), (0, 0.779, 0.778)], [(0.043, 0.039), (0.032, 0.032), (0.024, 0.026)], k=0.02), k=0.035)
        add(E((0, 0.779, 0.788), (0.0215, 0.017, 0.0145)), k=0.008)
        sub(E((0.0088, 0.775, 0.8), (0.005, 0.0039, 0.0066)), k=0.003, mirror=True)
        add(T([(0, 0.778, 0.6), (0, 0.757, 0.68), (0, 0.749, 0.762)], [(0.041, 0.025), (0.027, 0.0175), (0.0165, 0.0118)], k=0.01), k=0.02)
        sub(T([(0.036, 0.765, 0.645), (0.027, 0.763, 0.7), (0.0, 0.76, 0.77)], [(0.0045, 0.003), (0.0045, 0.0028), (0.003, 0.0024)]), k=0.006, mirror=True)
        # Очі в очницях, надбрівні дуги.
        ex, ey, ez = self.EYE
        sub(E((ex, ey, ez), (0.0135, 0.0125, 0.013)), k=0.005, mirror=True)
        add(E((ex, ey, ez), (0.0112, 0.0112, 0.0112)), k=0.002, mirror=True)
        add(E((0.034, 0.851, 0.655), (0.016, 0.0062, 0.013)), k=0.01, mirror=True)
        # Вуха: стоячі, трикутні, увігнуті спереду.
        eb, et = P('earL') * M, np.asarray(self.EAR_TIP) * M
        add(T([eb, eb + (et - eb) * 0.5, et], [(0.032, 0.0125), (0.021, 0.0098), (0.0045, 0.0035)], lateral=(1, 0, 0.5)), k=0.018, mirror=True)
        sub(E(eb + (et - eb) * 0.42 + np.array([0.002, 0, 0.0085]), (0.016, 0.03, 0.006), along(eb, et)), k=0.004, mirror=True)
        # Передні ноги: лікоть, передпліччя, зап'ясток із подушечкою, п'ясток, лапа.
        add(E((0.092, 0.402, 0.208), (0.028, 0.034, 0.028)), k=0.025, mirror=True)
        add(T([(0.093, 0.412, 0.236), (0.091, 0.3, 0.246), (0.088, 0.168, 0.256)], [(0.034, 0.045), (0.027, 0.033), (0.0215, 0.025)], k=0.02), k=0.03, mirror=True)
        add(E((0.088, 0.152, 0.256), (0.0195, 0.023, 0.0225)), k=0.012, mirror=True)
        add(E((0.088, 0.138, 0.234), (0.008, 0.012, 0.008)), k=0.008, mirror=True)
        add(T([(0.088, 0.152, 0.256), (0.086, 0.062, 0.268)], [(0.018, 0.0205), (0.02, 0.021)]), k=0.02, mirror=True)
        self.claws += canid_paw(vol, self.PAW_F, 1.08)
        # Задні ноги: колінний суглоб, гомілка, ахіллове сухожилля й п'ятка, плесно, лапа.
        add(E((0.088, 0.43, -0.238), (0.034, 0.044, 0.044)), k=0.03, mirror=True)
        add(T([(0.089, 0.44, -0.255), (0.086, 0.3, -0.33), (0.083, 0.19, -0.395)], [(0.034, 0.06), (0.025, 0.041), (0.0165, 0.023)], k=0.02), k=0.04, mirror=True)
        add(RC((0.083, 0.31, -0.355), (0.082, 0.182, -0.424), 0.0135, 0.0125), k=0.02, mirror=True)
        add(T([(0.082, 0.18, -0.405), (0.081, 0.06, -0.39)], [(0.0175, 0.0225), (0.0195, 0.0215)]), k=0.02, mirror=True)
        self.claws += canid_paw(vol, self.PAW_B, 1.0)
        # Хвіст: пухнастий, звисає майже до скакальних суглобів.
        add(T([(0, 0.665, -0.425), (0, 0.63, -0.5), (0, 0.55, -0.562), (0, 0.44, -0.598), (0, 0.33, -0.613), (0, 0.255, -0.618)],
              [(0.034, 0.034), (0.045, 0.048), (0.057, 0.061), (0.058, 0.062), (0.045, 0.048), (0.011, 0.011)], k=0.03), k=0.05)
        # Плоскі подушечки лап.
        vol.apply(HalfSpace((0, 0, 0), (0, -1, 0), (-0.3, -0.01, -0.8), (0.3, 0.02, 0.9)), 'inter', 0.0)

    def displace(self, v, n):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        tail = sm(-0.42, -0.47, z) * sm(0.2, 0.3, y)
        mane = sm(0.25, 0.32, z) * sm(0.56, 0.5, z) * sm(0.6, 0.7, y)
        throat = sm(0.4, 0.45, z) * sm(0.54, 0.5, z) * sm(0.74, 0.68, y)
        cheek = sm(0.6, 0.57, z) * sm(0.54, 0.57, z) * sm(0.86, 0.8, y) * sm(0.73, 0.76, y) * sm(0.04, 0.06, np.abs(x))
        pants = sm(-0.36, -0.42, z) * sm(0.38, 0.48, y) * sm(0.66, 0.6, y) * sm(0.02, 0.05, np.abs(x))
        belly = sm(0.48, 0.52, y) * sm(0.58, 0.54, y) * sm(-0.2, 0.0, z) * sm(0.32, 0.2, z) * sm(-0.3, -0.7, n[:, 1])
        amp = 0.0011 + 0.0045 * tail + 0.0032 * mane + 0.0028 * throat + 0.0032 * cheek + 0.0028 * pants + 0.0022 * belly
        clump = stretch_noise(v, 55.0, (1.0, 0.45, 0.35), 3) * 0.65 + vnoise(v, 140.0, 5) * 0.35
        return amp * (clump - 0.45) * 2.0

    # ─── Скін ───
    def weight_setup(self, v, B):
        return quad_weights(v, B, 1.0, self.NOSE, (0, 0.75, 0.77), self.TAIL_TIP, self.EAR_TIP, self.TOE_F, self.TOE_B,
                            self.mouth_y, 0.6, 0.72, 0.865, -0.415, 0.075, 0.36, (0, 0.64, -0.43), self.weight_iters, tail_r=0.078)

    # ─── Хутро ───
    def paint(self, v, n, lab, ao, B):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        ax = np.abs(x)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        names = self.bone_names
        bl = np.array([names[i] for i in lab])
        isleg = np.isin(bl, ['radFL', 'radFR', 'canFL', 'canFR', 'pasFL', 'pasFR', 'tibBL', 'tibBR', 'metBL', 'metBR', 'pasBL', 'pasBR']).astype(float)
        ispaw = np.isin(bl, ['pasFL', 'pasFR', 'pasBL', 'pasBR']).astype(float)
        istail = np.isin(bl, ['tail1', 'tail2', 'tail3']).astype(float)
        isear = np.isin(bl, ['earL', 'earR']).astype(float)
        head = ((z > 0.52) & (y > 0.7)).astype(float)
        g1 = fbm(v, 4.0, 2, 1)
        g2 = vnoise(v, 85.0, 7)
        agouti = stretch_noise(v, 18.0, (1.0, 0.6, 0.4), 11)
        # Основа: сіро-бура з чорними кінчиками остьового волосся й світлими смугами (агуті).
        col = mix(lin(0x877a69), lin(0x75695b), g1)
        col = mix(col, lin(0x55493f), sm(0.55, 0.85, agouti) * 0.18)
        # Темне «сідло» на спині й холці.
        saddle = sm(0.62, 0.74, y) * sm(0.2, 0.75, ny) * sm(-0.36, -0.2, z) * sm(0.42, 0.3, z) * (1 - head)
        col = mix(col, lin(0x433c35), saddle * 0.72)
        # Боки й стегна — тепліші.
        flank = sm(0.3, -0.2, ny) * sm(0.38, 0.5, y) * (1 - head)
        col = mix(col, lin(0x9c8268), flank * 0.3)
        # Світлий низ: горло, груди, живіт, внутрішні сторони ніг.
        under = sm(-0.25, -0.75, ny) * sm(0.62, 0.5, y)
        throat = sm(0.4, 0.47, z) * sm(0.6, 0.53, z) * sm(0.76, 0.62, y) * sm(0.35, -0.3, ny) * sm(-0.6, 0.2, nz) * (1 - head)
        chest = sm(0.33, 0.43, z) * sm(0.63, 0.5, y) * sm(-0.2, 0.5, nz)
        inner = isleg * sm(0.15, 0.7, -np.sign(x) * nx)
        light = np.clip(under + throat + chest * 0.85 + inner * 0.75, 0, 1)
        col = mix(col, lin(0xcdbfa6), light * 0.85)
        # Ноги: палеві; темна смуга спереду передпліччя; світлі лапи.
        legc = mix(lin(0x8d7257), lin(0x9c8367), sm(0.3, 0.05, y))
        col = mix(col, legc, isleg * (1 - inner * 0.6) * 0.85)
        stripe = np.isin(bl, ['radFL', 'radFR']) * sm(0.5, 0.9, nz) * sm(0.17, 0.25, y) * sm(0.4, 0.3, y)
        col = mix(col, lin(0x4f4235), stripe * 0.7)
        col = mix(col, lin(0x8f7a63), ispaw * 0.5)
        # Голова: палева морда, кремові губи й щоки, світлі «брови», темна лінія губ, чорний ніс.
        muzzle = head * sm(0.66, 0.7, z)
        col = mix(col, lin(0x9a7d60), head * sm(0.2, 0.8, ny) * 0.5)
        col = mix(col, mix(lin(0x8a6c50), lin(0x9b7a5a), g2), muzzle * sm(0.2, 0.7, ny))
        cheeks = head * sm(0.59, 0.65, z) * sm(0.35, -0.35, ny) * sm(0.81, 0.775, y) * sm(0.5, 0.0, nz)
        lips = head * sm(0.63, 0.67, z) * sm(0.752, 0.766, y) * sm(0.79, 0.775, y)
        col = mix(col, lin(0xe4dac8), np.clip(cheeks * 0.9 + lips * 0.8, 0, 1))
        jawc = head * (bl == 'jaw')
        col = mix(col, lin(0xddd2bf), jawc * 0.85)
        brow = head * np.exp(-((ax - 0.03) ** 2 + (y - 0.851) ** 2 + (z - 0.646) ** 2) / (2 * 0.011 ** 2))
        col = mix(col, lin(0xd9cbb1), brow * 0.75)
        lipline = head * sm(0.61, 0.65, z) * np.exp(-((y - self.mouth_y(z)) / 0.005) ** 2) * (ax > 0.004)
        bridge = head * sm(0.64, 0.68, z) * sm(0.79, 0.775, z) * sm(0.45, 0.85, ny)
        col = mix(col, lin(0x6e5641), bridge * 0.6)
        tear = head * np.exp(-((ax - 0.03) ** 2 / 0.006 ** 2 + (y - 0.822) ** 2 / 0.006 ** 2 + (z - 0.672) ** 2 / 0.012 ** 2))
        col = mix(col, lin(0x3e332a), tear * 0.6)
        col = mix(col, lin(0x1d1a17), lipline * 0.9)
        nose = sm(0.032, 0.02, np.linalg.norm((v - [0, 0.779, 0.79]) * [1.0, 1.25, 1.0], axis=1)) * sm(0.772, 0.786, z)
        col = mix(col, lin(0x141211), nose)
        rough = np.full(len(v), 0.92)
        col, rough = paint_eyes(v, col, rough, self.EYE, 0.0112, (0.5, 0.05, 0.86), 0xb98a2e, 0x6b4a18)
        # Вуха: зовні рудуваті з темними краями й кінчиками, всередині світла шерсть.
        earfront = isear * sm(0.0, 0.5, nz)
        col = mix(col, mix(lin(0x8e6d4f), lin(0x6c5440), g2), isear * (1 - earfront))
        col = mix(col, lin(0xb3a48d), earfront * 0.8)
        tip = np.stack([np.where(x < 0, self.EAR_TIP[0], -self.EAR_TIP[0]), np.full(len(v), self.EAR_TIP[1]), np.full(len(v), self.EAR_TIP[2])], axis=1)
        etip = isear * sm(0.032, 0.012, np.linalg.norm(v - tip, axis=1))
        erim = isear * sm(0.35, 0.1, np.abs(nz)) * sm(0.9, 0.93, y)
        col = mix(col, lin(0x2c2622), np.clip(etip + erim * 0.6, 0, 1) * 0.65)
        # Хвіст: темніший зверху, світліший знизу, чорний кінчик і темна пляма біля основи.
        t1y = self.B['tail1'][1][1]
        tt = np.clip((t1y - y) / (t1y - self.TAIL_TIP[1]), 0, 1)
        tailtop = istail * sm(-0.3, 0.4, -nz)
        col = mix(col, lin(0x5a5046), istail * tailtop * 0.7)
        col = mix(col, lin(0xb3a28a), istail * (1 - tailtop) * 0.5)
        col = mix(col, lin(0x1b1816), istail * sm(0.76, 0.9, tt))
        gland = istail * np.exp(-((y - 0.632) ** 2 + (z + 0.478) ** 2) / (2 * 0.018 ** 2)) * sm(0.0, -0.6, nz)
        col = mix(col, lin(0x2a2521), gland * 0.8)
        # Подушечки й кігті — темні.
        pad = ispaw * sm(0.012, 0.002, y)
        col = mix(col, lin(0x2b2623), pad * 0.9)
        claw = claw_mask(v, self.claws, 0.0075)
        col = mix(col, lin(0x2a2724), claw)
        col = col * (0.55 + 0.45 * ao)[:, None] * (0.94 + 0.12 * g2)[:, None]
        rough = np.where(nose > 0.5, 0.4, rough)
        rough = np.where(pad > 0.5, 0.75, rough)
        rough = np.where(claw > 0.5, 0.55, rough)
        return col, rough
