"""Руда лисиця (Vulpes vulpes), доросла ≈ 6 кг: висота в холці ≈ 0,39 м, тулуб ≈ 0,65 м, хвіст ≈ 0,42 м."""
import numpy as np

from anat import BONE_ORDER, along, canid_paw, claw_mask, mirror_bones, paint_eyes, quad_weights
from beast import Species, fbm, lin, mix, sm, stretch_noise, vnoise
from sdf import Ellipsoid as E, HalfSpace, RoundCone as RC, Tube as T


class Fox(Species):
    name = 'fox'
    h = 0.002
    bounds = ((-0.1, -0.003, -0.64), (0.1, 0.525, 0.43))
    tris0 = 11000
    tris1 = 2200
    weight_iters = 18
    focus = ((0, 0.385, 0.34), 0.055, 1.5)

    B = {
        'root': (None, (0, 0.3, 0.0)),
        'pelvis': ('root', (0, 0.312, -0.15)),
        'chest': ('root', (0, 0.29, 0.12)),
        'neck1': ('chest', (0, 0.31, 0.2)),
        'neck2': ('neck1', (0, 0.352, 0.248)),
        'head': ('neck2', (0, 0.39, 0.288)),
        'jaw': ('head', (0, 0.37, 0.31)),
        'tail1': ('pelvis', (0, 0.305, -0.222)),
        'tail2': ('tail1', (0, 0.262, -0.312)),
        'tail3': ('tail2', (0, 0.198, -0.41)),
        'scapFL': ('chest', (-0.04, 0.34, 0.142)),
        'humFL': ('scapFL', (-0.05, 0.26, 0.18)),
        'radFL': ('humFL', (-0.05, 0.19, 0.118)),
        'canFL': ('radFL', (-0.046, 0.072, 0.128)),
        'pasFL': ('canFL', (-0.045, 0.022, 0.138)),
        'femBL': ('pelvis', (-0.045, 0.29, -0.172)),
        'tibBL': ('femBL', (-0.05, 0.2, -0.122)),
        'metBL': ('tibBL', (-0.046, 0.086, -0.215)),
        'pasBL': ('metBL', (-0.044, 0.021, -0.205)),
        'earL': ('head', (-0.029, 0.418, 0.296)),
    }
    TAIL_TIP = (0, 0.112, -0.585)
    NOSE = (0, 0.367, 0.414)
    CHIN = (0, 0.352, 0.398)
    TOE_F = (-0.045, 0.0, 0.172)
    TOE_B = (-0.044, 0.0, -0.172)
    EAR_TIP = (-0.046, 0.494, 0.286)
    EYE = (0.0205, 0.3935, 0.3375)
    PAW_F = (0.045, 0.0, 0.149)
    PAW_B = (0.044, 0.0, -0.192)

    def __init__(self):
        hb = self.B['head'][1]
        self.extra_header = {
            'headPitch': round(float(np.arctan2(hb[1] - self.NOSE[1], self.NOSE[2] - hb[2])), 4),
            'nose': list(self.NOSE),
            'footFwd': [0.034, 0.033],
            'tailTip': list(self.TAIL_TIP),
            'eye': list(self.EYE),
        }
        self.claws = []

    def bones(self):
        return mirror_bones(self.B, BONE_ORDER)

    def mouth_y(self, z):
        return 0.3625 - 0.006 * np.clip((z - 0.33) / 0.07, 0, 1)

    def sculpt(self, vol):
        add, sub = vol.add, vol.sub
        P = lambda nm: np.asarray(self.B[nm][1], float)  # noqa: E731
        M = np.array([-1, 1, 1])
        # Тулуб (з густим хутром): грудна клітка, лопатки, лінія спини, передгруддя, живіт, круп, стегна.
        add(E((0, 0.282, 0.07), (0.056, 0.09, 0.128)))
        add(E((0, 0.24, 0.105), (0.04, 0.04, 0.072)), k=0.035)
        add(E((0.033, 0.277, 0.146), (0.025, 0.05, 0.042), along(P('scapFL') * M, P('humFL') * M)), k=0.03, mirror=True)
        add(T([(0, 0.333, 0.19), (0, 0.334, 0.1), (0, 0.333, -0.02), (0, 0.332, -0.12), (0, 0.325, -0.19), (0, 0.307, -0.228)],
              [(0.038, 0.04), (0.044, 0.041), (0.044, 0.041), (0.045, 0.041), (0.046, 0.04), (0.03, 0.026)], k=0.025), k=0.035)
        add(E((0.046, 0.23, 0.155), (0.027, 0.048, 0.037), along((0.05, 0.26, 0.18), (0.05, 0.195, 0.12))), k=0.025, mirror=True)
        add(E((0, 0.25, 0.168), (0.046, 0.058, 0.05)), k=0.035)
        add(E((0, 0.29, -0.06), (0.048, 0.06, 0.122), along((0, 0.33, -0.17), (0, 0.24, 0.05))), k=0.045)
        add(E((0, 0.31, -0.165), (0.049, 0.046, 0.072)), k=0.04)
        add(E((0.039, 0.25, -0.165), (0.031, 0.072, 0.056), along((0.045, 0.3, -0.18), (0.05, 0.2, -0.122))), k=0.035, mirror=True)
        add(E((0.03, 0.27, -0.205), (0.029, 0.055, 0.04)), k=0.03, mirror=True)
        # Шия, біла «манишка» на горлі й грудях.
        add(T([(0, 0.285, 0.145), (0, 0.328, 0.208), (0, 0.364, 0.258), (0, 0.384, 0.285)],
              [(0.046, 0.072), (0.042, 0.063), (0.036, 0.049), (0.031, 0.038)], k=0.02), k=0.035)
        add(T([(0, 0.345, 0.17), (0, 0.362, 0.212), (0, 0.386, 0.255), (0, 0.404, 0.282)], [(0.03, 0.021), (0.031, 0.022), (0.027, 0.019), (0.022, 0.016)], k=0.02), k=0.03)
        add(E((0, 0.285, 0.232), (0.04, 0.052, 0.042)), k=0.035)
        # Голова: широкий череп, «бакенбарди», вузька довга морда, ніс, нижня щелепа.
        add(E((0, 0.394, 0.298), (0.035, 0.033, 0.043)), k=0.025)
        add(E((0, 0.4, 0.321), (0.028, 0.022, 0.026)), k=0.018)
        add(E((0.025, 0.381, 0.316), (0.016, 0.016, 0.028)), k=0.016, mirror=True)
        add(E((0.034, 0.369, 0.293), (0.022, 0.036, 0.03)), k=0.02, mirror=True)
        add(T([(0, 0.386, 0.33), (0, 0.376, 0.37), (0, 0.368, 0.404)], [(0.0205, 0.019), (0.0145, 0.0138), (0.0095, 0.0098)], k=0.012), k=0.02)
        add(E((0, 0.368, 0.408), (0.0098, 0.0078, 0.0068)), k=0.004)
        sub(E((0.0044, 0.366, 0.414), (0.0026, 0.002, 0.0032)), k=0.0015, mirror=True)
        add(T([(0, 0.366, 0.31), (0, 0.356, 0.36), (0, 0.353, 0.396)], [(0.021, 0.0128), (0.0128, 0.0088), (0.0078, 0.0058)], k=0.006), k=0.01)
        sub(T([(0.0195, 0.3625, 0.33), (0.0138, 0.361, 0.365), (0.0, 0.3565, 0.4)], [(0.0018, 0.0016), (0.0018, 0.0015), (0.0012, 0.0012)]), k=0.003, mirror=True)
        ex, ey, ez = self.EYE
        sub(E((ex, ey, ez), (0.0086, 0.0078, 0.0082)), k=0.0025, mirror=True)
        add(E((ex, ey, ez), (0.0072, 0.0072, 0.0072)), k=0.001, mirror=True)
        add(E((0.0175, 0.4045, 0.334), (0.009, 0.0032, 0.0075)), k=0.005, mirror=True)
        # Великі гострі вуха.
        eb, et = P('earL') * M, np.asarray(self.EAR_TIP) * M
        add(T([eb, eb + (et - eb) * 0.45, et], [(0.027, 0.0082), (0.0185, 0.0064), (0.0028, 0.0022)], lateral=(1, 0, 0.45)), k=0.01, mirror=True)
        sub(E(eb + (et - eb) * 0.38 + np.array([0.001, 0.002, 0.0062]), (0.0135, 0.026, 0.003), along(eb, et)), k=0.002, mirror=True)
        # Передні ноги.
        add(E((0.049, 0.188, 0.103), (0.015, 0.019, 0.016)), k=0.014, mirror=True)
        add(T([(0.05, 0.196, 0.12), (0.048, 0.132, 0.124), (0.046, 0.082, 0.128)], [(0.0165, 0.0215), (0.0128, 0.0158), (0.0105, 0.0118)], k=0.01), k=0.016, mirror=True)
        add(E((0.046, 0.073, 0.128), (0.0105, 0.0122, 0.012)), k=0.006, mirror=True)
        add(T([(0.046, 0.073, 0.128), (0.045, 0.03, 0.136)], [(0.0098, 0.0108), (0.0106, 0.011)]), k=0.01, mirror=True)
        self.claws += canid_paw(vol, self.PAW_F, 0.54)
        # Задні ноги.
        add(E((0.049, 0.205, -0.123), (0.018, 0.023, 0.023)), k=0.016, mirror=True)
        add(T([(0.05, 0.21, -0.133), (0.048, 0.15, -0.172), (0.046, 0.092, -0.21)], [(0.0185, 0.031), (0.0135, 0.0215), (0.0092, 0.0125)], k=0.01), k=0.02, mirror=True)
        add(RC((0.046, 0.15, -0.185), (0.046, 0.088, -0.226), 0.0072, 0.0066), k=0.01, mirror=True)
        add(T([(0.046, 0.088, -0.216), (0.044, 0.03, -0.207)], [(0.0094, 0.012), (0.0104, 0.0112)]), k=0.01, mirror=True)
        self.claws += canid_paw(vol, self.PAW_B, 0.5)
        # Пухнастий хвіст («труба»).
        add(T([(0, 0.307, -0.21), (0, 0.292, -0.255), (0, 0.262, -0.315), (0, 0.218, -0.385), (0, 0.172, -0.455), (0, 0.135, -0.522), (0, 0.112, -0.582)],
              [(0.021, 0.022), (0.036, 0.038), (0.05, 0.053), (0.056, 0.059), (0.054, 0.057), (0.042, 0.044), (0.01, 0.01)], k=0.02), k=0.03)
        vol.apply(HalfSpace((0, 0, 0), (0, -1, 0), (-0.2, -0.01, -0.7), (0.2, 0.012, 0.5)), 'inter', 0.0)

    def displace(self, v, n):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        tail = sm(-0.21, -0.25, z)
        cheek = sm(0.31, 0.29, z) * sm(0.27, 0.285, z) * sm(0.4, 0.37, y) * sm(0.34, 0.35, y) * sm(0.022, 0.034, np.abs(x))
        bib = sm(0.2, 0.22, z) * sm(0.27, 0.25, z) * sm(0.33, 0.3, y) * sm(0.22, 0.25, y)
        pants = sm(-0.18, -0.21, z) * sm(0.19, 0.23, y) * sm(0.3, 0.27, y)
        neck = sm(0.15, 0.18, z) * sm(0.27, 0.24, z) * sm(0.3, 0.35, y)
        amp = 0.0007 + 0.0032 * tail + 0.0018 * cheek + 0.0014 * bib + 0.0014 * pants + 0.0012 * neck
        clump = stretch_noise(v, 90.0, (1.0, 0.45, 0.35), 3) * 0.65 + vnoise(v, 220.0, 5) * 0.35
        return amp * (clump - 0.45) * 2.0

    def weight_setup(self, v, B):
        return quad_weights(v, B, 0.5, self.NOSE, self.CHIN, self.TAIL_TIP, self.EAR_TIP, self.TOE_F, self.TOE_B,
                            self.mouth_y, 0.3, 0.335, 0.405, -0.205, 0.06, 0.165, (0, 0.3, -0.22), self.weight_iters, jaw_xmax=0.026, tail_r=0.07)

    def paint(self, v, n, lab, ao, B):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        ax = np.abs(x)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        names = self.bone_names
        bl = np.array([names[i] for i in lab])
        isin = lambda *a: np.isin(bl, a).astype(float)  # noqa: E731
        lowF = isin('radFL', 'radFR', 'canFL', 'canFR', 'pasFL', 'pasFR')
        lowB = isin('metBL', 'metBR', 'pasBL', 'pasBR')
        tib = isin('tibBL', 'tibBR')
        ispaw = isin('pasFL', 'pasFR', 'pasBL', 'pasBR')
        istail = isin('tail1', 'tail2', 'tail3')
        isear = isin('earL', 'earR')
        head = ((z > 0.27) & (y > 0.33)).astype(float)
        g1 = fbm(v, 7.0, 2, 1)
        g2 = vnoise(v, 120.0, 7)
        streak = stretch_noise(v, 30.0, (1.0, 0.6, 0.35), 11)
        # Рудувато-вогняна основа; темніша спина з сивими остюками.
        col = mix(lin(0xb0643a), lin(0x9e5631), g1)
        back = sm(0.0, 0.8, ny) * sm(0.29, 0.33, y) * (1 - head)
        col = mix(col, mix(lin(0x8e4a25), lin(0x7a4a33), sm(0.5, 0.8, streak)), back * 0.55)
        flank = sm(0.25, -0.3, ny) * sm(0.2, 0.28, y) * (1 - head)
        col = mix(col, lin(0xbb7444), flank * 0.35)
        # Білий низ: підборіддя, горло, груди; світло-сірий живіт; білі внутрішні сторони стегон.
        under = sm(-0.3, -0.8, ny) * sm(0.3, 0.24, y)
        throat = sm(0.17, 0.22, z) * sm(0.31, 0.27, z) * sm(0.36, 0.25, y) * sm(0.3, -0.25, ny) * sm(-0.6, 0.15, nz) * (1 - head)
        chest = sm(0.13, 0.19, z) * sm(0.29, 0.21, y) * sm(-0.2, 0.45, nz)
        light = np.clip(throat + chest * 0.9, 0, 1)
        col = mix(col, lin(0xd8d0c4), under * 0.75)
        col = mix(col, lin(0xece6dc), light * 0.92)
        # «Панчохи»: чорні ноги від ліктя й скакального суглоба, перед передньої лапи чорний вище.
        stock = np.clip(lowF * sm(0.2, 0.15, y) + lowF * sm(0.2, 0.7, nz) + lowB + tib * sm(0.12, 0.09, y), 0, 1)
        col = mix(col, mix(lin(0x231c18), lin(0x34271f), g2), stock * 0.92)
        # Голова: руда маківка й спинка носа, білі щоки й губи, темні «сльозинки» й вуса, чорний ніс.
        col = mix(col, lin(0xa95f36), head * sm(0.1, 0.7, ny) * 0.6)
        cheeks = head * sm(0.3, 0.335, z) * sm(0.3, -0.35, ny) * sm(0.385, 0.37, y) * sm(0.6, 0.0, nz)
        lips = head * sm(0.33, 0.36, z) * sm(0.352, 0.36, y) * sm(0.374, 0.364, y)
        jawc = head * (bl == 'jaw')
        col = mix(col, lin(0xeae3d8), np.clip(cheeks * 0.95 + lips * 0.85 + jawc * 0.85 * sm(0.4, 0.385, z), 0, 1))
        col = mix(col, lin(0x3b2a20), jawc * sm(0.385, 0.398, z) * 0.8)
        tear = head * np.exp(-((ax - 0.016) ** 2 / 0.0035 ** 2 + (y - 0.386) ** 2 / 0.004 ** 2 + (z - 0.348) ** 2 / 0.009 ** 2))
        col = mix(col, lin(0x2e221b), tear * 0.65)
        whisk = head * np.exp(-((ax - 0.011) ** 2 / 0.004 ** 2 + (y - 0.364) ** 2 / 0.0035 ** 2 + (z - 0.385) ** 2 / 0.008 ** 2))
        col = mix(col, lin(0x2e241e), whisk * 0.55)
        lipline = head * sm(0.32, 0.345, z) * np.exp(-((y - self.mouth_y(z)) / 0.0042) ** 2)
        col = mix(col, lin(0x1d1714), lipline * 0.9)
        nose = sm(0.016, 0.0095, np.linalg.norm((v - [0, 0.368, 0.408]) * [1.0, 1.25, 1.0], axis=1)) * sm(0.401, 0.408, z)
        col = mix(col, lin(0x121010), nose)
        rough = np.full(len(v), 0.9)
        col, rough = self._fox_eyes(v, col, rough)
        # Вуха: чорні ззаду, білі всередині, руда основа.
        earfront = isear * sm(-0.1, 0.45, nz)
        earback = isear * (1 - earfront) * sm(0.43, 0.445, y)
        col = mix(col, lin(0x1c1714), earback * 0.95)
        col = mix(col, lin(0xcfc4b3), earfront * sm(0.425, 0.44, y) * 0.85)
        # Хвіст: рудувато-сірий із темними остюками зверху, білий кінчик.
        t1y, tty = self.B['tail1'][1][1], self.TAIL_TIP[1]
        tz0, tz1 = self.B['tail1'][1][2], self.TAIL_TIP[2]
        tt = np.clip((z - tz0) / (tz1 - tz0), 0, 1)
        tailtop = istail * sm(-0.2, 0.5, ny * 0.6 - nz * 0.4)
        col = mix(col, mix(lin(0x9c5a34), lin(0x87583c), streak), istail * 0.6)
        col = mix(col, lin(0x4a3428), istail * tailtop * sm(0.4, 0.75, streak) * 0.55)
        col = mix(col, lin(0xc9b8a4), istail * (1 - tailtop) * 0.35)
        col = mix(col, lin(0xf0ebe1), istail * sm(0.8, 0.88, tt))
        pad = ispaw * sm(0.007, 0.0015, y)
        col = mix(col, lin(0x231e1b), pad * 0.9)
        claw = claw_mask(v, self.claws, 0.0045)
        col = mix(col, lin(0x2c2824), claw)
        col = col * (0.55 + 0.45 * ao)[:, None] * (0.94 + 0.12 * g2)[:, None]
        rough = np.where(nose > 0.5, 0.4, rough)
        rough = np.where(pad > 0.5, 0.75, rough)
        return col, rough

    def _fox_eyes(self, v, col, rough):
        """Очі лисиці: бурштинова райдужка з вертикальною щілиною зіниці."""
        r = 0.0072
        for sg in (-1, 1):
            c = np.array([abs(self.EYE[0]) * sg, self.EYE[1], self.EYE[2]])
            d = np.linalg.norm(v - c, axis=1)
            eye = sm(r * 1.17, r * 1.02, d)
            g = np.array([0.55 * sg, 0.08, 0.83])
            g /= np.linalg.norm(g)
            q = (v - c) / np.maximum(d, 1e-6)[:, None]
            cosg = q @ g
            # Вертикальна щілина: відхилення вбік (у площині ока) мале, вгору-вниз — більше.
            side = np.cross(g, [0, 1, 0])
            side /= np.linalg.norm(side)
            lat = np.abs(q @ side)
            vert = np.abs(q[:, 1] - g[1] * cosg)
            iris = mix(lin(0xd39b2b), lin(0x8a5a18), sm(0.93, 0.72, cosg))
            slit = sm(0.12, 0.06, lat) * sm(0.55, 0.35, vert) * sm(0.7, 0.8, cosg)
            ecol = mix(iris, lin(0x070605), slit)
            ecol = mix(lin(0x2a1d12), ecol, sm(0.5, 0.7, cosg))
            col = mix(col, ecol, eye)
            rim = sm(r + 0.003, r + 0.0008, d) * (1 - eye)
            col = mix(col, lin(0x1a1410), rim * 0.9)
            rough = np.where(eye > 0.5, 0.4, rough)
        return col, rough
