"""Записує дані моделей звірів (WHD1, base64) у модуль гри src/07d-beast-data.js."""
import os
import sys
OUT = os.environ.get('BEAST_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, '07d-beast-data.js')
names = ['wolf', 'fox', 'hare', 'boar']
lines = ['// Дані сіток звірів (формат WHD1, base64): створено генератором анатомії beast-gen (див. розділ 11c).',
         'const BEAST_MODEL_DATA = {']
for nm in names:
    b64 = open(os.path.join(OUT, nm + '.b64')).read().strip()
    lines.append("  %s: '%s'," % (nm, b64))
lines.append('};')
open(DST, 'w').write('\n'.join(lines) + '\n')
print('written', DST, os.path.getsize(DST))
