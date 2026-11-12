import sys, importlib
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beast import build, write_bundle
names = sys.argv[1].split(',')
for nm in names:
    mod = importlib.import_module(nm)
    cls = [getattr(mod, a) for a in dir(mod) if isinstance(getattr(mod, a), type) and getattr(getattr(mod, a), 'name', None) == nm][0]
    build(cls())
if len(sys.argv) > 2:
    write_bundle(sys.argv[2].split(','))
