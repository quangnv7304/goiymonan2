import sys
import os

print('DIAG: sys.executable=', sys.executable)
print('DIAG: sys.version=', sys.version)
print('DIAG: cwd=', os.getcwd())
print('DIAG: files=', os.listdir('.'))
