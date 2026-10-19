import os
import sys

# torch.compile/Triton needs a C compiler; scripts run the env's python without `conda activate`, so point at conda-forge gcc
_cc = os.path.join(os.path.dirname(sys.executable), 'x86_64-conda-linux-gnu-gcc')
if os.path.exists(_cc):
    os.environ.setdefault('CC', _cc)
