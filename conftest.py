# The exactness checks are many small matrix products; threaded BLAS only gets in the way
# (and in the way of whatever else the machine is doing).
import os

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
