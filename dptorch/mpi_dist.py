import os
from typing import Any, List

import numpy as np
import importlib

try:
	MPI = importlib.import_module("mpi4py_dist")

	# mpi4py_dist may import successfully even if mpi4py is unavailable.
	# In distributed mode, force fallback to torch_dist in that case.
	if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1 and getattr(MPI, "MPI", None) is None:
		raise ImportError("mpi4py is not available in mpi4py_dist")
except ImportError:
	print(
		"mpi4py backend unavailable, falling back to torch_dist; note that torch_dist requires custom build of PyTorch with MPI support."
	)
	MPI = importlib.import_module("torch_dist")

dist = MPI._MPIDistCompat()