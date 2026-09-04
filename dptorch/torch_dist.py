import os
from typing import Any, List

import numpy as np
import torch

try:
    import torch.distributed as MPI
except ImportError:
    MPI = None


class _MPIDistCompat:
    def __init__(self):
        self._initialized = False

    def init_process_group(self, backend: str = "mpi"):
        if backend != "mpi":
            raise ValueError(f"Only backend='mpi' is supported, got backend='{backend}'")

        if MPI is None and int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
            raise ImportError(
                "torch distributed with MPI is required for distributed runs."
            )
        
        MPI.init_process_group(backend="mpi")

        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_rank(self) -> int:
        if MPI is None:
            return 0
        return MPI.get_rank()

    def get_world_size(self) -> int:
        if MPI is None:
            return 1
        return MPI.get_world_size()

    def barrier(self):
        if MPI is None:
            return
        MPI.barrier()

    def broadcast_object_list(self, object_list: List[Any], src: int = 0):
        if MPI is None:
            return
        MPI.broadcast_object_list(object_list, src=src)

    def broadcast_constant(self, value: torch.Tensor, src: int = 0) -> Any:
        if MPI is None:
            return value
        MPI.broadcast(value, src=src)
        return value

    def reduce_mean(self, value: torch.Tensor, dst: int = 0) -> torch.Tensor:
        if MPI is None:
            return value
        MPI.all_reduce(value, op=MPI.ReduceOp.SUM)
        value = value / MPI.get_world_size()
        return value

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if MPI is None:
            return tensor

        MPI.broadcast(tensor, src)
        return tensor

    def all_gather(self, output_tensors: List[torch.Tensor], input_tensor: torch.Tensor):
        if MPI is None:
            return

        MPI.all_gather(output_tensors, input_tensor)

