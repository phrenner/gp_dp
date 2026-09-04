import os
from typing import Any, List

import numpy as np
import torch

try:
    from mpi4py import MPI
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
                "mpi4py is required for distributed runs. Install mpi4py and rerun."
            )

        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_rank(self) -> int:
        if MPI is None:
            return 0
        return MPI.COMM_WORLD.Get_rank()

    def get_world_size(self) -> int:
        if MPI is None:
            return 1
        return MPI.COMM_WORLD.Get_size()

    def barrier(self):
        if MPI is None:
            return
        MPI.COMM_WORLD.Barrier()

    def broadcast_object_list(self, object_list: List[Any], src: int = 0):
        if MPI is None:
            return
        comm = MPI.COMM_WORLD
        for i in range(len(object_list)):
            payload = object_list[i] if comm.Get_rank() == src else None
            object_list[i] = comm.bcast(payload, root=src)

    def broadcast_constant(self, value: torch.Tensor, src: int = 0) -> Any:
        if MPI is None:
            return value
        comm = MPI.COMM_WORLD
        return torch.tensor(comm.bcast(value.detach().cpu().numpy(), root=src))
    
    def reduce_mean(self, value: torch.Tensor, dst: int = 0) -> torch.Tensor:
        if MPI is None:
            return value
        comm = MPI.COMM_WORLD
        value_np = value.detach().cpu().numpy()
        mean_value = np.zeros_like(value_np)
        comm.Allreduce(value_np, mean_value, op=MPI.SUM)
        mean_value /= comm.Get_size()
        return torch.tensor(mean_value)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if MPI is None:
            return tensor

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == src:
            arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
            meta = (arr.shape, arr.dtype.str)
        else:
            arr = None
            meta = None

        # Ensure all ranks allocate the exact sender buffer shape/dtype before Bcast.
        shape, dtype_str = comm.bcast(meta, root=src)
        if rank != src:
            arr = np.empty(shape, dtype=np.dtype(dtype_str))

        comm.Bcast(arr, root=src)

        if rank == src:
            return tensor

        recv_tensor = torch.from_numpy(arr).to(dtype=tensor.dtype, device=tensor.device)
        if tensor.shape == recv_tensor.shape:
            tensor.copy_(recv_tensor)
            return tensor
        return recv_tensor

    def all_gather(self, output_tensors: List[torch.Tensor], input_tensor: torch.Tensor):
        if MPI is None:
            if len(output_tensors) == 1:
                output_tensors[0].copy_(input_tensor)
            return

        comm = MPI.COMM_WORLD
        send_arr = np.ascontiguousarray(input_tensor.detach().cpu().numpy())
        recv_arrs = comm.allgather(send_arr)

        for i, arr in enumerate(recv_arrs):
            output_tensors[i].copy_(
                torch.from_numpy(np.ascontiguousarray(arr)).to(
                    dtype=output_tensors[i].dtype,
                    device=output_tensors[i].device,
                )
            )



