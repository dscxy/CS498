###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for s in range(world - 1):
        send_idx = (rank - s) % world
        recv_idx = (rank - s - 1) % world
        send_s = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_r = dist.irecv(tensor=tmp, src=left)
        recv_r.wait()
        send_s.wait()
        chunks[recv_idx].add_(tmp)
    return (rank + 1) % world
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    cur = current
    for s in range(world - 1):
        send_idx = cur
        recv_idx = (cur - 1) % world
        send_r = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_r = dist.irecv(tensor=tmp, src=left)
        recv_r.wait()
        send_r.wait()
        chunks[recv_idx].copy_(tmp)
        cur = recv_idx

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_num = chunk * world
    padded_flat = torch.zeros(padded_num, dtype=flat.dtype, device=flat.device)
    if n > 0:
        padded_flat[:n].copy_(flat)
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])
    current = reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, current, world, rank, left, right)
    
    # stitch & unpad  
    padded_flat /= world
    tensor.view(-1).copy_(padded_flat[:n])
    return