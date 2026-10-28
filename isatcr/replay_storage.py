"""Lossless immutable storage for native CPU replay graphs.

Only activated for accelerated native GAT/GraphSAGE/JKNet training. Network
parameters, replay capacity, index order, sampling and optimizer stay unchanged.
"""
import ctypes
from pathlib import Path
import sys
import weakref
import numpy as np
import torch
from torch_geometric.data import Batch, Data

SCHEMAS = {}

class PackedGraph:
    __slots__ = ('payload', 'schema')

    def __init__(self, data):
        fields = data.to_dict()
        if type(data) is not Data or any(
                not torch.is_tensor(t) or t.device.type != 'cpu' or t.requires_grad
                for t in fields.values()):
            raise ValueError('Only detached CPU tensor replay graphs are supported')
        arrays = [(name, t.detach().contiguous().numpy()) for name, t in fields.items()]
        schema, offset, pieces = [], 0, []
        for name, array in arrays:
            payload = array.tobytes()
            schema.append((name, array.dtype.str, array.shape, offset, len(payload)))
            offset += len(payload)
            pieces.append(payload)
        key = tuple(schema)
        self.schema = SCHEMAS.setdefault(key, key)
        self.payload = b''.join(pieces)

    def arrays(self):
        return {name: np.frombuffer(self.payload, dtype=dtype,
                                   count=size // np.dtype(dtype).itemsize,
                                   offset=offset).reshape(shape)
                for name, dtype, shape, offset, size in self.schema}

def collate(graphs):
    stores = [g.arrays() for g in graphs]
    nodes = np.asarray([s['x'].shape[0] for s in stores], dtype=np.int64)
    ptr = np.concatenate((np.zeros(1, dtype=np.int64), nodes.cumsum()))
    batch, slices, increments = Batch(), {}, {}
    for name in stores[0]:
        index = name in ('message_edge_index1', 'message_edge_index2')
        dim = 1 if index else 0
        arrays = [s[name] for s in stores]
        sizes = np.asarray([a.shape[dim] for a in arrays], dtype=np.int64)
        joined = np.concatenate(arrays, axis=dim)
        slices[name] = torch.from_numpy(np.concatenate((np.zeros(1, dtype=np.int64), sizes.cumsum())))
        if index:
            joined += np.repeat(ptr[:-1], sizes)[None, :]
            increments[name] = torch.from_numpy(ptr[:-1].copy())
        else:
            increments[name] = torch.zeros(len(graphs), dtype=torch.long)
        batch[name] = torch.from_numpy(joined)
    batch.batch = torch.from_numpy(np.repeat(np.arange(len(nodes), dtype=np.int64), nodes))
    batch.ptr = torch.from_numpy(ptr)
    batch._num_graphs = len(graphs)
    batch._slice_dict, batch._inc_dict = slices, increments
    return batch

LIBRARY = Path(sys.prefix)/'Library/bin/liblz4.dll'

library = ctypes.CDLL(str(LIBRARY))

library.LZ4_compressBound.argtypes = [ctypes.c_int]

library.LZ4_compressBound.restype = ctypes.c_int

library.LZ4_compress_default.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

library.LZ4_compress_default.restype = ctypes.c_int

library.LZ4_decompress_safe.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

library.LZ4_decompress_safe.restype = ctypes.c_int

class LZ4Graph(PackedGraph):
    __slots__ = ('raw_size',)

    def __init__(self, data):
        super().__init__(data)
        self.raw_size = len(self.payload)
        capacity = library.LZ4_compressBound(self.raw_size)
        target = ctypes.create_string_buffer(capacity)
        size = library.LZ4_compress_default(self.payload, target, self.raw_size, capacity)
        if size <= 0:
            raise RuntimeError('LZ4 compression failed')
        self.payload = target.raw[:size]

    def arrays(self):
        target = ctypes.create_string_buffer(self.raw_size)
        size = library.LZ4_decompress_safe(self.payload, target, len(self.payload), self.raw_size)
        if size != self.raw_size:
            raise RuntimeError('LZ4 replay length mismatch')
        payload = target.raw
        return {name: np.frombuffer(payload, dtype=dtype,
                                   count=size // np.dtype(dtype).itemsize,
                                   offset=offset).reshape(shape)
                for name, dtype, shape, offset, size in self.schema}

class MemoizedLZ4Graph(LZ4Graph):
    __slots__ = ('_initialized',)
    cache = {}
    counts = dict(hits=0, misses=0, changed=0)

    def __new__(cls, data=None):
        if data is None:  # Pickle restores slots without a source graph.
            return object.__new__(cls)
        signature = tuple((name,id(t),t._version,t.data_ptr(),t.dtype,t.device,
                           tuple(t.shape),tuple(t.stride())) for name,t in data.to_dict().items())
        key = id(data)
        cached = cls.cache.get(key)
        if cached is not None and cached[0]() is data and cached[1] == signature:
            cls.counts['hits'] += 1
            return cached[2]
        obj = object.__new__(cls)
        obj._initialized = False
        def expired(ref):
            current = cls.cache.get(key)
            if current is not None and current[0] is ref:
                cls.cache.pop(key,None)
        cls.cache[key] = (weakref.ref(data,expired),signature,obj)
        cls.counts['misses'] += 1
        cls.counts['changed'] += cached is not None
        return obj

    def __init__(self, data):
        if self._initialized:
            return
        try:
            super().__init__(data)
        except BaseException:
            self.cache.pop(id(data),None)
            raise
        self._initialized = True

def pack_experiences(experiences):
    packed = []
    for experience in experiences:
        fields = list(experience)
        for index in (0, 4):
            graph, other = fields[index]
            fields[index] = [MemoizedLZ4Graph(graph), other]
        packed.append(tuple(fields))
    return packed
