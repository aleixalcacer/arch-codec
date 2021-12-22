# Hey Cython, this is Python 3!
# cython: language_level=3

#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import msgpack
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from libc.stdint cimport uintptr_t
from libc.string cimport strdup
from cpython cimport (
    PyObject_GetBuffer, PyBuffer_Release,
    PyBUF_SIMPLE, PyBUF_WRITABLE, Py_buffer,
    PyBytes_FromStringAndSize
)
from .utils import Codec, Filter
import os.path

cdef extern from "<stdint.h>":
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long long uint64_t


cdef extern from "blosc2.h":
    ctypedef enum:
        BLOSC_NOFILTER
        BLOSC_SHUFFLE
        BLOSC_BITSHUFFLE
        BLOSC_DELTA
        BLOSC_TRUNC_PREC
        BLOSC_BLOSCLZ
        BLOSC_LZ4
        BLOSC_LZ4HC
        BLOSC_ZLIB
        BLOSC_ZSTD
        BLOSC2_MAX_FILTERS
        BLOSC2_MAX_METALAYERS
        BLOSC2_MAX_VLMETALAYERS
        BLOSC_MAX_OVERHEAD
        BLOSC_ALWAYS_SPLIT = 1,
        BLOSC_NEVER_SPLIT = 2,
        BLOSC_AUTO_SPLIT = 3,
        BLOSC_FORWARD_COMPAT_SPLIT = 4,

    ctypedef int *blosc2_prefilter_fn
    ctypedef struct blosc2_prefilter_params
    ctypedef int *blosc2_postfilter_fn
    ctypedef struct blosc2_postfilter_params

    ctypedef struct blosc2_cparams:
        uint8_t compcode
        uint8_t compcode_meta
        uint8_t clevel
        int use_dict
        int32_t typesize
        int16_t nthreads
        int32_t blocksize
        int32_t splitmode
        void * schunk
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params * preparams
        blosc2_btune *udbtune

    ctypedef struct blosc2_dparams:
        int nthreads
        void * schunk
        blosc2_postfilter_fn postfilter
        blosc2_postfilter_params *postparams

    ctypedef struct blosc2_storage
    ctypedef struct blosc2_btune
    ctypedef struct blosc2_context
    ctypedef struct blosc2_frame
    ctypedef struct blosc2_metalayer:
        char *name;
        uint8_t *content;
        int32_t content_len;

    ctypedef struct blosc2_schunk:
        uint8_t version;
        uint8_t compcode;
        uint8_t compcode_meta;
        uint8_t clevel;
        int32_t typesize;
        int32_t blocksize;
        int32_t chunksize;
        uint8_t filters[BLOSC2_MAX_FILTERS];
        uint8_t filters_meta[BLOSC2_MAX_FILTERS];
        int32_t nchunks;
        int64_t nbytes;
        int64_t cbytes;
        uint8_t** data;
        size_t data_len;
        blosc2_storage *storage;
        blosc2_frame *frame;
        blosc2_context *cctx;
        blosc2_context *dctx;
        blosc2_metalayer *metalayers[BLOSC2_MAX_METALAYERS];
        int16_t nmetalayers;
        blosc2_metalayer *vlmetalayers[BLOSC2_MAX_VLMETALAYERS];
        blosc2_btune *udbtune;

    int blosc2_meta_exists(blosc2_schunk *schunk, const char *name)
    int blosc2_meta_add(blosc2_schunk *schunk, const char *name, uint8_t *content,
                                 uint32_t content_len)
    int blosc2_meta_update(blosc2_schunk *schunk, const char *name, uint8_t *content,
                                    uint32_t content_len)
    int blosc2_meta_get(blosc2_schunk *schunk, const char *name, uint8_t ** content,
                    uint32_t *content_len)

    ctypedef int(*blosc2_codec_encoder_cb)(const uint8_t *input, int32_t input_len, uint8_t *output, int32_t output_len,
                                          uint8_t meta, blosc2_cparams *cparams);
    ctypedef int(*blosc2_codec_decoder_cb)(const uint8_t *input, int32_t input_len, uint8_t *output, int32_t output_len,
                                          uint8_t meta, blosc2_dparams *dparams);

    ctypedef struct blosc2_codec:
        uint8_t compcode;
        char * compname;
        uint8_t complib;
        uint8_t compver;
        blosc2_codec_encoder_cb encoder;
        blosc2_codec_decoder_cb decoder;

    int blosc2_register_codec(blosc2_codec * codec);



cdef extern from "caterva.h":
    ctypedef enum:
        CATERVA_MAX_DIM
        CATERVA_MAX_METALAYERS

    ctypedef struct caterva_config_t:
        void *(*alloc)(size_t)
        void (*free)(void *)
        uint8_t compcodec
        uint8_t compmeta
        uint8_t complevel
        int32_t splitmode
        int usedict
        int16_t nthreads
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filtersmeta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params *pparams
        blosc2_btune *udbtune;

    ctypedef struct caterva_ctx_t:
        caterva_config_t *cfg

    ctypedef struct caterva_metalayer_t:
        char *name
        uint8_t *sdata
        int32_t size

    ctypedef struct caterva_storage_t:
        int32_t chunkshape[CATERVA_MAX_DIM]
        int32_t blockshape[CATERVA_MAX_DIM]
        bool sequencial
        char* urlpath
        caterva_metalayer_t metalayers[CATERVA_MAX_METALAYERS]
        int32_t nmetalayers

    ctypedef struct caterva_params_t:
        int64_t shape[CATERVA_MAX_DIM]
        uint8_t ndim
        uint8_t itemsize


    cdef struct chunk_cache_s:
        uint8_t *data
        int32_t nchunk

    ctypedef struct caterva_array_t:
        blosc2_schunk *sc;
        uint8_t *buf;
        int64_t shape[CATERVA_MAX_DIM];
        int32_t chunkshape[CATERVA_MAX_DIM];
        int64_t extshape[CATERVA_MAX_DIM];
        int32_t blockshape[CATERVA_MAX_DIM];
        int64_t extchunkshape[CATERVA_MAX_DIM];
        int64_t nitems;
        int32_t chunknitems;
        int64_t extnitems;
        int32_t blocknitems;
        int64_t extchunknitems;
        uint8_t ndim;
        uint8_t itemsize;
        int64_t nchunks;
        chunk_cache_s chunk_cache;

    int caterva_ctx_new(caterva_config_t *cfg, caterva_ctx_t **ctx);
    int caterva_ctx_free(caterva_ctx_t **ctx);
    int caterva_empty(caterva_ctx_t *ctx, caterva_params_t *params,
                      caterva_storage_t *storage, caterva_array_t ** array);
    int caterva_zeros(caterva_ctx_t *ctx, caterva_params_t *params,
                      caterva_storage_t *storage, caterva_array_t ** array);
    int caterva_full(caterva_ctx_t *ctx, caterva_params_t *params,
                     caterva_storage_t *storage, void *fill_value, caterva_array_t ** array);
    int caterva_free(caterva_ctx_t *ctx, caterva_array_t ** array);
    int caterva_from_schunk(caterva_ctx_t *ctx, blosc2_schunk *schunk,
                            caterva_array_t **array);
    int caterva_from_serial_schunk(caterva_ctx_t *ctx, uint8_t *serial_schunk, int64_t len,
                                   caterva_array_t ** array);
    int caterva_open(caterva_ctx_t *ctx, const char *urlpath, caterva_array_t ** array);
    int caterva_from_buffer(caterva_ctx_t *ctx, void *buffer, int64_t buffersize,
                            caterva_params_t *params, caterva_storage_t *storage,
                            caterva_array_t ** array);
    int caterva_to_buffer(caterva_ctx_t *ctx, caterva_array_t *array, void *buffer,
                          int64_t buffersize);
    int caterva_get_slice(caterva_ctx_t *ctx, caterva_array_t *src, int64_t *start,
                          int64_t *stop, caterva_storage_t *storage, caterva_array_t ** array);
    int caterva_squeeze_index(caterva_ctx_t *ctx, caterva_array_t *array,
                              bool *index);
    int caterva_squeeze(caterva_ctx_t *ctx, caterva_array_t *array);
    int caterva_get_slice_buffer(caterva_ctx_t *ctx, caterva_array_t *array,
                                 int64_t *start, int64_t *stop,
                                 void *buffer, int64_t *buffershape, int64_t buffersize);
    int caterva_set_slice_buffer(caterva_ctx_t *ctx,
                                 void *buffer, int64_t *buffershape, int64_t buffersize,
                                 int64_t *start, int64_t *stop, caterva_array_t *array);
    int caterva_copy(caterva_ctx_t *ctx, caterva_array_t *src, caterva_storage_t *storage,
                     caterva_array_t ** array);


# Defaults for compression params
config_dflts = {
    'codec': Codec.LZ4,
    'codecmeta': 0,
    'clevel': 5,
    'usedict': False,
    'nthreads': 1,
    'filters': [Filter.SHUFFLE],
    'filtersmeta': [0],  # no actual meta info for SHUFFLE, but anyway...
    }


cdef class Context:
    cdef caterva_ctx_t *context_
    cdef uint8_t compcode
    cdef uint8_t compmeta
    cdef uint8_t complevel
    cdef int32_t splitmode
    cdef int usedict
    cdef int16_t nthreads
    cdef int32_t blocksize
    cdef uint8_t filters[BLOSC2_MAX_FILTERS]
    cdef uint8_t filtersmeta[BLOSC2_MAX_FILTERS]
    cdef blosc2_prefilter_fn prefilter
    cdef blosc2_prefilter_params* pparams

    def __init__(self, **kwargs):
        cdef caterva_config_t config
        config.free = free
        config.alloc = malloc
        config.compcodec = kwargs.get('codec', config_dflts['codec']).value
        config.compmeta = kwargs.get('codecmeta', config_dflts['codecmeta'])
        config.complevel = kwargs.get('clevel', config_dflts['clevel'])
        config.splitmode = BLOSC_AUTO_SPLIT
        config.usedict =  kwargs.get('usedict', config_dflts['usedict'])
        config.nthreads = kwargs.get('nthreads', config_dflts['nthreads'])
        config.prefilter = NULL
        config.pparams = NULL
        config.udbtune = NULL

        for i in range(BLOSC2_MAX_FILTERS):
            config.filters[i] = 0
            config.filtersmeta[i] = 0

        filters = kwargs.get('filters', config_dflts['filters'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters), BLOSC2_MAX_FILTERS):
            config.filters[i] = filters[i - BLOSC2_MAX_FILTERS + len(filters)].value

        filtersmeta = kwargs.get('filtersmeta', config_dflts['filtersmeta'])
        for i in range(BLOSC2_MAX_FILTERS - len(filtersmeta), BLOSC2_MAX_FILTERS):
            self.filtersmeta[i] = filtersmeta[i - BLOSC2_MAX_FILTERS + len(filtersmeta)]

        caterva_ctx_new(&config, &self.context_)

    def __dealloc__(self):
        caterva_ctx_free(&self.context_)

    def tocapsule(self):
        return PyCapsule_New(self.context_, "caterva_ctx_t*", NULL)


cdef create_caterva_params(caterva_params_t *params, shape, itemsize):
    params.ndim = len(shape)
    params.itemsize = itemsize
    for i in range(params.ndim):
        params.shape[i] = shape[i]


cdef create_caterva_storage(caterva_storage_t *storage, kwargs):
    chunks = kwargs.get('chunks', None)
    blocks = kwargs.get('blocks', None)
    urlpath = kwargs.get('urlpath', None)
    sequential = kwargs.get('sequential', False)
    meta = kwargs.get('meta', None)

    if not chunks:
        raise AttributeError("chunks must be specified")
    if not blocks:
        raise AttributeError("blocks must be specified")

    if urlpath is not None:
        urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath
        storage.urlpath = urlpath
    else:
        storage.urlpath = NULL
    storage.sequencial = sequential
    for i in range(len(chunks)):
        storage.chunkshape[i] = chunks[i]
        storage.blockshape[i] = blocks[i]

    if meta is None:
        storage.nmetalayers = 0
    else:
        storage.nmetalayers = len(meta)
        for i, (name, content) in enumerate(meta.items()):
            name2 = name.encode("utf-8") if isinstance(name, str) else name # do a copy
            storage.metalayers[i].name = strdup(name2)
            storage.metalayers[i].sdata = <uint8_t *> strdup(content)
            storage.metalayers[i].size = len(content)


cdef class NDArray:
    cdef caterva_array_t *array
    cdef kwargs
    cdef usermeta_len
    cdef cframe
    cdef Py_buffer *py_buf
    cdef Py_ssize_t bp_shape[CATERVA_MAX_DIM]
    cdef Py_ssize_t bp_strides[CATERVA_MAX_DIM]

    @property
    def shape(self):
        """The shape of this container."""
        return tuple([self.array.shape[i] for i in range(self.array.ndim)])

    @property
    def chunks(self):
        """The chunk shape of this container."""
        return tuple([self.array.chunkshape[i] for i in range(self.array.ndim)])

    @property
    def blocks(self):
        """The block shape of this container."""
        return tuple([self.array.blockshape[i] for i in range(self.array.ndim)])

    @property
    def cratio(self):
        """The compression ratio for this container."""
        return self.size / (self.array.sc.cbytes + BLOSC_MAX_OVERHEAD * self.nchunks)

    @property
    def clevel(self):
        """The compression level for this container."""
        return self.array.sc.clevel

    @property
    def codec(self):
        """The compression codec name for this container."""
        return Codec(self.array.sc.compcode)

    @property
    def filters(self):
        """The filters list for this container."""
        return [Filter(self.array.sc.filters[i]) for i in range(BLOSC2_MAX_FILTERS)]

    @property
    def itemsize(self):
        """The itemsize of this container."""
        return self.array.itemsize

    @property
    def chunksize(self):
        """The chunk size (in bytes) for this container."""
        return self.array.chunknitems * self.itemsize

    @property
    def blocksize(self):
        """The block size (in bytes) for this container."""
        return self.array.blocknitems * self.itemsize

    @property
    def size(self):
        """The size (in bytes) for this container."""
        return self.array.nitems * self.itemsize

    @property
    def nchunks(self):
        """The number of chunks in this container."""
        return int(self.array.extnitems / self.array.chunknitems)

    @property
    def ndim(self):
        """The number of dimensions of this container."""
        return self.array.ndim

    @property
    def c_array(self):
        return <uintptr_t> self.array

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.usermeta_len = 0
        self.cframe = False
        self.array = NULL
        self.py_buf = NULL

    def squeeze(self, **kwargs):
        ctx = Context(**kwargs)
        caterva_squeeze(ctx.context_, self.array)

    def to_buffer(self, **kwargs):
        ctx = Context(**kwargs)
        buffersize = self.size
        buffer = bytes(buffersize)
        caterva_to_buffer(ctx.context_, self.array, <void *> <char *> buffer, buffersize)
        return buffer

    def __dealloc__(self):
        if self.py_buf != NULL:
            PyBuffer_Release(self.py_buf)
            free(self.py_buf)

        if self.array != NULL:
            ctx = Context(**self.kwargs)
            caterva_free(ctx.context_, &self.array)


def get_slice_numpy(arr, NDArray src, key, mask, **kwargs):
    ctx = Context(**kwargs)
    ndim = src.ndim
    start, stop = key

    cdef int64_t[CATERVA_MAX_DIM] start_, stop_
    cdef int64_t buffersize_ = src.itemsize
    cdef int64_t[CATERVA_MAX_DIM] buffershape_
    for i in range(src.ndim):
        start_[i] = start[i]
        stop_[i] = stop[i]
        buffershape_[i] = stop_[i] - start_[i]
        buffersize_ *= buffershape_[i]

    buffershape = [sp - st for st, sp in zip(start, stop)]
    cdef int64_t buffersize = src.itemsize

    cdef Py_buffer view
    PyObject_GetBuffer(arr, &view, PyBUF_SIMPLE)

    cdef caterva_array_t *array_
    caterva_get_slice_buffer(ctx.context_, src.array, start_, stop_, <void *> view.buf, buffershape_, buffersize_)
    PyBuffer_Release(&view)

    return arr.squeeze()


def get_slice(NDArray arr, NDArray src, key, mask, **kwargs):
    ctx = Context(**kwargs)
    ndim = src.ndim
    start, stop = key

    cdef int64_t[CATERVA_MAX_DIM] start_, stop_

    for i in range(src.ndim):
        start_[i] = start[i]
        stop_[i] = stop[i]

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_get_slice(ctx.context_, src.array, start_, stop_, &storage_, &array_)

    cdef bool mask_[CATERVA_MAX_DIM]
    for i in range(src.ndim):
        mask_[i] = mask[i]

    caterva_squeeze_index(ctx.context_, array_, mask_)
    arr.array = array_
    return arr

def set_slice(NDArray dst, key, ndarray):
    ctx = Context(**dst.kwargs)
    ndim = dst.ndim
    start, stop = key
    interface = ndarray.__array_interface__
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(ndarray, buf, PyBUF_SIMPLE)

    cdef int64_t[CATERVA_MAX_DIM] buffershape_, start_, stop_
    for i in range(ndim):
        start_[i] = start[i]
        stop_[i] = stop[i]
        buffershape_[i] = stop[i] - start[i]

    caterva_set_slice_buffer(ctx.context_, buf.buf, buffershape_, buf.len, start_, stop_, dst.array)
    PyBuffer_Release(buf)
    return dst


def empty(NDArray arr, shape, itemsize, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, itemsize)

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_empty(ctx.context_, &params_, &storage_, &array_)
    arr.array = array_


def zeros(NDArray arr, shape, itemsize, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, itemsize)

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_zeros(ctx.context_, &params_, &storage_, &array_)
    arr.array = array_


def full(NDArray arr, shape, value, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, len(value))

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)
    cdef uint8_t *fill_value_ = <uint8_t *> value
    cdef caterva_array_t *array_
    caterva_full(ctx.context_, &params_, &storage_, fill_value_, &array_)
    arr.array = array_


def copy(NDArray arr, NDArray src, **kwargs):
    ctx = Context(**kwargs)
    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_copy(ctx.context_, src.array, &storage_, &array_)
    arr.array = array_
    return arr


def from_file(NDArray arr, urlpath, **kwargs):
    ctx = Context(**kwargs)

    urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath
    if not os.path.exists(urlpath):
        raise FileNotFoundError

    cdef caterva_array_t *array_
    caterva_open(ctx.context_, urlpath, &array_)
    arr.array = array_


def from_buffer(NDArray arr, buf, shape, itemsize, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, itemsize)

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_from_buffer(ctx.context_, <void*> <char *> buf, len(buf), &params_, &storage_, &array_)
    arr.array = array_


def asarray(NDArray arr, ndarray, **kwargs):
    ctx = Context(**kwargs)

    interface = ndarray.__array_interface__
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(ndarray, buf, PyBUF_SIMPLE)

    shape = interface["shape"]
    itemsize = buf.itemsize

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, itemsize)

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_from_buffer(ctx.context_, <void*> <char *> buf.buf, buf.len, &params_, &storage_, &array_)
    arr.array = array_
    arr.py_buf = buf


def meta__contains__(self, name):
    cdef caterva_array_t *array = <caterva_array_t *><uintptr_t> self.c_array
    name = name.encode("utf-8") if isinstance(name, str) else name
    n = blosc2_meta_exists(array.sc, name)
    return False if n < 0 else True

def meta__getitem__(self, name):
    cdef caterva_array_t *array = <caterva_array_t *><uintptr_t> self.c_array
    name = name.encode("utf-8") if isinstance(name, str) else name
    cdef uint8_t *content
    cdef uint32_t content_len
    n = blosc2_meta_get(array.sc, name, &content, &content_len)
    return PyBytes_FromStringAndSize(<char *> content, content_len)

def meta__setitem__(self, name, content):
    cdef caterva_array_t *array = <caterva_array_t *><uintptr_t> self.c_array
    name = name.encode("utf-8") if isinstance(name, str) else name
    old_content = meta__getitem__(self, name)
    if len(old_content) != len(content):
        raise ValueError("The length of the content in a metalayer cannot change.")
    n = blosc2_meta_update(array.sc, name, content, len(content))
    return n

def meta__len__(self):
    cdef caterva_array_t *arr = <caterva_array_t *><uintptr_t> self.c_array
    return arr.sc.nmetalayers

def meta_keys(self):
    cdef caterva_array_t *arr = <caterva_array_t *><uintptr_t> self.c_array
    keys = []
    for i in range(meta__len__(self)):
        name = arr.sc.metalayers[i].name.decode("utf-8")
        keys.append(name)
    return keys


# AACODEC

import ctypes
import archetypes as arch

def split(array: np.ndarray, tile: tuple):
    height, width = array.shape
    t_height, t_width = tile

    tiled_array = array.reshape((height // t_height, t_height,
                                width // t_width, t_width))
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array.reshape(array.shape)


def split_backward(array: np.ndarray, tile: tuple):
    height, width = array.shape
    t_height, t_width = tile

    tiled_array = array.reshape((height // t_height,
                                 width // t_width,
                                 t_height,
                                 t_width))

    tiled_array = tiled_array.swapaxes(2, 1)
    return tiled_array.reshape(array.shape)


def quantizer(data):
    data = (np.digitize(data, np.linspace(0, 1, 256)) - 1)
    data = data.astype(np.uint8)
    return data

from dataclasses import dataclass

@dataclass
class CatervaMeta:
    version: int
    ndim: int
    shape: list
    chunks: list
    blocks: list

    def __init__(self, version, ndim, shape, chunks, blocks):
        self.version = version
        self.ndim = ndim
        self.shape = shape
        self.chunks = chunks
        self.blocks = blocks


cdef get_caterva_metalayer(blosc2_schunk *sc):
    name = "caterva".encode("utf8")
    cdef uint8_t *metalayer
    cdef uint32_t metalayer_len
    blosc2_meta_get(sc, <char *> name, &metalayer, &metalayer_len)
    cdef uint8_t [:] meta= <uint8_t[:metalayer_len]> metalayer
    version, ndim, shape, chunks, blocks = msgpack.unpackb(memoryview(meta))
    return CatervaMeta(version, ndim, shape, chunks, blocks)

# Encoder
cdef int encoder(const uint8_t *input, int32_t input_len, uint8_t *output,
             int32_t output_len, uint8_t meta, blosc2_cparams *cparams) with gil:
    cdef blosc2_schunk *sc = <blosc2_schunk *> cparams.schunk
    meta_caterva = get_caterva_metalayer(sc)

    dtype = np.float64 if cparams.typesize == 8 else np.float32

    cdef uint8_t [:] input_ = <uint8_t[:input_len]> input
    ndinput = np.frombuffer(bytes(input_), dtype=dtype).reshape(meta_caterva.blocks)
    ndinput = split(ndinput, (4, 4))

    cdef uint8_t [:] output_ = <uint8_t[:output_len]> output
    mvoutput = memoryview(output_)

    narch = meta

    size = 4 * 4

    mvoutput[0] = narch
    mvoutput[1] = size
    mvoutput[2] = ndinput.dtype.itemsize

    aa = arch.AA(n_archetypes=narch, n_init=1, max_iter=100)

    ndinput = ndinput.reshape(-1, size)
    data_trans = aa.fit_transform(ndinput)
    data_trans = quantizer(data_trans)

    archetypes = aa.archetypes_
    archetypes_ind = 3
    mvoutput[archetypes_ind: archetypes_ind + archetypes.nbytes] = bytes(archetypes)
    data_trans_ind = archetypes_ind + archetypes.nbytes

    mvoutput[data_trans_ind: data_trans_ind + data_trans.nbytes] = bytes(data_trans)

    return data_trans_ind + data_trans.nbytes


cdef int decoder(const uint8_t *input, int32_t input_len, uint8_t *output,
             int32_t output_len, uint8_t meta, blosc2_dparams *dparams) with gil:
    cdef blosc2_schunk *sc = <blosc2_schunk *> dparams.schunk
    meta_caterva = get_caterva_metalayer(sc)

    cdef uint8_t [:] input_ = <uint8_t[:input_len]> input
    mvinput = memoryview(input_)

    cdef uint8_t [:] output_ = <uint8_t[:output_len]> output
    mvoutput = memoryview(output_)

    narch = mvinput[0]
    size = mvinput[1]
    itemsize = mvinput[2]
    dtype = np.float32 if itemsize == 4 else np.float64

    archetypes_start = 3
    data_trans_start = archetypes_start + narch * size * 8
    archetypes = np.asarray(mvinput[archetypes_start: data_trans_start]).view(np.float64).reshape(narch, size)

    data_trans = ((np.asarray(mvinput[data_trans_start:], dtype=dtype) + 1) / 255).reshape(-1, narch)
    data_trans /= data_trans.sum(1)[:, None]

    res = (data_trans @ archetypes).reshape(meta_caterva.blocks)
    res = split_backward(res, (4, 4))
    mvoutput[:] = bytes(res)

    return res.nbytes


AACODEC = 230

def register_archetypes_codec():
    cdef blosc2_codec arch_codec
    arch_codec.compcode = AACODEC
    compname = "archetypal analysis codec".encode("utf-8")
    arch_codec.compname = compname
    arch_codec.complib = 1
    arch_codec.compver = 1
    arch_codec.decoder = decoder
    arch_codec.encoder = encoder

    blosc2_register_codec(&arch_codec)

    return 0
