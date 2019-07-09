from . import container_ext as ext
import numpy as np


class Container(ext._Container):

    def __init__(self, pshape=None, filename=None, **kargs):
        self.kargs = kargs
        super(Container, self).__init__(pshape, filename, **kargs)

    def __getitem__(self, key):
        buff = ext._getitem(self, key)
        return buff

    def __setitem__(self, key, item):
        ext._setitem(self, key, item)

    def copy(self, pshape=None, filename=None):
        arr = Container(pshape=pshape, filename=filename, **self.kargs)
        ext._copy(self, arr)
        return arr

    def to_buffer(self):
        return ext._to_buffer(self)

    def to_numpy(self, dtype):
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)


def empty(shape, pshape=None, filename=None, **kargs):
    arr = Container(pshape, filename, **kargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, pshape=None, filename=None, **kargs):
    arr = Container(pshape, filename, **kargs)
    ext._from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, pshape=None, filename=None, **kargs):
    arr = from_buffer(bytes(nparray), nparray.shape, pshape, filename, **kargs)
    return arr


def from_file(filename):
    arr = Container()
    ext._from_file(arr, filename)
    return arr