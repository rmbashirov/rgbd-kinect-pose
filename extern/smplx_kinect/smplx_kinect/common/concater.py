import torch
import numpy as np
from copy import deepcopy


def features_axis_is_np(is_np, is_seq=False):
    """
    B - batch
    S - sequence
    F - features
    :param d: torch.Tensor or np.ndarray of shape specified below
    :param is_seq: True if d has sequence dim
    :return: axis of features
    """
    if is_np:
        # in case of sequence (is_seq == True): (S, F)
        # in case of sample: (F)
        return 0 + is_seq
    else:
        # in case of sequence: (B, S, F)
        # in case of sample: (B, F)
        return 1 + is_seq


def universe_len_is_np(shape, is_np, is_seq=False):
    ax = features_axis_is_np(is_np, is_seq=is_seq)
    return shape[ax]


def features_axis(d, is_seq=False):
    """
    B - batch
    S - sequence
    F - features
    :param d: torch.Tensor or np.ndarray of shape specified below
    :param is_seq: True if d has sequence dim
    :return: axis of features
    """
    return features_axis_is_np(
        isinstance(d, np.ndarray),
        is_seq=is_seq
    )


def universe_len(d, is_seq=False):
    if isinstance(d, NamedDataConcater):
        return len(d)
    else:
        return universe_len_is_np(d.shape, isinstance(d, np.ndarray), is_seq=is_seq)


def universe_concat(ds, name=None, is_seq=False):
    if isinstance(ds[0], NamedDataConcater):
        return NamedDataConcater.concat(ds, name=name)
    else:
        ax = features_axis(ds[0], is_seq=is_seq)
        if isinstance(ds[0], np.ndarray):
            return np.concatenate(ds, axis=ax)
        else:
            return torch.cat(ds, dim=ax)


def np_slice(arr, slc, ax):
    sl = [slice(None)] * arr.ndim
    sl[ax] = slc
    return arr[tuple(sl)]


def universe_get_slice(d, slc, is_seq=False):
    if isinstance(d, NamedDataConcater):
        return universe_get_slice(d.data, slc, is_seq=is_seq)
    else:
        ax = features_axis(d, is_seq=is_seq)
        if isinstance(d, np.ndarray):
            return np_slice(d, slc, ax=ax)
        else:
            return torch.narrow(d, dim=ax, start=slc.start, length=slc.stop - slc.start)


class NamedSliceHierarchy:
    def __init__(self, slc, name=None, **kwargs):
        assert name is None or isinstance(name, str)
        assert isinstance(slc, slice)
        self.slice = slc
        self.name = name
        self.children = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_child(self, named_slice_hierarchy):
        assert isinstance(named_slice_hierarchy, NamedSliceHierarchy)
        self.children.append(named_slice_hierarchy)

    def is_leaf(self):
        return len(self.children) == 0

    def shift_slices(self, value):
        for child in self.children:
            child.shift_slices(value)
        new_slice = slice(self.slice.start + value, self.slice.stop + value)
        self.slice = new_slice

    def get_name(self, level=0):
        prefix = ' ' * level
        is_left = ' (leaf)' if self.is_leaf() else ''
        result = f'{prefix}name: {self.name}, [{self.slice.start}, {self.slice.stop}){is_left}'
        for child in self.children:
            result += f'\n{child.get_name(level=level + 1)}'
        return result

    def __str__(self):
        return self.get_name()

    def name2slice(self, name):
        if self.name == name:
            return self.slice
        else:
            for child in self.children:
                result = child.name2slice(name)
                if result is not None:
                    return result
        return None

    def __eq__(self, other):
        if not isinstance(other, NamedSliceHierarchy):
            return False
        if not self.slice == other.slice:
            return False
        if not self.name == other.name:
            return False
        if len(self.children) != len(other.children):
            return False
        for i in range(len(self.children)):
            if not self.children[i] == other.children[i]:
                return False
        return True


class NamedDataConcater:
    def __init__(self, data, name=None, named_slice_hierarchy=None, is_seq=False, **kwargs):
        self.data = data
        self.is_seq = is_seq
        if named_slice_hierarchy is not None:
            self.named_slice_hierarchy = named_slice_hierarchy
        else:
            self.named_slice_hierarchy = NamedSliceHierarchy(slice(0, len(self)), name, **kwargs)

    def __len__(self):
        return universe_len(self.data, is_seq=self.is_seq)

    def __str__(self):
        return str(self.named_slice_hierarchy)

    def name2data(self, name):
        sl = self.named_slice_hierarchy.name2slice(name)
        return universe_get_slice(self.data, sl, is_seq=self.is_seq) if sl is not None else None

    def name2len(self, name):
        sl = self.named_slice_hierarchy.name2slice(name)
        return sl.stop - sl.start

    def transform_data(self, f):
        self.data = f(self.data)

    @staticmethod
    def concat(named_data_concaters, name=None):
        ndc0 = named_data_concaters[0]
        for ndc in named_data_concaters:
            assert isinstance(ndc, NamedDataConcater), type(ndc)

        data = universe_concat([ndc.data for ndc in named_data_concaters], is_seq=ndc0.is_seq)
        result = NamedDataConcater(data=data, name=name, is_seq=named_data_concaters[0].is_seq)

        cur_len = 0
        for ndc in named_data_concaters:
            child = deepcopy(ndc.named_slice_hierarchy)
            child.shift_slices(cur_len)
            result.named_slice_hierarchy.add_child(child)
            cur_len += len(ndc)

        return result

    @staticmethod
    def stack(named_data_concaters, f):
        ndc0 = named_data_concaters[0]
        for ndc in named_data_concaters:
            assert isinstance(ndc.data, torch.Tensor)
            assert ndc0.named_slice_hierarchy == ndc.named_slice_hierarchy
        data = f([ndc.data for ndc in named_data_concaters])
        result = NamedDataConcater(
            data,
            named_slice_hierarchy=deepcopy(ndc0.named_slice_hierarchy),
            is_seq=ndc0.is_seq
        )
        return result
