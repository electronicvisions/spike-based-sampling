
import numpy as np
import copy
import os.path as osp
import json
from pprint import pformat as pf

from ..logcfg import log

__all__ = [
        "MetaData",
        "Data",
    ]


def join_data_attribute_types(bases, dct):
    joined = {}

    for d in ( b.__dict__.get("data_attribute_types", {})
            for b in bases):
        joined.update(d)

    joined.update(dct.get("data_attribute_types", {}))

    return joined


class MetaData(type):
    """
        Meta class to store all defined classes:
    """
    registry = {}

    def __new__(cls, name, bases, dct):
        dct["data_attribute_types"] = join_data_attribute_types(bases, dct)

        klass = super(MetaData, cls).__new__(cls, name, bases, dct)

        cls.registry[name] = klass
        return klass

    @classmethod
    def get_class(cls, name):
        return cls.registry[name]


class Data(object):
    """
        Baseclass for all data objects
    """
    # dict mapping data attribute names to types
    data_attribute_types = {}
    # dict for default values for unspecified attributes
    data_attribute_defaults = {}

    __metaclass__ = MetaData

    @classmethod
    def load(cls, filepath=None):
        with open(filepath) as f:
            datadict = json.load(f)

        if datadict["_type"] != cls.__name__:
            log.warn("Using json data for type {} to create type {}".format(
                datadict["_type"], cls.__name__))
        del datadict["_type"]

        return cls(**datadict)

    @classmethod
    def convert(cls, convert_from):
        """
            Convert one Data-like object to another.

            All attributes not present in the target type will be discarded.
        """
        assert isinstance(convert_from, Data)

        if not issubclass(cls, convert_from.__class__):
            log.warn("Converting {} to non-subclass {}.".format(
                convert_from.__class__.__name__, cls.__name__))

        attributes = {k: v for k,v in convert_from.to_dict().iteritems()
                if k in cls.data_attribute_types}

        return cls(**attributes)

    def __init__(self, **attributes):
        self.from_dict(attributes)

        # set all those attributes that werent specified
        self.set_empty()

    def __str__(self):
        return pf(self.to_dict())

    def get_dict(self):
        return self.to_dict(with_type=False)

    def write(self, path):
        if osp.splitext(path)[1] != ".json":
            path += ".json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f,
                    ensure_ascii=False, indent=2)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def copy(self):
        return self.__class__(**self.to_dict())

    def set_empty(self):
        """
            Set everything apart from class constants
        """
        for d in self.data_attribute_types:
            if not hasattr(self, d):
                setattr(self, d, self.data_attribute_defaults.get(d, None))

    def to_dict(self, with_type=True):
        """
            If `with_type` is True, the returned dictionary
            contains a special "_type" field.
        """
        dikt = {d: self._convert_attr(d, with_type=with_type)
                for d in self.data_attribute_types}
        if with_type:
            dikt["_type"] = self.__class__.__name__
        return dikt

    def _convert_attr(self, name, with_type):
        """
            Helper functions properly converting attributes.

            Returns a JSON compatible datatype.
        """
        d = getattr(self, name)

        if isinstance(d, Data):
            return d.to_dict(with_type=with_type)

        if isinstance(d, np.ndarray):
            return d.tolist()

        return copy.deepcopy(d)

    def from_dict(self, dikt):
        for name, d in dikt.iteritems():
            if name == "_type":
                continue
            try:
                desired_type = self.data_attribute_types[name]
            except KeyError:
                log.error("[{}] Unexpected attribute: {}".format(
                    self.__class__.__name__, name))
                continue

            if hasattr(self, name):
                # class constants are skipped
                continue

            if isinstance(d, dict) and issubclass(desired_type, Data):
                if d["_type"] != desired_type.__name__:
                    new_desired_type = MetaData.get_class(d["_type"])
                    assert issubclass(new_desired_type, desired_type)
                    desired_type = new_desired_type
                del d["_type"]
                d = desired_type(**d)

            elif d is not None and issubclass(desired_type, np.ndarray):
                d = np.array(d)

            setattr(self, name, d)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        if not isinstance(other, Data):
            return False

        d_self = self.to_dict()
        d_other = other.to_dict()

        if len(set(d_self.iterkeys()) ^ set(d_other.iterkeys())) > 0:
            # we have different sets of keys -> not equal
            return False

        return all((d_self[k] == d_other[k] for k in d_self.iterkeys()))

