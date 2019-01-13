
import numpy as np
import copy
import os.path as osp
import json
import logging
from pprint import pformat as pf

from ..logcfg import log

__all__ = [
        "Data",
    ]


def join_base_dicts(bases, dct, attribute):
    """
        Join dictionaries from base classes and then update with dct.
    """
    joined = {}

    for d in (b.__dict__.get(attribute, {})
              for b in bases):
        joined.update(d)

    joined.update(dct.get(attribute, {}))

    return joined


class MetaData(type):
    """
        Meta class to store all defined classes:
    """
    registry = {}

    def __new__(cls, name, bases, dct):
        for attr in ["data_attribute_types", "data_attribute_defaults"]:
            dct[attr] = join_base_dicts(bases, dct, attribute=attr)

        klass = super(MetaData, cls).__new__(cls, name, bases, dct)

        cls.register_class(name, klass)

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("Registered {} -> {}".format(
                name, id(cls.get_class(name))))

        return klass

    @classmethod
    def get_class(cls, name):
        return cls.registry[name]

    @classmethod
    def register_class(cls, name, klass):
        cls.registry[name] = klass


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
    def load(cls, filepath):
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("Loading {} [id: {}]".format(cls.__name__, id(cls)))

        with open(filepath, "r") as f:
            datadict = json.load(f)

        if datadict["_type"] != cls.__name__:
            log.warn("Using json data for type {} to create type {}".format(
                datadict["_type"], cls.__name__))
        del datadict["_type"]

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(pf({k: id(v) for k, v in
                      cls.__metaclass__.registry.iteritems()}))

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

        attributes = {k: v for k, v in convert_from.to_dict().iteritems()
                      if k in cls.data_attribute_types}

        return cls(**attributes)

    def __init__(self, **attributes):
        self.from_dict(attributes)

        # set all those attributes that werent specified
        self.set_empty()

    def __str__(self):
        return pf(self.to_dict())

    def __setattr__(self, name, value):
        """
            Make sure that the attribute we set is part of our default
            dictionary.
        """
        if name not in self.data_attribute_types.keys():
            error_msg = "{} not part of {}'s data-attributes! "\
                "Setting anyway..".format(name, self.__class__.__name__)
            log.error(error_msg)
            raise ValueError(error_msg)

        return object.__setattr__(self, name, value)

    def get_dict(self):
        return self.to_dict(with_type=False)

    @classmethod
    def get_attr_keys(cls):
        """
            Return a list of all attribute keys.
        """
        return cls.data_attribute_types.keys()

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
            contains a special "_type" field and is JSON-compatible.
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

            If `with_type` is True, the returned dictionary
            contains a special "_type" field and is JSON-compatible.
        """
        d = getattr(self, name)

        if isinstance(d, Data):
            return d.to_dict(with_type=with_type)

        if isinstance(d, np.ndarray):
            if with_type:
                return d.tolist()
            else:
                return d.copy()

        return copy.deepcopy(d)

    def from_dict(self, dikt):
        for name, d in dikt.iteritems():
            if name == "_type":
                # _type is present in the dictionary supplied when loading
                # recursive data structures
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
                # always load class from metaclass to prevent duplicates
                desired_type = MetaData.get_class(d["_type"])

                if log.getEffectiveLevel() <= logging.DEBUG:
                    log.debug("Creating: {} [id: {}]".format(
                        desired_type.__name__, id(desired_type)))
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
