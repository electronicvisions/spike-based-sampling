
import logging
import os

LOGNAME = "SbS"
LOGFILE = os.path.expanduser("~/." + LOGNAME + ".log")
log = None
default_verbose_formatter = logging.Formatter("%(asctime)s %(levelname)s "
        "%(funcName)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%y-%m-%d %H:%M:%S")
default_formatter = logging.Formatter("%(asctime)s %(levelname)s: "
        "%(message)s", datefmt="%y-%m-%d %H:%M:%S")

formatter_in_use = default_formatter # allows switching of the global formatter
loglevel_in_use = "INFO"               # same for loglevels

log = logging.getLogger(LOGNAME)
log.setLevel(logging.INFO)

default_handler_stream = None
default_handler_file = None


def set_loglevel(lg, lvl):
    if not str(lvl).isdigit():
        lvl = getattr(logging, lvl.upper())
    lg.setLevel(lvl)


def set_loglevel_stream(lvl="INFO"):
    set_loglevel(default_handler_stream, lvl)


def set_loglevel_file(lvl="DEBUG"):
    set_loglevel(default_handler_file, lvl)


def add_handler(type_, handler_kwargs=None, loglevel=None,
                formatter=None):
    kwargs = {}
    if handler_kwargs is not None:
        kwargs.update(handler_kwargs)
    handler = type_(**kwargs)

    # allow dynamic defaults
    if formatter is None:
        formatter = formatter_in_use
    if loglevel is None:
        loglevel = loglevel_in_use

    handler.setFormatter(formatter)
    set_loglevel(handler, loglevel)
    log.addHandler(handler)
    return handler


def add_stream_handler(**kwargs):
    """
        Adds new stream handler.

        **kwargs are passed to `add_handler`.
    """
    return add_handler(logging.StreamHandler)


def add_file_handler(filename, mode="a", **kwargs):
    """
        Adds new file handler with specified filename and mode.

        **kwargs are passed to `add_handler`.
    """
    handler_kwargs = kwargs.setdefault("handler_kwargs", {})
    handler_kwargs["filename"] = filename
    handler_kwargs["mode"] = mode
    return add_handler(logging.FileHandler, **kwargs)


def make_verbose():
    global formatter_in_use
    formatter_in_use = default_verbose_formatter
    verbose_loglevel = "DEBUG"
    loglevel_in_use = verbose_loglevel
    set_loglevel(log, verbose_loglevel)
    for h in log.handlers:
        set_loglevel(h, verbose_loglevel)
        h.setFormatter(default_verbose_formatter)



if "DEBUG" in os.environ:
    formatter_in_use = default_verbose_formatter

default_handler_stream = add_stream_handler(
        loglevel="INFO")

# TODO: Delete
# default_handler_file = add_file_handler(filename=LOGFILE, loglevel="DEBUG")

if "LOGTOFILE" in os.environ:
    default_handler_file = add_file_handler(LOGFILE, loglevel="DEBUG")

if "DEBUG" in os.environ:
    make_verbose()

for i in range(3):
    log.debug("-" * 80)



