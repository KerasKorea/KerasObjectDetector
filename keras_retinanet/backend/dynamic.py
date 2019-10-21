import os

_BACKEND = "tensorflow"

if "KERAS_BACKEND" in os.environ:
    _backend = os.environ["KERAS_BACKEND"]

    backends = {
        "cntk",
        "tensorflow",
        "theano"
    }

    assert _backend in backends

    _BACKEND = _backend

if _BACKEND == "cntk":
    from .cntk_backend import *  # noqa: F401,F403
elif _BACKEND == "theano":
    from .theano_backend import *  # noqa: F401,F403
elif _BACKEND == "tensorflow":
    from .tensorflow_backend import *  # noqa: F401,F403
else:
    raise ValueError("Unknown backend: " + str(_BACKEND))
