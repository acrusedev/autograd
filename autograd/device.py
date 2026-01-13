from functools import cached_property

available_devices = ["CPU"]

class _Device:
    """
    Device is where Tensors are stored and compute is run.
    We will autodetect the best device on your system and makes it the default.
    For now tho we will use CPU
    """

    @cached_property
    def DEFAULT(self) -> str:
        return "CPU"

Device = _Device()