import builtins


def print(value, *args, enabled=False, **kwargs):
    """Builtin print() method modified so can be switched on/off via parameter enabled"""
    if enabled:
        return builtins.print(value, *args, **kwargs)

