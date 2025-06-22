"""
The script was taken from 
https://github.com/allenai/allenact/blob/main/allenact_plugins/__init__.py
"""
try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from allenact_plugins._version import __version__
except ModuleNotFoundError:
    __version__ = None
