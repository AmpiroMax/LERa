"""
The script was taken from 
https://github.com/allenai/allenact/blob/main/allenact_plugins/ithor_plugin/__init__.py
"""
from fiqa.navigation.allenact.allenact.utils.system import ImportChecker

with ImportChecker(
    "Cannot `import ai2thor`, please install `ai2thor` (`pip install ai2thor`)."
):
    # noinspection PyUnresolvedReferences
    import ai2thor
