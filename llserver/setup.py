from setuptools import find_packages, setup

setup(
    name="llserver",
    packages=find_packages(),
    python_requires=">=3",
    description="LLServer: project for serving LLM",
    author="anonymus",
    version="0.1.0",
    install_requires=[
        # Installing llava from GitHub with its "train" extras
        # 'llava[train] @ git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@main#egg=llava',
    ],
    extras_require={
        'train': [
            # You can add any additional dependencies required for training here if necessary
        ]
    },
)
