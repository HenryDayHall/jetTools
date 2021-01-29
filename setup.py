from setuptools import setup

setup(
    name='jet_tools',
    version='0.1.0',
    author='Henry Day Hall',
    packages=['jet_tools'],
    url='https://github.com/HenryDayHall/jetTools',
    long_description=open('README.md').read(),
    install_requires=[
        "uproot == 3.10.12",
        "awkward == 0.13.0",
        "numpy >= 1.16.1",
        "matplotlib >= 3.1.2",
        "ipdb >= 0.12.3",
        "scipy >= 1.3.3",
        "sklearn >= 0.0",
        "networkx >= 2.4",
        "pydot >= 1.4.1",
        "tabulate >= 0.8.6",
        "torch >= 1.3.1",
        "scikit-hep == 0.5.1",
        "bokeh >= 1.4.0",
        "psutil >= 5.6.7",
        "pygit2 >= 1.2.1"
    ]
)
