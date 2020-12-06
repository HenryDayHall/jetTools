from setuptools import setup

setup(
    name='jet_tools',
    version='0.1.0',
    author='Henry Day Hall',
    packages=['jet_tools', 'jet_tools.mini_data', 'jet_tools.tree_tagger',
        'jet_tools.test'],
    url='https://github.com/jacanchaplais/jetTools',
    scripts=['bin/script1','bin/script2'],
    long_description=open('README.txt').read(),
    install_requires='requirements.txt',
)
