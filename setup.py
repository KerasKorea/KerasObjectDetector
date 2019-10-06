import os
import sys
import platform
import subprocess
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
from distutils.command.build import build as _build

from setuptools import setup
from setuptools import find_packages


if sys.version.info < (3.0):
    raise Exception('Python2 is not supported')

PLATFORMS = {'windows', 'linux', 'darwin'}

target = platform.system().lower()

for known in PLATFORMS:
    if target.startswith(known):
        target = known

if target not in PLATFORMS:
    target = 'linux'

# For mac
if target == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.14' and current_system >= '10.14':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.14'


libraries = {
    'windows': [],
    'linux': [],
    'darwin': [],
}


extra_compile_args = {
    'windows': [],
    'linux': [],
    'darwin': [],
}

extra_linker_args = {
    'windows': [],
    'linux': [],
    'darwin': [],
}


long_description = '''
Keras Object Detector is a high-level object detection API,
written in Python and capable of running on top of
TensorFlow, CNTK, or Theano.

Use Keras Object Detector if you need a object detecting library that:

- Allows for easy and fast prototyping
  (through user friendliness, modularity, and extensibility).
- Runs seamlessly on CPU and GPU.

Read the documentation at: # todo: keras object detector tutorial, quick-start

For a detailed overview of what makes Keras special, see:
https://keras.io/why-use-keras/
'''


setup(name='KerasObjectDetector',
      version='0.0.1',
      description='Object Detector for Keras User',
      long_description=long_description,
      author='Keras Korea',
      author_email='',
      url='https://github.com/KerasKorea/KerasObjectDetector',
      download_url='https://github.com/keras-team/keras/tarball/2.3.0',
      license='MIT or Apache',
      libraries=libraries[target],
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py',
                        'six',
                        'tensorflow'],
      extras_require={
          'visualize': ['pydot>=1.2.4', 'matplotlib'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',


                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'markdown'],
          'image': ['pillow',
                    'opencv-python']
      },
      packages=find_packages(),
      python_requires='>=3.5',
      )

command_list = ['update', 'dist-upgrade',
                'libatlas-base-dev', 'libxml2-dev', 'libxslt-dev', 'python-tk']