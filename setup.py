from setuptools import setup
from setuptools import find_packages

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
          'image': ['pillow']
      },
      packages=find_packages())