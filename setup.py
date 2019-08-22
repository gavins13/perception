from setuptools import setup

setup(name='Perception',
      version='2.0',
      description='A deep learning framework based on TensorFlow',
      url='http://github.com/gavinlive/perception',
      author='Gavin Seegoolam',
      author_email='kgs13@ic.ac.uk',
      license='MIT',
      packages=['perception'],
      install_requires=[
          'tensorflow==1.8.0',
          'numpy==1.15.3',
      ],
      zip_safe=False)
