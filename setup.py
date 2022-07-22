from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'seml',
    'tensorboardx',
    'merge_args'
]


setup(name='seml_logger',
      version='0.1.0',
      description='SEML logging utility',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
