"""
Setting up repository as a module
"""

from setuptools import setup, find_packages

setup(name='source',
      package=find_packages('.'),
      zip_safe=False,
      py_modules=[]  
)
