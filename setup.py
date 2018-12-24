'''
Created on 27 set 2018

@author: gcarla
'''

from setuptools import setup

setup(name='tesi',
      version='0.1',
      description='la tesi della giulia',
      url='',
      author='lagiulia',
      author_email='',
      license='MIT',
      packages=['tesi'],
      install_requires=[
          "astropy",
          "ccdproc",
          "photutils",
        ],
      zip_safe=False)