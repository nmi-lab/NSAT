#!/bin/python
#-----------------------------------------------------------------------------
# File Name : setup.py
# Author: Emre Neftci
#
# Creation Date : Thu 08 Dec 2016 05:41:37 PM PST
# Last Modified : Sun 29 Jan 2017 07:24:26 PM PST
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from setuptools import setup
from setuptools.command.install import install as SetupInstall
import subprocess

class MyInstall(SetupInstall):
	def run(self):
		subprocess.call('make')
		SetupInstall.run(self)

class MakeClean(SetupInstall):
	def run(self):
		subprocess.call(['make','clean'])

def readme():
	with open('README.md') as f:
        	return f.read()

setup(name='pyNSATlib',
      version='0.1',
      description='Python bindings for c_nsat',
      url='https://github.com/nmi-lab-ucsd/HiAER-NSAT',
      author='Emre Neftci, Georgios Detorakis, Sadique Sheik',
      author_email='eneftci@uci.edu',
      license='GPLv2',
      packages=['pyNSATlib'],
      include_package_data=True,
      install_requires=['numpy',
                        'pyNCSre',
                        'igraph',
                        'matplotlib'],
      dependency_links=['https://github.com/inincs/pyNCS/tarball/master#egg=package-1.0'],
      zip_safe=False,
      cmdclass={'install': MyInstall,
		'clean': MakeClean})
