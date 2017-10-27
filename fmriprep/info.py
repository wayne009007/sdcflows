# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

__version__ = '1.0.0-rc7-dev'
__author__ = 'The CRN developers'
__copyright__ = 'Copyright 2017, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Craig Moodie', 'Ross Blair', 'Oscar Esteban', 'Chris Gorgolewski',
               'Shoshana Berleant', 'Christopher J. Markiewicz', 'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Ross Blair'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fmriprep'
__packagename__ = 'fmriprep'
__description__ = ("fMRIprep is a functional magnetic resonance image pre-processing pipeline "
                   "that is designed to provide an easily accessible, state-of-the-art interface "
                   "that is robust to differences in scan acquisition protocols and that requires "
                   "minimal user input, while providing easily interpretable and comprehensive "
                   "error and output reporting.")
__longdesc__ = ("This package is a functional magnetic resonance image preprocessing pipeline "
                "that is designed to provide an easily accessible, state-of-the-art interface "
                "that is robust to differences in scan acquisition protocols and that requires "
                "minimal user input, while providing easily interpretable and comprehensive error "
                "and output reporting. This open-source neuroimaging data processing tool is "
                "being developed as a part of the MRI image analysis and reproducibility platform "
                "offered by the CRN. This pipeline is heavily influenced by the `Human Connectome "
                "Project analysis pipelines "
                "<https://github.com/Washington-University/Pipelines>`_ and, as such, the "
                "backbone of this pipeline is a python reimplementation of the HCP "
                "`GenericfMRIVolumeProcessingPipeline.sh` script. However, a major difference is "
                "that this pipeline is executed using a `nipype workflow framework "
                "<http://nipype.readthedocs.io/en/latest/>`_. This allows for each call to a "
                "software module or binary to be controlled within the workflows, which removes "
                "the need for manual curation at every stage, while still providing all the "
                "output and error information that would be necessary for debugging and "
                "interpretation purposes. The fmriprep pipeline primarily utilizes FSL tools, but "
                "also utilizes ANTs tools at several stages such as skull stripping and template "
                "registration. This pipeline was designed to provide the best software "
                "implementation for each state of preprocessing, and will be updated as newer and "
                "better neuroimaging software become available.")

DOWNLOAD_URL = (
    'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))


SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

REQUIRES = [
    'numpy',
    'lockfile',
    'future',
    'scikit-learn',
    'matplotlib',
    'nilearn',
    'sklearn',
    'nibabel>=2.1.0',
    'pandas',
    'grabbit',
    'pybids',
    'nitime',
    'niworkflows>=0.1.8',
    'statsmodels',
    'nipype',
    'seaborn',
    'indexed_gzip>=0.6.1',
]

LINKS_REQUIRES = [
    'git+https://github.com/poldracklab/niworkflows.git'
    '@d30cb4fd7e387484c67ef687bea3260496fcca6e#egg=niworkflows-0.1.9-dev',
    'git+https://github.com/oesteban/pybids.git'
    '@448e926c6a2cbad4a9520fdd79c3f8c8893f1ac2#egg=pybids-0.3.1-dev',
]

TESTS_REQUIRES = [
    "mock",
    "codecov",
    "pytest",
]

EXTRA_REQUIRES = {
    'doc': ['sphinx>=1.5.3', 'pydotplus', 'pydot>=1.2.3', 'sphinx_rtd_theme', 'sphinx-argparse'],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'datalad': ['datalad'],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
