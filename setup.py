import setuptools
import platform

if platform.system().lower() == 'linux':
    import pkg_resources
    pkg_resources.require(['pip >= 20.3'])

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='tllab_common',
    version='2022.3.0',
    author='Lenstra lab NKI',
    author_email='t.lenstra@nki.nl',
    description='Common code for the Lenstra lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.rhpc.nki.nl/LenstraLab/tllab_common',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=['untangle', 'pandas', 'psutil', 'numpy', 'tqdm', 'tifffile', 'czifile', 'pyyaml', 'dill',
                      'colorcet', 'multipledispatch', 'numba', 'scipy', 'tiffwrite'],
    extras_require={'transforms': 'SimpleITK-SimpleElastix',
                    'bioformats': ['python-javabridge', 'python-bioformats']},
    tests_require=['pytest-xdist'],
    scripts=['bin/wimread'],
    package_data={'': ['transform.txt']},
    include_package_data=True,
)
