import setuptools
import platform
import os

version = '2022.6.0'

if platform.system().lower() == 'linux':
    import pkg_resources
    pkg_resources.require(['pip >= 20.3'])

with open('README.md', 'r') as fh:
    long_description = fh.read()


with open(os.path.join(os.path.dirname(__file__), 'tllab_common', '_version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    try:
        import git
        f.write(f"__git_commit_hash__ = '{git.Repo(os.path.dirname(__file__)).head.object.hexsha}'\n")
    except Exception:
        f.write(f"__git_commit_hash__ = 'unknown'\n")


setuptools.setup(
    name='tllab_common',
    version=version,
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
                      'colorcet', 'multipledispatch', 'numba', 'scipy', 'tiffwrite', 'roifile'],
    extras_require={'transforms': 'SimpleITK-SimpleElastix',
                    'bioformats': ['python-javabridge', 'python-bioformats']},
    tests_require=['pytest-xdist'],
    scripts=['bin/wimread'],
    package_data={'': ['transform.txt']},
    include_package_data=True,
)
