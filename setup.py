import setuptools
import git

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tllab_common",
    version=[i for i in git.Git('.').log('-1', '--date=format:%Y%m%d%H%M').splitlines() if i.startswith('Date:')][0][-12:],
    author="Wim Pomp @ Lenstra lab NKI",
    author_email="w.pomp@nki.nl",
    description="Common code for the Lenstra lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.rhpc.nki.nl/LenstraLab/common-code",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
    install_requires=['untangle', 'javabridge', 'python-bioformats', 'pandas', 'psutil', 'numpy', 'tqdm', 'tifffile',
                      'czifile', 'pyyaml', 'dill', 'colorcet'],
    scripts=['bin/wimread'],
)