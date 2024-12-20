# Common Code
Code that both LiveCellAnalysis and smFISH (or others) use and doesn't get major changes goes here.

# Installation (Ubuntu)

    pip install tllab_common

or editable:

    git clone git@github.com:Lenstralab/tllab_common.git
    pip install -e tllab_common/ --user

### Installation of Maven and OpenJDK
Running the livecell_track_movies pipeline with segmentation using trackmate requires
[OpenJDK](https://en.wikipedia.org/wiki/OpenJDK) and [Maven](https://maven.apache.org/).
If not installed already you will have to install them manually and make sure mvn is on the path.