# Common Code
Code that both LiveCellAnalysis and smFISH (or others) use and doesn't get major changes goes here. This code should be a submodule in smFISH and LiveCellAnalysis.

# Command line tools:
## wimread
    wimread imagefile
Displays information (pixel size, interval time etc.) about the image file.

# Installation (Ubuntu)
python-javabridge is required, which requires java, for ubuntu:

    sudo apt install openjdk-8-jdk-headless

Set java 8 as default java: 

    sudo update-alternatives --config java

Then this code can be installed as a package:

    pip install tllab_common

or editable:

    git clone git@github.com:Lenstralab/tllab_common.git
    pip install -e tllab_common/ --user