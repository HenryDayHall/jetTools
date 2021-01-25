# This is a simple docker for running the tree_tagger repo
# designed to showcase reading hepmc and root files

# To build
# > docker build -t tree_tagger .
# To run 
# > docker run -ti tree_tagger

# base image
FROM ubuntu:20.04

# some meta data
LABEL version=1.0
LABEL description="Example for tree_tagger"
LABEL author="Henry Day-Hall"

# things to prevent docker from asking for inputs the user cannot give
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV NOTVISIBLE "in users profile"


# install packages
RUN apt update && \
    apt -y install python3 python3-pip vim-nox git

RUN  pip3 install --upgrade pip

RUN pip3 install uproot matplotlib ipdb scipy sklearn networkx tabulate torch scikit-hep
# mayavi would be nice too, but it is dificult to install....

RUN echo "echo 'Use ipython3, not python3. Python3 has some problem with realtive imports.'" >> ~/.bashrc
# add the data
ADD h1bBatch2.hepmc /data/
ADD h1bBatch2.root /data/

# get the code
RUN git clone https://github.com/HenryDayHall/jetTools.git

# make the example file run as soon as the docker is opened
RUN echo "echo 'Running the example program'" >> ~/.bashrc   && \
    echo "cd /jetTools" >> ~/.bashrc  && \
    echo "ipython3 tree_tagger/example.py" >> ~/.bashrc

