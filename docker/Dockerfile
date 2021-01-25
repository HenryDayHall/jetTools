# To build;
# docker build -t dockjets .
# to run;
# mkdir output
# docker run -v $(pwd)/output:/tohost -ti dockjets

# base image
FROM ubuntu:18.04

# some meta data
LABEL version=1.0
LABEL description="A docker container for processing jets"
LABEL author="Henry Day-Hall"

# set a timezone so I don't get asked
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# install packages
RUN apt update && \
    apt -y install python3.6 ipython3 python3-pip \
                       wget vim-nox nano make g++ \
                       git python3-ipdb tmux
#RUN echo "alias vim=vim.tiny" >> ~/.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib sklearn awkward

# fetch the packages
RUN  wget -O fastjet.tar.gz http://fastjet.fr/repo/fastjet-3.3.3.tar.gz
RUN  mkdir fastjet  && \
     tar -xf fastjet.tar.gz -C fastjet --strip-components 1  && \
     rm fastjet.tar.gz
## start making packages
RUN cd fastjet && \
    ./configure  && \
    make && \
    make check && \
    make install
# fetch custom code
RUN git clone https://github.com/HenryDayHall/jetTools.git
# leave some instructions
RUN echo "echo" >> ~/.bashrc                                                                                                      && \
    echo "echo" >> ~/.bashrc                                                                                                      && \
    echo "echo 'In order to work with these scripts you would need a hepmc file with particle data from madgraph.'" >> ~/.bashrc                                           && \
    echo "echo 'If you have that the place to start is ReadHepmc.py'" >> ~/.bashrc                                                               && \
    echo "echo" >> ~/.bashrc                                                                                                      && \
    echo "echo" >> ~/.bashrc
