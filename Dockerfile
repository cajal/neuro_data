#FROM eywalker/pytorch-jupyter:v0.4.0-updated
FROM atlab/pytorch

#RUN apt-get -y update && apt-get  -y install ffmpeg libhdf5-10 fish git
RUN apt-get -y update && apt-get  -y install ffmpeg fish git
RUN pip install imageio ffmpy h5py opencv-python statsmodels
#RUN pip3 install mock
##RUN pip3 install git+https://github.com/circstat/pycircstat
RUN pip install imageio-ffmpeg
RUN pip install git+https://github.com/atlab/attorch
#RUN pip install --upgrade torch torchvision

ADD . /src/neuro_data
RUN pip install -e /src/neuro_data


WORKDIR /notebooks

