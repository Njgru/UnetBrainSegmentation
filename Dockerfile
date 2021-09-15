# Dockerfile, Image, Container

FROM ubuntu
WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.5 \
    python3-pip 
RUN apt-get update
RUN apt-get install python3-pip

RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install monai 
RUN pip3 install torchio 
RUN pip3 install pytorch-lightning 

RUN pip3 install pathlib 
RUN pip3 install DateTime 
RUN pip3 install matplotlib numpy 
# RUN sudo pip3 install torch torchvision torchaudio
# RUN sudo apt-get install python-numpy

COPY UnetLoader.py .
COPY Unet_Model_Multichannel_Statedict.pth .

#RUN  ./run_unet.py

# CMD ["python", "./UnetLoader.py"]

RUN chmod a+x UnetLoader.py .

## Make Docker container executable
ENTRYPOINT ["python3", "./UnetLoader.py"]