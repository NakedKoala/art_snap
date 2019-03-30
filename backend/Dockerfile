FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir /usr/storage

WORKDIR /usr/storage

RUN pip install scikit-image && \
     apt-get update && \
     apt install ffmpeg -y && \
    pip install boto3

RUN  mkdir ~/.aws/
RUN printf "[default]\naws_access_key_id = ***\naws_secret_access_key = ***" >  \
    ~/.aws/credentials && \
    printf "[default]\nregion=us-east-1" > ~/.aws/config
COPY ./ ./
CMD ["/bin/bash"]