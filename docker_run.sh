 docker run -it -v ~/:/main --shm-size=8192mb --gpus '"device=1"' --user $(id -u):$(id -g) --rm sr-codecs
