#!/bin/bash

# Script that fetches all necessary data for training and eval

# Model constants etc.
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
# Initial fits to start training
wget http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz && tar -xvf static_fits.tar.gz --directory data && rm -r static_fits.tar.gz
# List of preprocessed .npz files for each dataset
wget https://keeper.mpdl.mpg.de/f/cc0efe451f9f4845932d/?dl=1 --max-redirect=2 --trust-server-names && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
# Pretrained checkpoint
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data
# Pretrained IPMAN checkpoint
wget https://keeper.mpdl.mpg.de/f/553bb5deaf5e4e29bfca/?dl=1 --max-redirect=2 --trust-server-names && tar -xvf ipman_checkpoints.tar.gz --directory data && rm -r ipman_checkpoints.tar.gz

# GMM prior from vchoutas/smplify0x
wget https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py -O smplify/prior.py
