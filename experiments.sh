#!/bin/bash
#transformed_mode=PowEMLR #PowEMLR,ScalePowEMLR,PowTMLR

#First change the datadir in conf/dataset/base_dataset.yaml

# --- fgvc-aircraft ---
dataset=fgvc-aircraft # fgvc-aircraft, CUB_200_2011, fgvc-cars

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=ScalePowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowTMLR

# --- CUB_200_2011 ---
dataset=CUB_200_2011

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=ScalePowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowTMLR\
  dataset.training.lr=5e-3\
  dataset.training.batch_size=6\
  dataset.training.classifier_factor=1

# --- fgvc-cars ---
dataset=fgvc-cars

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=ScalePowEMLR

[ $? -eq 0 ] && python main.py \
  dataset=$dataset\
  representation.params.transformed_mode=PowTMLR\
  dataset.training.lr=5e-3\
  dataset.training.classifier_factor=1