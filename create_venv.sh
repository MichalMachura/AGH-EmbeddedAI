#!/bin/bash
python -m venv tensor_rt_venv

source tensor_rt_venv/bin/activate

pip install -r requirements.txt
pip install torch-tensorrt==1.3.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.3.0
pip install git+https://github.com/LukeTonin/simple-deep-learning --no-deps
