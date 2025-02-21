@echo off
set TF_ENABLE_ONEDNN_OPTS=0
echo. > logs/training.log
python src/train.py