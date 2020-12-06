CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.75 --model_type roberta-large > run_0.75_large_wm.log &
CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.90 --model_type roberta-large > run_0.90_large_wm.log &
CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.95 --model_type roberta-large > run_0.95_large_wm.log &
CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.98 --model_type roberta-large > run_0.98_large_wm.log &
CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.99 --model_type roberta-large > run_0.99_large_wm.log &
CUDA_VISIBLE_DEVICES=6 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.50 --model_type roberta-large > run_0.50_large_wm.log

CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.25 --model_type roberta-large > run_0.25_large_wm.log &
CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.10 --model_type roberta-large > run_0.10_large_wm.log &
CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.05 --model_type roberta-large > run_0.05_large_wm.log &
CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.02 --model_type roberta-large > run_0.02_large_wm.log &
CUDA_VISIBLE_DEVICES=6 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage.py --percentage 0.01 --model_type roberta-large > run_0.01_large_wm.log
