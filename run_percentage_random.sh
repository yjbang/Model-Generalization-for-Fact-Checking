CUDA_VISIBLE_DEVICES=6 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.50 > run_rand_0.50.log &
CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.25 > run_rand_0.25.log &
CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.10 > run_rand_0.10.log &
CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.05 > run_rand_0.05.log &
CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.02 > run_rand_0.02.log &
CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.01 > run_rand_0.01.log

CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.75 > run_rand_0.75.log &
CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.90 > run_rand_0.90.log &
CUDA_VISIBLE_DEVICES=3 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.95 > run_rand_0.95.log &
CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.98 > run_rand_0.98.log &
CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random.py --percentage 0.99 > run_rand_0.99.log
