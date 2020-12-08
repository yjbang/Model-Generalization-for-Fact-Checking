if [ $1 == 0 ] ; then
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.75 --model_type roberta-large > log_yejin/run_0.75_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.90 --model_type roberta-large > log_yejin/run_0.90_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.95 --model_type roberta-large > log_yejin/run_0.95_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.98 --model_type roberta-large > log_yejin/run_0.98_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.99 --model_type roberta-large > log_yejin/run_0.99_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.75 --model_type roberta-large > log_yejin/run_0.75_rand_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.90 --model_type roberta-large > log_yejin/run_0.90_rand_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.95 --model_type roberta-large > log_yejin/run_0.95_rand_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.98 --model_type roberta-large > log_yejin/run_0.98_rand_wm.log
    CUDA_VISIBLE_DEVICES=0 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.99 --model_type roberta-large > log_yejin/run_0.99_rand_wm.log
elif [ $1 == 1 ] ; then
    CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.50 --model_type roberta-large > log_yejin/run_0.50_wm.log
    CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.25 --model_type roberta-large > log_yejin/run_0.25_wm.log
    CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.50 --model_type roberta-large > log_yejin/run_0.50_rand_wm.log
    CUDA_VISIBLE_DEVICES=1 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.25 --model_type roberta-large > log_yejin/run_0.25_rand_wm.log
elif [ $1 == 2 ] ; then
    CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.10 --model_type roberta-large > log_yejin/run_0.10_wm.log
    CUDA_VISIBLE_DEVICES=2 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.10 --model_type roberta-large > log_yejin/run_0.10_rand_wm.log
elif [ $1 == 3 ] ; then
    CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.05 --model_type roberta-large > log_yejin/run_0.05_wm.log
    CUDA_VISIBLE_DEVICES=4 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.05 --model_type roberta-large > log_yejin/run_0.05_rand_wm.log
elif [ $1 == 4 ] ; then
    CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.02 --model_type roberta-large > log_yejin/run_0.02_wm.log
    CUDA_VISIBLE_DEVICES=5 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.02 --model_type roberta-large > log_yejin/run_0.02_rand_wm.log
elif [ $1 == 5 ] ; then
    CUDA_VISIBLE_DEVICES=6 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_yejin.py --percentage 0.01 --model_type roberta-large > log_yejin/run_0.01_wm.log
    CUDA_VISIBLE_DEVICES=6 /home/samuel/anaconda2/envs/env_py3.7/bin/python run_percentage_random_yejin.py --percentage 0.01 --model_type roberta-large > log_yejin/run_0.01_rand_wm.log
fi