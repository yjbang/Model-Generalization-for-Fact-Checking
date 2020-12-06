CUDA=1
LOSS_FNC=CE
EPOCH=20
lr=5e-6
EXP_NAME=roberta_finetune
MODEL=bert-base-uncased

CUDA_VISIBLE_DEVICES=$CUDA python finetune_fake_news.py \
    --exp_name $EXP_NAME\
    --model_name_or_path $MODEL \
    --per_gpu_train_batch_size 4\
    --per_gpu_eval_batch_size 4\
    --loss $LOSS_FNC \
    --num_train_epochs $EPOCH\
    --learning_rate $lr\
    --model_save_path /home/yejin/math6380/save/${EXP_NAME}.${LOSS_FNC}.${lr}/
