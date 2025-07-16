export  OPENAI_LOGDIR="./results"

MODEL_FLAGS="--attention_resolutions 32,16,8 \
--class_cond False \
--diffusion_steps 1000 \
--image_size 256 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 128 \
--num_head_channels 64 \
--num_res_blocks 2 \
--resblock_updown True \
--use_fp16 False \
--use_scale_shift_norm True"

TRAIN_FLAGS="--lr 1e-4 \
--weight_decay 0.05 \
--save_interval 10000 \
--batch_size 8"

DIFFUSION_FLAGS="--diffusion_steps 1000 \
--noise_schedule linear \
--batch_size 8"

mpiexec -n 4 python scripts/image_train.py --data_dir data_path $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS