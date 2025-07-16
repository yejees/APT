CUDA_DEVICE=$1 
CONTRAST_NUM=$2
CONTRAST_NUM2=$3

export CUDA_VISIBLE_DEVICES=$1

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True" 

python test.py \
 $MODEL_FLAGS \
 --repeat_steps 10 \
 --num_samples 4 \
 --model_path  ./checkpoints/last.pt \
 --timestep_respacing 50 \
 --start_idx1 20 \
 --start_idx2 2 \
 --total_loop_num2 50 \
 --sample_method ours \
 --mask_type box \
 --save_dir ./sample_results \
 --contrast_num $2 \
 --contrast_num2 $3 \