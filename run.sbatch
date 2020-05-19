#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:30:00
#SBATCH --mem=88GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jh6926@nyu.edu

NETID=jh6926

DATA_DIR=/scratch/${NETID}/nlu/SST
OUTPUT_DIR=/scratch/${NETID}/nlu/SST_result

# run roberta training on SST-2
python examples/run_glue.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name SST-2 \
--do_train \
--do_eval \
--do_lower_case \
--data_dir $DATA_DIR \
--max_seq_length 128 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-6 \
--num_train_epochs 3 \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir
