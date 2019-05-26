
RESULTS_HOME="results"
MDL_CFGS="model_configs"
GLOVE_PATH="dictionaries"

DATA_DIR="TFRecords"
NUM_INST=11157649 # Number of sentences

#DATA_DIR="data/BC_UMBC/TFRecords"
#NUM_INST=174817800

CFG="BS400-W620-S1200-case-bidir"

BS=400
SEQ_LEN=30

export CUDA_VISIBLE_DEVICES=0
python src/train.py \
    --input_file_pattern="$DATA_DIR/train-?????-of-00100" \
    --train_dir="$RESULTS_HOME/$CFG/train" \
    --learning_rate_decay_factor=0 \
    --batch_size=$BS \
    --sequence_length=$SEQ_LEN \
    --nepochs=1 \
    --num_train_inst=$NUM_INST \
    --save_model_secs=1800 \
    --Glove_path=$GLOVE_PATH \
    --model_config="$MDL_CFGS/$CFG/train.json" &

python src/encoder_manager.py


