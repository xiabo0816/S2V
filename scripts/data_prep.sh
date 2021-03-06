
NUM_WORDS=50001
OUTPUT_DIR="TFRecords" 
TOKENIZED_FILES="Questions_utf8_2.seg.utf8"

python src/data/preprocess_dataset.py \
  --input_files "$TOKENIZED_FILES" \
  --output_dir $OUTPUT_DIR \
  --num_words $NUM_WORDS \
  --max_sentence_length 50 \
  --case_sensitive
