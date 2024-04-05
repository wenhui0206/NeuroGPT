DATA_ROOT=$1
EXPORT_DIR=$2
python3 ../src/preprocess.py \
    --data-root $DATA_ROOT \
    --export-dir $EXPORT_DIR