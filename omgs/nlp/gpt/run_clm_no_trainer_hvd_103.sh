

OUT_DIR=${OUT_DIR:-"./log"}
epochs="${epochs:-30}"
density="${density:-0.1}"
compressor="${compressor:-topkef}"
memory="${memory:-residual}"
threshold="${threshold:-8192}"
percent="${percent:-0}"
per_device_train_batch_size="${per_device_train_batch_size:-2}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi



CMD=" HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 "
CMD=" horovodrun  -np  4 -H  node15:1,node16:1,node19:1,node20:1  python run_clm_no_trainer_hvd.py   "
# CMD=" horovodrun  -np  7 -H  node15:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1  python run_clm_no_trainer_hvd.py   "
CMD+=" --dataset_name /data/dataset/nlp/openai-community/wikitext-103-raw-v1 --dataset_config_name default  "
CMD+=" --model_name_or_path /data/dataset/nlp/openai-community/gpt2 "
# CMD+=" --model_name_or_path /data/dataset/nlp/openai-community/gpt2-medium "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent  "
CMD+=" --per_device_train_batch_size=$per_device_train_batch_size "


LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE



