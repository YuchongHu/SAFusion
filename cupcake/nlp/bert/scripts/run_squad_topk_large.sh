
echo "Container nvidia build = " $NVIDIA_BUILD_ID

# export DIR_Model="/data/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16"
export DIR_Model="/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16"
export DIR_DataSet="/data/dataset/nlp/bert"


# init_checkpoint=${1:-"/data/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12/bert_model.ckpt"}
# init_checkpoint=${1:-"$DIR_Model/bert_model.ckpt"}
init_checkpoint=${1:-"$DIR_Model/bert_large_pretrained_amp.pt"}
epochs=${2:-"30.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp16"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"$DIR_DataSet/squad"}
vocab_file=${10:-"$DIR_Model/vocab.txt"}


# 输出模型和预测结果
# OUT_DIR=${11:-"./squad_large/actopk/8"}
# OUT_DIR=${11:-"./squad_large/compression_rate/01"}
OUT_DIR=${11:-"./horovod/example/elastic/pytorch/nlp/bert/scripts/squad_large/squad_topk_001"}


# train+eval
mode=${12:-"train eval"}
# mode=${12:-"train"}
CONFIG_FILE=${13:-"$DIR_Model/bert_config.json"}
max_steps=${14:-"-1"}


# setup
density="${density:-0.1}"
# density="${density:-0.05}"
# density="${density:-0.1}"
threshold="${threshold:-8192}"
# compressor="${compressor:-sidco}"
compressor="${compressor:-dgc}"
# max_epochs="${max_epochs:-200}"
memory="${memory:-residual}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  # mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
  # mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

# CMD="python  $mpi_command ../run_squad_hvd.py "
CMD="HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 "
# CMD=" horovodrun -np 4 -H n15:1,n16:1,n17:1,n18:1  python ../run_squad_topk.py "

# CMD=" horovodrun -np 8 -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python ../run_squad_topk.py "
# CMD=" horovodrun  -np 4 -H  n16:1,n15:1,n19:1,n20:1  python ../run_squad_topk.py "

# CMD=" horovodrun -np 7 -H n15:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python ../run_squad_topk.py  "
# CMD=" horovodrun -np 4 -H  n15:1,n16:1,n19:1,n20:1   python ../run_squad_topk.py  "
CMD=" horovodrun  -np 8  -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1   python ../run_squad_topk.py  "
CMD+="--init_checkpoint=$init_checkpoint  "
CMD+="--density=$density  "
CMD+="--compressor=$compressor  "
CMD+="--threshold  $threshold  "


if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
# CMD+=" --bert_model=bert-large-uncased "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
# CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE


# Buffers = baseline
# [0]<stdout>:topk_time =  25.81660270690918
# [0]<stdout>:threshold_time =  0
# [0]<stdout>:io_time =  0.00018668174743652344
# [0]<stdout>:forward_time =  2.9541423320770264
# [0]<stdout>:backward_time =  8.653934240341187
# [0]<stdout>:step_time =  45.41554594039917
# [0]<stdout>:communication_time =  35.88586664199829
# [0]<stdout>:para_update_time =  0.4444699287414551
# [0]<stdout>:hook_time =  28.292547702789307
# [0]<stdout>:---------------------------------
# [0]<stdout>:step_time =  82.96324157714844


# Buffers = 20
# [0]<stdout>:topk_time =  36.332496643066406
# [0]<stdout>:threshold_time =  0
# [0]<stdout>:io_time =  0.0001952648162841797
# [0]<stdout>:forward_time =  2.971493721008301
# [0]<stdout>:backward_time =  11.621936321258545
# [0]<stdout>:step_time =  44.319305419921875
# [0]<stdout>:communication_time =  34.63810753822327
# [0]<stdout>:para_update_time =  0.6015372276306152
# [0]<stdout>:hook_time =  41.98200297355652
# [0]<stdout>:---------------------------------
# [0]<stdout>:step_time =  95.34632015228271


# Buffers = 20
# <stderr>:Iteration:  11%|█         | 303/2771 [04:45<38:23,  1.07it/s]
# [0]<stdout>:topk_time =  51.11087489128113
# [0]<stdout>:threshold_time =  0
# [0]<stdout>:io_time =  0.0002315044403076172
# [0]<stdout>:forward_time =  2.968623638153076
# [0]<stdout>:backward_time =  9.148458480834961
# [0]<stdout>:step_time =  30.529781103134155
# [0]<stdout>:communication_time =  21.09539008140564
# [0]<stdout>:para_update_time =  0.43097448348999023
# [0]<stdout>:hook_time =  53.64043974876404
# [0]<stdout>:---------------------------------
# [0]<stdout>:step_time =  93.94350051879883


# Buffers = baseline
# [1]<stderr>:Iteration:   2%|▏         | 64/2771 [01:15<50:17,  1.11s/it] 
# [0]<stdout>:topk_time =  62.072439432144165
# [0]<stdout>:threshold_time =  0
# [0]<stdout>:io_time =  0.00020003318786621094
# [0]<stdout>:forward_time =  3.019441843032837
# [0]<stdout>:backward_time =  12.39412808418274
# [0]<stdout>:step_time =  36.28776025772095
# [0]<stdout>:communication_time =  26.753859519958496
# [0]<stdout>:para_update_time =  0.6175704002380371
# [0]<stdout>:hook_time =  68.02856421470642
# [0]<stdout>:---------------------------------
# [0]<stdout>:step_time =  113.88558006286621




