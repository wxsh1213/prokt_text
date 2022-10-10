# Proximal Knowledge Teaching for Neural Networks
This is the code for text classification in [Follow your path: a progressive method for knowledge distillation](https://link.springer.com/chapter/10.1007/978-3-030-86523-8_36) (ECML 2021).

Our codebase is modified on huggingface's [transformers](https://github.com/huggingface/transformers).

## Install

### Build the environment
conda create -n prokt_text python=3.7

### Install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
### Install transformer (+prokt)
pip install --editable ./
### And requirements
pip install -r ./examples/requirements.txt

## GLUE data:
Before running anyone of these GLUE tasks you should download the GLUE data by running this script and unpack it to some directory $GLUE_DIR.

python download_glue_data.py 

## Distill BERT-6 from BERT-base
Taking MNLI as an example:

    STUDENT="google/bert_uncased_L-6_H-768_A-12"
    TEACHER="bert-base-uncased"
    TASK=MNLI
    LR=1e-4
    LR_TEACHER=3e-5
    LAMBDA=0.4
    OUTPUT_DIR="checkpoints/MNLI_base_to_mid_0.4"
    EVAL_METRIC=acc

    python ./examples/text-classification/run_glue_milestone.py --model_name_or_path "$STUDENT:$TEACHER" --task_name $TASK --do_eval --data_dir glue_data/$TASK --max_seq_length 128 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --learning_rate $LR --num_train_epochs 6.0 --output_dir $OUTPUT_DIR --move_back_weight $LAMBDA --init_classifier_to_zero True --overwrite_output_dir --evaluate_during_training --best_checkpoint_metric $EVAL_METRIC --maximize_best_checkpoint_metric --do_train --train_teacher --fp16 --learning_rate_of_teacher $LR_TEACHER


STUDENT: the pre-trained student model.

TEACHER: the pre-trained teacher model.

TASK: task name.

LR: learning rate of student.

LR_TEACHER: learning rate of teacher.

LAMBDA: the weight of constraint term from student.

OUTPUT_DIR: directory of saved checkpoint.

EVAL_METRIC: evaluation metric for saving best results, choosing from acc/f1.

## Distill biLSTM from BERT-base

    TEACHER="bert-base-uncased"
    TASK=MNLI
    LR=1e-3
    LR_TEACHER=3e-5
    LAMBDA=0.3
    OUTPUT_DIR="checkpoints/MNLI_base_to_bilstm_0.3"
    EVAL_METRIC=acc

    python ./examples/text-classification/run_glue_milestone_bilstm.py --model_name_or_path ":$TEACHER" --task_name $TASK --do_eval --data_dir glue_data/$TASK --max_seq_length 128 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --learning_rate $LR --num_train_epochs 6.0 --output_dir $OUTPUT_DIR --move_back_weight $LAMBDA --init_classifier_to_zero True --overwrite_output_dir --evaluate_during_training --best_checkpoint_metric $EVAL_METRIC --maximize_best_checkpoint_metric --do_train --train_teacher --fp16 --learning_rate_of_teacher $LR_TEACHER --model_type bilstm --fp16 --sep_pair --comb_pair


