#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR  # entering "xxx/fairseq" directory

# Install fairseq
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128
pip3 install --editable .

# Print parameters
echo $@

# Copy soft link
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/data-bin/ data-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/checkpoints/ checkpoints
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/cmlm_checkpoints/ cmlm_checkpoints
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/MNLI-bin/ MNLI-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/CoLA-bin/ CoLA-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/MRPC-bin/ MRPC-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/QNLI-bin/ QNLI-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/QQP-bin/ QQP-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/RTE-bin/ RTE-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/SST-2-bin/ SST-2-bin
ln -s /mnt/cephfs_new_wj/mlnlp/shiwenxian/fairseq/SST-B-bin/ SST-B-bin

# Begin to train
fairseq-train $@
