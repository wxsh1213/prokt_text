# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import copy
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    # is_tpu_available,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



# if is_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


class KDCriterion(object):
    def __init__(self, soft_label_weight=0.0):
        self.soft_label_weight = soft_label_weight

    def forward(self, outputs, **kwargs):
        loss = outputs[0]
        if isinstance(loss, tuple):
            kld_loss, ce_loss = loss
            loss = (1 - self.soft_label_weight) * ce_loss + self.soft_label_weight * kld_loss
        return loss

class KDDualCriterion(object):
    def __init__(self, epsilon, lr, lambda_initialization=1.0):
        self.epsilon = epsilon
        self.lr = lr
        self.beta = lambda_initialization

    def forward(self, outputs, **kwargs):
        loss = outputs[0]
        if isinstance(loss, tuple):
            kld_loss, ce_loss = loss
            loss = ce_loss + self.beta * (kld_loss - self.epsilon)
            # print(ce_loss, kld_loss, loss, self.beta)
        # print(loss)
        return loss

    def update_beta(self, constraint_loss):
        self.beta = max(0, self.beta + self.lr * constraint_loss)
        # print(self.beta)
        # print(loss)

    def get_constraint_loss(self, outputs):
        loss = outputs[0]
        if isinstance(loss, tuple):
            kld_loss, ce_loss = loss
            return (kld_loss - self.epsilon)
        else:
            return 0


class softLossCriterion(nn.Module):
    def __init__(self, teacher_config, student_config, use_attn=True, use_emb=True, use_hid=True, soft_label_weight=0.0):
        super().__init__()

        self.use_attn = use_attn
        self.use_emb = use_emb
        self.use_hid = use_hid
        self.teacher_config = teacher_config
        self.student_config = student_config
        self.soft_label_weight = soft_label_weight

        if use_hid:
            self.hid_w = nn.Linear(student_config.hidden_size, teacher_config.hidden_size, bias=False)
            self.mse_hid = torch.nn.MSELoss()
        if use_emb:
            self.emb_w = nn.Linear(student_config.hidden_size, teacher_config.hidden_size, bias=False)
            self.mse_emb = torch.nn.MSELoss()
        if use_attn:
            self.loss_attn = torch.nn.MSELoss() # torch.nn.KLDivLoss(reduction="batchmean")  # TODO: or MSELoss
        pass

    def forward(self, outputs, **kwargs):
        teacher_outputs = kwargs.get("teacher_outputs", None)
        if teacher_outputs is None:
            raise ValueError("We need teacher outputs for softLossCriterion.")

        loss = outputs[0]
        if isinstance(loss, tuple):
            kld_loss, ce_loss = loss
            loss = (1 - self.soft_label_weight) * ce_loss + self.soft_label_weight * kld_loss


        if self.use_hid:
            hidden_states_of_student = outputs[2][1:]  # Batch_size x seq_len x hid_size
            hidden_states_of_teacher = teacher_outputs[2][1:]
            mulk = len(hidden_states_of_teacher) // len(hidden_states_of_student)
            hidden_states_of_teacher = hidden_states_of_teacher[::mulk]
            assert len(hidden_states_of_student) == len(hidden_states_of_teacher)
            hidden_states_of_student = torch.cat(hidden_states_of_student, dim=1)
            hidden_states_of_student = hidden_states_of_student.view(-1, hidden_states_of_student.size(-1))
            hidden_states_of_teacher = torch.cat(hidden_states_of_teacher, dim=1)
            hidden_states_of_teacher = hidden_states_of_teacher.view(-1, hidden_states_of_teacher.size(-1))
            mapped_hidden_states_of_student = self.hid_w(hidden_states_of_student)
            hid_loss = self.mse_hid(input=mapped_hidden_states_of_student, target=hidden_states_of_teacher.detach())
            # print(hid_loss)
            loss += hid_loss
        if self.use_emb:
            emb_of_student = outputs[2][0]
            emb_of_teacher = teacher_outputs[2][0]
            emb_of_student = emb_of_student.view(-1, emb_of_student.size(-1))
            emb_of_teacher = emb_of_teacher.view(-1, emb_of_teacher.size(-1))
            mapped_emb_of_student = self.emb_w(emb_of_student)
            emb_loss = self.mse_hid(input=mapped_emb_of_student, target=emb_of_teacher.detach())
            loss += emb_loss
            # print(emb_loss)
        if self.use_attn:
            attentions_of_student = outputs[3]  # Batch_size x head x seq_len x seq_len
            attentions_of_teacher = teacher_outputs[3]

            mulk = len(attentions_of_teacher) // len(attentions_of_student)
            attentions_of_teacher = attentions_of_teacher[::mulk]

            assert len(attentions_of_student) == len(attentions_of_teacher)

            attentions_of_student = torch.cat(attentions_of_student, dim=1)
            attentions_of_student = attentions_of_student.view(-1, attentions_of_student.size(-1))
            attentions_of_teacher = torch.cat(attentions_of_teacher, dim=1)
            attentions_of_teacher = attentions_of_teacher.view(-1, attentions_of_teacher.size(-1))

            # print(attentions_of_student)
            # print(torch.sum(attentions_of_student, dim=-1))
            attn_loss = self.loss_attn(input=attentions_of_student, target=attentions_of_teacher.detach())

            loss += attn_loss
            # print(attn_loss)

        return loss


def init_classifier_as_zero(model):
    try:
        for params in model.classifier.parameters():
            params.data.fill_(0.0)
    except Exception as e:
        for params in model.module.classifier.parameters():
            params.data.fill_(0.0)
        

class loss_record():
    def __init__(self, args):
        self.best_checkpoint_metric = getattr(args, "best_checkpoint_metric", "loss")
        self.maximize_best_checkpoint_metric = getattr(args, "maximize_best_checkpoint_metric", False)
        self._record = []
        self.best_index = -1

    def add(self, loss_dict, step):
        self._record.append({"step": step, "loss_dict": loss_dict})
        key_key = "eval_" + self.best_checkpoint_metric
        key_values = [x["loss_dict"][key_key] for x in self._record]
        func = max if self.maximize_best_checkpoint_metric else min
        best_value = func(key_values)
        best_index = key_values.index(best_value)
        self.best_index = best_index

    def get_best(self):
        if self.best_index > -1:
            return self._record[self.best_index]
        else:
            print("No any loss record...")
            return None



def train(trainer_student, trainer_teacher, model_path_student, model_path_teacher):
    """
            Main training entry point.

            Args:
                model_path:
                    (Optional) Local path to model if model to train has been instantiated from a local path
                    If present, we will try reloading the optimizer/scheduler states from there.
            """
    train_dataloader = trainer_student.get_train_dataloader()
    if trainer_student.args.max_steps > 0:
        t_total = trainer_student.args.max_steps
        num_train_epochs = (
                trainer_student.args.max_steps // (len(train_dataloader) // trainer_student.args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = int(len(train_dataloader) // trainer_student.args.gradient_accumulation_steps * trainer_student.args.num_train_epochs)
        num_train_epochs = trainer_student.args.num_train_epochs

    # Prepare ...
    model_t, optimizer_t, scheduler_t, total_train_batch_size = trainer_teacher.training_prepare(t_total, model_path=model_path_teacher)
    model_s, optimizer_s, scheduler_s, total_train_batch_size = trainer_student.training_prepare(t_total, model_path=model_path_student)

    if trainer_teacher.args.init_classifier_to_zero and trainer_teacher.args.train_teacher:
        init_classifier_as_zero(model_t)
    if trainer_student.args.init_classifier_to_zero:
        init_classifier_as_zero(model_s)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", trainer_student.num_examples(train_dataloader))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", trainer_student.args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", trainer_student.args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer_student.global_step = 0
    trainer_student.epoch = 0
    trainer_teacher.global_step = 0
    trainer_teacher.epoch = 0


    # Build criterion
    if trainer_teacher.args.dual_optimization:
        criterion_t = KDDualCriterion(epsilon=trainer_teacher.args.lambda_epsilon,
                                      lr=trainer_teacher.args.lambda_lr,
                                      lambda_initialization=trainer_teacher.args.lambda_initialization)
    else:
        criterion_t = KDCriterion(soft_label_weight=trainer_teacher.args.move_back_weight)
    if trainer_student.args.hidden_alignment:
        criterion_s = softLossCriterion(trainer_teacher.model.config, trainer_student.model.config,
                                        use_attn=not trainer_teacher.args.no_attn,
                                        use_emb=not trainer_teacher.args.no_emb,
                                        use_hid=not trainer_teacher.args.no_hid,
                                        soft_label_weight=1.0).cuda()
    else:
        criterion_s = KDCriterion(soft_label_weight=1.0)

    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    def _check_continuing_training_from_checkpoint(model_path, trainer):
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                trainer.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = trainer.global_step // (len(train_dataloader) // trainer.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = trainer.global_step % (
                        len(train_dataloader) // trainer.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", trainer.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                trainer.global_step = 0
                logger.info("  Starting fine-tuning.")

    print("*** Teacher ***")
    _check_continuing_training_from_checkpoint(model_path_teacher, trainer_teacher)
    print("*** Student ***")
    _check_continuing_training_from_checkpoint(model_path_student, trainer_student)

    def _optim_step(model, trainer, optimizer, scheduler):
        if (step + 1) % trainer.args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= trainer.args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
        ):
            if trainer.args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), trainer.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm)

            optimizer.step()

            scheduler.step()
            model.zero_grad()

            trainer.global_step += 1
            trainer.epoch = epoch + (step + 1) / len(epoch_iterator)

    tr_loss = 0.0
    tr_loss_teacher = 0.0
    logging_loss_s = 0.0
    logging_loss_t = 0.0
    loss_record_teacher = loss_record(trainer_teacher.args)
    loss_record_student = loss_record(trainer_student.args)

    model_t.zero_grad()
    model_s.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=not trainer_student.is_local_master()
    )
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not trainer_student.is_local_master())
        for step, inputs in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if trainer_student.args.train_teacher:
                model_t.train()
                if trainer_student.args.dual_optimization:
                    student_output, _ = trainer_student.output(inputs, model_s)
                    loss, constraint_loss = trainer_teacher._training_step(model_t, student_output, optimizer_t,
                                                                           criterion=criterion_t)
                    tr_loss_teacher += loss
                    criterion_t.update_beta(constraint_loss)  # TODO: 在这里更新拉格朗日系数
                else:
                    if trainer_student.args.move_back_weight > 0.0:
                        student_output, _ = trainer_student.output(inputs, model_s)
                        tr_loss_teacher += trainer_teacher._training_step(model_t, student_output, optimizer_t, criterion=criterion_t)
                    else:
                        tr_loss_teacher += trainer_teacher._training_step(model_t, inputs, optimizer_t, criterion=criterion_t)

                _optim_step(model_t, trainer_teacher, optimizer_t, scheduler_t)


            teacher_output, teacher_outputs = trainer_teacher.output(inputs, model_t)  # TODO: teacher_output

            model_s.train()
            tr_loss += trainer_student._training_step(model_s, teacher_output, optimizer_s, criterion=criterion_s, teacher_outputs=teacher_outputs)
            _optim_step(model_s, trainer_student, optimizer_s, scheduler_s)

            if (step + 1) % trainer_student.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= trainer_student.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):

                    if trainer_student.is_local_master():
                        if (trainer_student.args.logging_steps > 0 and trainer_student.global_step % trainer_student.args.logging_steps == 0) or (
                                trainer_student.global_step == 1 and trainer_student.args.logging_first_step
                        ):
                            logs: Dict[str, float] = {}
                            logs["loss_student"] = (tr_loss - logging_loss_s) / trainer_student.args.logging_steps
                            logs["learning_rate"] = scheduler_s.get_last_lr()[0]
                            logging_loss_s = tr_loss

                            trainer_student._log(logs)

                            logs: Dict[str, float] = {}
                            logs["loss_teacher"] = (tr_loss_teacher - logging_loss_t) / trainer_student.args.logging_steps
                            logs["learning_rate"] = scheduler_t.get_last_lr()[0]
                            logging_loss_t = tr_loss_teacher

                            trainer_teacher._log(logs)

                            if trainer_student.args.evaluate_during_training:
                                print("*** Student ***")
                                eval_res = trainer_student.evaluate()
                                loss_record_student.add(eval_res, trainer_student.global_step)
                            if trainer_teacher.args.evaluate_during_training:
                                print("*** Teacher ***")
                                eval_res = trainer_teacher.evaluate()
                                loss_record_teacher.add(eval_res, trainer_teacher.global_step)

                        if trainer_student.args.save_steps > 0 and trainer_student.global_step % trainer_student.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model_s, "module"):
                                assert model_s.module is trainer_student.model
                            else:
                                assert model_s is trainer_student.model
                            # Save model checkpoint
                            output_dir = os.path.join(
                                trainer_student.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{trainer_student.global_step}"
                            )

                            trainer_student.save_model(output_dir)
                            trainer_student._rotate_checkpoints()
                            torch.save(optimizer_s.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler_s.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if trainer_student.args.max_steps > 0 and trainer_student.global_step > trainer_student.args.max_steps:
                epoch_iterator.close()
                break
        if trainer_student.args.max_steps > 0 and trainer_student.global_step > trainer_student.args.max_steps:
            train_iterator.close()
            break

        # if trainer_student.args.tpu_metrics_debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

    if trainer_student.tb_writer:
        trainer_student.tb_writer.close()
    if trainer_teacher.tb_writer:
        trainer_teacher.tb_writer.close()

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

    return trainer_student.pack_train_output(tr_loss), loss_record_teacher, loss_record_student


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        _, _, training_args_of_teacher = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # split model_name_or_path into teacher / student models.
    try:
        student_model_name_or_path, teacher_model_name_or_path = model_args.model_name_or_path.split(":")
    except Exception as e:
        raise ValueError("Fail to split model_name_or_path into student & teacher model path.")

    # Build for teacher:
    config_teacher = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else teacher_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    # TODO: tokenizer应该是同一个。。
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if training_args.hidden_alignment:
        config_teacher.output_hidden_states = True
        config_teacher.output_attentions = True

    model_teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_name_or_path,
        from_tf=bool(".ckpt" in teacher_model_name_or_path),
        config=config_teacher,
        cache_dir=model_args.cache_dir,
    )
    # print(model_teacher)

    # Build for student:
    config_student = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else student_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # tokenizer_student = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else student_model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )
    if training_args.hidden_alignment:
        config_student.output_hidden_states = True
        config_student.output_attentions = True

    model_student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name_or_path,
        from_tf=bool(".ckpt" in student_model_name_or_path),
        config=config_student,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_args_test = dataclasses.replace(data_args, overwrite_cache=True)
    test_dataset = GlueDataset(data_args_test, tokenizer=tokenizer, evaluate=True,
                               eval_set="test") if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # Initialize our Trainer
    # TODO: 对于student和teacher构建两个不同的trainer
    trainer_student = Trainer(
        model=model_student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    from argparse import Namespace
    # training_args_of_teacher = Namespace(**vars(training_args))
    training_args_of_teacher.learning_rate=training_args_of_teacher.learning_rate_of_teacher
    trainer_teacher = Trainer(
        model=model_teacher,
        args=training_args_of_teacher,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer = trainer_student
    if training_args.do_train:
        # TODO: 这里把training的步骤从trainer里面拆出来。

        _, train_record_teacher, train_record_student = \
            train(trainer_student, trainer_teacher, student_model_name_or_path, teacher_model_name_or_path)

        # trainer.train(
        #     model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        # )


        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                if training_args.do_train and train_record_student is not None:
                    logger.info("***** Best Eval results of student {} *****".format(eval_dataset.args.task_name))
                    result = train_record_student.get_best()
                    logger.info("best step %d", result["step"])
                    writer.write("best step %d\n" % result["step"])
                    for key, value in result["loss_dict"].items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                if training_args.do_train and train_record_teacher is not None:
                    logger.info("***** Best Eval results of teacher {} *****".format(eval_dataset.args.task_name))
                    result = train_record_teacher.get_best()
                    logger.info("best step %d", result["step"])
                    writer.write("best step %d\n" % result["step"])
                    for key, value in result["loss_dict"].items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            results.update(result)

        if training_args.do_predict:
            result = trainer.predict(test_dataset=test_dataset)  # test_dataset


            task_name = data_args.task_name.upper()
            if data_args.task_name == "mnli":
                task_name += "-m"

            if training_args.out_predict_file == "":
                out_predict_file = os.path.join(training_args.output_dir, task_name + ".tsv")
            else:
                out_predict_file = os.path.join(training_args.out_predict_file, task_name + ".tsv")

            fout_test_tsv = open(out_predict_file, "w")
            preds = np.argmax(result.predictions, axis=1)
            if result.label_ids is not None:
                print((preds == result.label_ids).mean())
            label_map = test_dataset.processor.get_labels()
            with open(out_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for i, xx in enumerate(test_dataset):
                    writer.write(test_dataset.uids[i] + "\t" + label_map[preds[i]] + "\n")

            if data_args.task_name == "mnli":
                mnli_mm_data_args = dataclasses.replace(data_args_test, task_name="mnli-mm")
                test_dataset = GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True, eval_set="test")

                result = trainer.predict(test_dataset=test_dataset)  # test_dataset

                task_name = data_args.task_name.upper() + "-mm"
                if training_args.out_predict_file == "":
                    out_predict_file = os.path.join(training_args.output_dir, task_name + ".tsv")
                else:
                    out_predict_file = os.path.join(training_args.out_predict_file, task_name + ".tsv")

                fout_test_tsv = open(out_predict_file, "w")
                preds = np.argmax(result.predictions, axis=1)
                if result.label_ids is not None:
                    print((preds == result.label_ids).mean())
                label_map = test_dataset.processor.get_labels()
                with open(out_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for i, xx in enumerate(test_dataset):
                        writer.write(test_dataset.uids[i] + "\t" + label_map[preds[i]] + "\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
