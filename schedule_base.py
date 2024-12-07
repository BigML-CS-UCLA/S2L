import numpy as np
import json
import torch

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import jload, jdump, make_supervised_data_module, get_model, rank0_print


# Base Schedule
class Schedule:
    def __init__(self, 
        model, 
        tokenizer,
        args,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.full_data_path = args["full_data_path"]
        self.val_data = None
        self.init_label_num = args["init_label_num"] if "init_label_num" in args else 0
        # load full-sized source data -> for indexing all samples
        if self.full_data_path.endswith(".jsonl"):
            with open(self.full_data_path, "r") as f:
                self.train_data = [json.loads(line) for line in f]
            val_data_path = self.full_data_path.replace("train", "validate")
            with open(val_data_path, "r") as f:
                self.val_data = [json.loads(line) for line in f]
            self.train_idx = torch.arange(len(self.train_data))
            if 'topic' in self.train_data[0]:
                self.train_data = [{'instruction':data['instruction'], 'output':data['output'], 'source':data['source'], 'field':data['topic']} for data in self.train_data]
                self.val_data = [{'instruction':data['instruction'], 'output':data['output'], 'source':data['source'], 'field':data['topic']} for data in self.val_data]
            if 'instruction' not in self.train_data[0]:
                self.train_data = [{'instruction':data['input'], 'output':data['output'], 'source':data['source']} for data in self.train_data]
                self.val_data = [{'instruction':data['input'], 'output':data['output'], 'source':data['source']} for data in self.val_data]
        elif self.full_data_path.endswith(".json"):
            with open(self.full_data_path, "r") as f:
                self.train_data = json.load(f)  # fixed -> for indexing all samples
        elif 'MathInstruct' in self.full_data_path:
            list_data_dict = load_dataset(self.full_data_path)["train"]  # fixed -> for indexing all samples
            self.train_data = [list_data_dict[i] for i in range(len(list_data_dict))]
            
            self.train_idx = torch.arange(len(self.train_data))
            self.val_idx = None
        else:
            data_df = load_dataset(self.full_data_path)["train"]  # fixed -> for indexing all samples
            # convert to json format
            list_data_dict = []
            for i in range(len(data_df)):
                # parse data_df[i]['conversations'] from str to list
                list_data_dict.append(dict(instruction=data_df[i]['conversations'][0], output=data_df[i]['conversations'][1]))
            self.train_data = [list_data_dict[i] for i in range(len(list_data_dict))]
        
        # make a supervised data module for the valiation set
        if self.val_data is not None:
            self.val_data = make_supervised_data_module(tokenizer=self.tokenizer, data_path=self.val_data)
            
        self.n_pool = len(self.train_data)
        # keep track of labeled/unlabeled (1/0) index
        self.labeled_idx = torch.zeros(self.n_pool, dtype=bool)  
        # saving options
        self.data_path_root = args["data_path_root"]
        self.output_dir_root = args["output_dir_root"]
        train_args = args["train_args"]
        train_args["output_dir"] = self.output_dir_root  # dummy init -> to update for each round
        # get the name of the transformer model
        if "t5" in self.model.__class__.__name__:
            self.training_args = Seq2SeqTrainingArguments(**train_args)
        else:
            self.training_args = TrainingArguments(**train_args)

    def initialize_labeled_data(self):
        """Randomly init labeled pool"""
        if torch.distributed.get_rank() == 0:
            tmp_idxs = torch.randperm(self.n_pool)  # randomly permute indices (total_data_size, )
            self.labeled_idx[tmp_idxs[:self.init_label_num]] = True  # labeled=1, unlabeled=0 (total_data_size,)

    def save_labeled_unlabeled_data(self):
        """update & save current labaled & unlabeled pool"""
        if torch.distributed.get_rank() == 0:
            # obtain & check labeled_idx for current round
            labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]  # self.labeled_idx -> kept upated

            # query self.train_data -> current labeled & unlabeled data
            labeled_data_json_format = [self.train_data[_] for _ in labeled_idx] 
            unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]
            unlabeled_data_json_format = [self.train_data[_] for _ in unlabeled_idx]
            rank0_print(f"*** labeled_idx: {labeled_idx}")
            # save current labeled & unlabeld data
            labeled_data_path = f"{self.data_path_root}/labeled.json"
            labeled_idx_path = f"{self.data_path_root}/labeled_idx.npy"
            unlabeled_data_path = f"{self.data_path_root}/unlabeled.json"
            if torch.distributed.get_rank() == 0:
                retry = 0
                while True:
                    jdump(labeled_data_json_format, labeled_data_path)
                    try:
                        temp_labeled = jload(labeled_data_path)
                        rank0_print(f"*** jdump(labeled_data_json_format, labeled_data_path) SUCESSFUL to --> {labeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** jdump(labeled_data_json_format, labeled_data_path) FAILED to --> {labeled_data_path}")
                        if retry > 5:
                            raise
                        continue
                retry = 0
                while True:
                    jdump(unlabeled_data_json_format, unlabeled_data_path)
                    try:
                        temp_unlabeled = jload(unlabeled_data_path)
                        rank0_print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) SUCESSFUL to --> {unlabeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) FAILED to --> {unlabeled_data_path}")
                        if retry > 5:
                            raise
                        continue
                np.save(labeled_idx_path, labeled_idx.numpy())
    
    def get_updated_train_data(self):
        """load & make labeled data -> training data"""
        data_path = f"{self.data_path_root}/labeled.json"
        labeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, data_path=data_path)
        return labeled_data_module
    
    def get_unlabeled_data(self):
        """load & make unlabeled data -> candidate data pool for selecting new samples"""
        data_path = f"{self.data_path_root}/unlabeled.json"
        unlabeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, 
                                                                data_path=data_path)
        return unlabeled_data_module
    
    def train(self):
        # get labeled data -> for training
        data_module = self.get_updated_train_data()
        # sanity-check
        if torch.distributed.get_rank() == 0:
            for sanity_sample in data_module["train_dataset"]:
                break
            rank0_print(f"*** SANITY-CHECK: Training-Sample#1. - TEXT.:\n\n{self.tokenizer.decode(sanity_sample['input_ids'])}\n\n")
        
        # get validation data
        if self.val_data is not None:
            data_module["eval_dataset"] = self.val_data["train_dataset"]
        
        output_dir = f"{self.output_dir_root}/"
        self.training_args.output_dir = output_dir # update round-output-dir
        # check if the model is a seq2seq model
        if "t5" in self.model.__class__.__name__:
            trainer = Seq2SeqTrainer(model=self.model, 
                                     tokenizer=self.tokenizer, 
                                     args=self.training_args,
                                     **data_module)
        else:
            trainer = Trainer(model=self.model, 
                              tokenizer=self.tokenizer, 
                              args=self.training_args,
                              **data_module)
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        rank0_print(f"*** Trainer State & Trained Model Saved To --> {output_dir} ***")
        self.model.save_pretrained(f"{output_dir}/pretrained")  # save_model() somehow may result in error -> save_pretrained() again, just in case.
        rank0_print(f"*** Trainer State & Trained Model Save-Pretrained To --> {output_dir}/pretrained ***")
