import argparse
import os
import yaml

from schedules import Full, S2L
from utils import get_tokenizer, smart_tokenizer_and_embedding_resize, get_model, rank0_print

## GET_SCHEDULES
def get_schedule(schedule_name):
    if schedule_name == "Full":
        return Full
    elif schedule_name == "S2L":
        return S2L
    
def set_default_values(args):
    # set default values
    if "ref_model_path" not in args:
        args["ref_model_path"] = None
    if "n_components" not in args:
        args["n_components"] = -1
    if "num_loss_ckpts" not in args:
        args["num_loss_ckpts"] = -1
    if "distance" not in args:
        args["distance"] = 'euclidean'
    if "seed" not in args:
        args["seed"] = 42
        
    return args


## RUN
def main(config_file):
    # load configuration
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    rank0_print('Configuration loaded!')
    
    # set default values
    args = set_default_values(args)
    rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    # makedirs    
    args["data_path_root"] = f"res/{args['result_dir_name']}/data"
    args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
    os.makedirs(args["data_path_root"], exist_ok=True)
    os.makedirs(args["output_dir_root"], exist_ok=True)

    # Initialize model and tokenizer
    model = get_model(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"])
    rank0_print('*** Model initialized!')
    
    tokenizer, special_tokens_dict = get_tokenizer(
        model_name_or_path=args["model_name_or_path"], 
        cache_dir=args["cache_dir"], 
        model_max_length=args["model_max_length"]
    )
    rank0_print('*** Tokenizer initialized!')
    
    tokenizer, model = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    rank0_print('*** Smart tokenizer and embedding resize done!')

    # Initialize schedule
    schedule = get_schedule(schedule_name=args["schedule_name"])(
        model=model,
        tokenizer=tokenizer,
        args=args
    )
    rank0_print('*** Schedule built!')

    # Initialize data
    schedule.initialize_labeled_data()
    
    schedule.save_labeled_unlabeled_data()
    rank0_print(f"*** Training-Data-Size = {len(schedule.labeled_idx[schedule.labeled_idx==True])}")

    # Train
    schedule.train()
    rank0_print("*** Training Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,)
    parser.add_argument('--wandb_key', type=str, required=True, help="wandb login key")
    args = parser.parse_args()
    
    import wandb
    wandb.login(key=args.wandb_key)
    
    main(config_file=args.config_file)
