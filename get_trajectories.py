import argparse
import torch
import yaml
from tqdm import tqdm
from utils import rank0_print, get_model, get_tokenizer, smart_tokenizer_and_embedding_resize, make_supervised_data_module
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def loss(data, model):
    """compute last hidden states for a data_module"""
    model.cuda()
    model.eval()
    
    losses = []
    
    with torch.no_grad():
        for _,datapoint in tqdm(enumerate(data["train_dataset"]), total=len(data["train_dataset"])):
            input_ids = datapoint["input_ids"].unsqueeze(0).cuda()
            labels = datapoint["labels"].unsqueeze(0).cuda()
            result = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = result.loss
            if _==1 or (_!=0 and _%10000 == 0): # report progress
                rank0_print(f"***** Predict-Progress -- {_} DONE !")
            losses.append(loss.detach().cpu())
    return losses

def main(model_path, config_file=None, ckpt=-1):
    if config_file:
        # Local model path logic
        with open(config_file, 'r') as f:
            args = yaml.full_load(f)
        rank0_print('Configuration loaded!')
        rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

        args["data_path_root"] = f"res/{args['result_dir_name']}/data"
        args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
        
        if ckpt == -1:
            model_path = args["output_dir_root"]+f"/"
        else:
            model_path = args["output_dir_root"]+f"/checkpoint-{ckpt}"
            
        loss_file = f"{model_path}/losses.pt"
    else:
        # HuggingFace model path logic
        args = {
            "cache_dir": None,
            "model_max_length": 2048,  # You might want to make this configurable
            "model_name_or_path": model_path
        }
        if ckpt != -1:
            model_path = f"{model_path}@{ckpt}"
        
        # Create a default output directory for HF models
        os.makedirs("hf_outputs", exist_ok=True)
        loss_file = f"hf_outputs/{model_path.replace('/', '_')}_losses.pt"

    if os.path.exists(loss_file):
        rank0_print(f"***** Losses already exist at {loss_file}!")
        return
    
    rank0_print(f"***======================================================================================================")
    rank0_print(f"***** Checkpoint {ckpt} ======================================================================================================")
    model = get_model(model_name_or_path=model_path, cache_dir=args["cache_dir"])
    rank0_print(f'***** Model loaded from {model_path}!') 
    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
    rank0_print(f'***** Tokenizer initilized!')
    tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                            tokenizer=tokenizer, 
                                                            model=model)  # fix tokenizer's special_token_maps
    rank0_print(f'***** smart_tokenizer_and_embedding_resize done!')
    all_data = make_supervised_data_module(tokenizer=tokenizer, data_path=args["full_data_path"])

    mean_entropies_all = loss(data=all_data, model=model)
    torch.save(mean_entropies_all, loss_file)
    print(f"***** Losses saved to {loss_file}")      
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default=None,
                        help='Config file path for local models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--ckpt', type=int, default=-1,)
    args = parser.parse_args()
    
    main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)