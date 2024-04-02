"""
Train the diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train_util import TrainLoop
from transformers import set_seed
# Learn how to use wandb
import wandb

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    # dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    torch.multiprocessing.set_start_method('spawn')

    logger.log("### Creating model and diffusion...")

    # create the diffusion model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.to(args.device)

    # load the training and validation set 
    data, data_valid = load_data_text( 
        batch_size=args.batch_size,
        split='train',
        metric_enc=model.metric_enc,
        dsp_place_enc=model.dsp_place_enc,
        bram_place_enc=model.bram_place_enc,
        region_enc=model.region_enc,      
        device = args.device,
    )

    # The number of the parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')

    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        device=args.device,
    ).run_loop()

if __name__ == "__main__":
    main()
