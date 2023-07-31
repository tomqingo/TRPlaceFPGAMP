"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.text_datasets import load_data_text

import matplotlib.pyplot as plt

import time
from diffuseq.utils import dist_util, logger
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=1,
        split='test',
        metric_enc=model.metric_enc,
        place_enc=model.place_enc,
        region_enc=model.region_enc,
        device = args.device,
        loop=False,
    )

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_valid)
            all_test_data.append(cond)
            break

    except StopIteration:
        print('### End of reading iteration...')
    
    from tqdm import tqdm

    for cond in tqdm(all_test_data[:100]):
        with th.no_grad():

            raw_input = cond.pop('raw_input').to(dist_util.dev())
            emb = cond.pop('emb').to(dist_util.dev())
            mask = cond.pop('mask').to(dist_util.dev())
            mask_decoder = cond.pop('mask_decoder').to(dist_util.dev())  
            emb_mask = cond.pop('emb_mask').to(dist_util.dev())    
            ordering = cond.pop('ordering')
            design_name = cond.pop('design_name')

            len_place = int(th.sum(mask_decoder).item()) - 1
            place_emb_size = int(th.sum(emb_mask[0][1]).item())
          
            x_start = emb
            noise = th.randn_like(x_start)

            # input_ids_mask = mask
            # input_ids_mask_original = mask
            # input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
            # x_noised = th.where(input_ids_mask==0, x_start, noise)

            input_ids_mask = emb_mask
            x_noised = noise * emb_mask + x_start * (th.ones_like(emb_mask) - emb_mask)

            model_kwargs = {}

            if args.step == args.diffusion_steps:
                args.use_ddim = False
                step_gap = 1
            else:
                args.use_ddim = True
                step_gap = args.diffusion_steps//args.step

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

            samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap,
            )

            sample = samples[-1]
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

            arr = np.concatenate(all_sentence, axis=0)
            x_t = th.tensor(arr).cuda()

            reshaped_x_t = x_t

            
            res_place = th.softmax(model.get_place(reshaped_x_t[:,1:,:place_emb_size]), dim=-1)[0]
            # direct_max_res = th.argmax(res_place, dim=-1).cpu().numpy().tolist()[:len_place]
            # num_duplicate = len(direct_max_res) - len(set(direct_max_res))
            # print('#duplicate:', num_duplicate)

            unduplicate_res = []
            occupied = set([2281,2282])
            for i in range(res_place.shape[0]):
                if len(unduplicate_res) == len_place:
                    break

                cur_pos_distribution = res_place[i]
                selected = False
                while not selected:
                    choice = th.argmax(cur_pos_distribution, dim=-1).item()
                    if choice in occupied:
                        cur_pos_distribution[choice] = 0
                        continue
                    else:
                        unduplicate_res.append(choice)
                        occupied.add(choice)
                        selected = True

            print('Loading site-to-index mapping...')
            with open('./dataset/site2idx.json', mode='r', encoding='utf-8') as f:
                dicts = json.load(f)
                dict_DSP = dicts[0]
                dict_BRAM = dicts[1]
                dict_URAM = dicts[2]

            dict_DSP_trans = dict([val, key] for key, val in dict_DSP.items())
            unduplicate_res = [dict_DSP_trans[str(i)] for i in unduplicate_res]

            with open('./generated_results/res_%s.txt' % design_name[0].split('@')[0], mode='w', encoding='utf-8') as f_res:
                macro_in_order = np.arange(len_place)[np.squeeze(ordering.numpy())].tolist()
                for m in range(len(macro_in_order)):
                    f_res.write(str(int(macro_in_order[m])) + ' ')
                    f_res.write(unduplicate_res[m] + '\n')
                print('results saved to ./generated_results/res_%s.txt' % design_name[0].split('@')[0])

            raise NotImplementedError

if __name__ == "__main__":
    main()
