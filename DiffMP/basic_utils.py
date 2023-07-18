import argparse
import torch
import json, os
import time

from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from linear_attention_transformer.linear_attention_transformer import LinearAttentionTransformerLM


def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('diffuseq/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    device,
    notes,
    **kwargs,
):

    ### Old transformer model
    # model = TransformerNetModel(
    #     device=device,
    # )

    ### New transformer model -> save memory and handle longer sequence
    model = LinearAttentionTransformerLM(
        num_tokens = 2282,
        dim = 512,
        heads = 8,
        depth = 1,
        # max_seq_len = 2282,
        max_seq_len = 2304,
        causal = True,                  # auto-regressive or not
        ff_dropout = 0.1,               # dropout for feedforward
        attn_layer_dropout = 0.1,       # dropout right after self-attention layer
        attn_dropout = 0.1,             # dropout post-attention
        emb_dim = 128,                  # embedding factorization, to save on memory
        dim_head = 128,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
        blindspot_size = 64,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
        n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
        local_attn_window_size = 128,   # receptive field of the local attention
        reversible = True,              # use reversible nets, from Reformer paper
        ff_chunks = 2,                  # feedforward chunking, from Reformer paper
        ff_glu = True,                  # use GLU variant for feedforward
        attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
        shift_tokens = True,            # add single token shifting, for great improved convergence
        device = device,
    )

    ### TODO: under implementation
    # model = ReformerLM(
    #     num_tokens= 2282,
    #     dim = 512,
    #     depth = 12,
    #     max_seq_len = 2282,
    #     heads = 8,
    # )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
