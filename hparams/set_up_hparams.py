import argparse
from .defaults.mbld_defaults import HparamsMBLD, add_mbld_args
from .defaults.mbae_default import HparamsMBAE, add_mbae_args
# from .defaults.experiment_defaults import add_PRDC_args, add_sampler_FID_args, add_big_sample_args


# args for training of all models: dataset, EMA and loading
def add_training_args(parser):
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema", const=True, action="store_const", default=False)
    parser.add_argument("--load_dir", type=str, default="test")
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000000)
    parser.add_argument("--train", type=str, default=True)
    parser.add_argument("--val", type=str, default=False)
    parser.add_argument("--test", type=str, default=False)
    parser.add_argument("--debug",  type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--motions_path", type=str, default="data/motionx/motions")
    parser.add_argument("--texts_path", type=str, default="data/motionx/texts")


# args required for logging
def add_logging_args(parser):
    parser.add_argument("--log_dir", type=str, default="test")
    parser.add_argument("--save_individually", const=True, action="store_const", default=False)
    parser.add_argument("--steps_per_checkpoint", type=int, default=25000)
    parser.add_argument("--steps_per_display_output", type=int, default=5000)
    parser.add_argument("--steps_per_eval", type=int, default=0)
    parser.add_argument("--steps_per_log", type=int, default=10)
    parser.add_argument("--steps_per_save_output", type=int, default=5000)
    parser.add_argument("--visdom_port", type=int, default=8097)
    parser.add_argument("--visdom_server", type=str)


def set_up_base_parser(parser):
    add_training_args(parser)
    add_logging_args(parser)


def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H


def get_mbae_hparams():
    parser = argparse.ArgumentParser("Parser for setting up MBAE training :)")
    set_up_base_parser(parser)
    add_mbae_args(parser)
    parser_args = parser.parse_args()
    H = HparamsMBAE(parser_args.dataset, parser_args.dataset_type)
    H = apply_parser_values_to_H(H, parser_args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H


# def get_sampler_H_from_parser(parser):
#     parser_args = parser.parse_args()
#     dataset = parser_args.dataset

#     # has to be in this order to overwrite duplicate defaults such as batch_size and lr
#     H = HparamsMBAE(dataset)
#     H.vqgan_batch_size = H.batch_size  # used for generating samples and latents

#     if parser_args.sampler == "bld":
#         H_sampler = HparamsMBLD(dataset)
#     else:
#         raise NotImplementedError
#     H.update(H_sampler)  # overwrites old (vqgan) H.batch_size
#     H = apply_parser_values_to_H(H, parser_args)
#     return H


def get_mbld_H_from_parser(parser):
    parser_args = parser.parse_args()
    dataset = parser_args.dataset
    dataset_type = parser_args.dataset_type

    # has to be in this order to overwrite duplicate defaults such as batch_size and lr
    H = HparamsMBAE(dataset, dataset_type)
    H.vqgan_batch_size = H.batch_size  # used for generating samples and latents

    if parser_args.sampler == "trans":
        H_sampler = HparamsMBLD(dataset, dataset_type, sampler="trans")
    elif parser_args.sampler == "mdm":
        H_sampler = HparamsMBLD(dataset, dataset_type, sampler="mdm")
    else:
        raise NotImplementedError
    H.update(H_sampler)  # overwrites old (vqgan) H.batch_size
    H = apply_parser_values_to_H(H, parser_args)
    return H


# def set_up_sampler_parser(parser):
#     set_up_base_parser(parser)
#     add_mbae_args(parser)
#     add_sampler_args(parser)
#     return parser


def set_up_mbld_parser(parser):
    set_up_base_parser(parser)
    add_mbae_args(parser)
    add_mbld_args(parser)
    return parser


# def get_sampler_hparams():
#     parser = argparse.ArgumentParser("Parser for training discrete latent sampler models :)")
#     set_up_sampler_parser(parser)
#     H = get_sampler_H_from_parser(parser)
#     return H


def get_mbld_hparams():
    parser = argparse.ArgumentParser("Parser for trining motion binary latent diffusion models :)")
    set_up_mbld_parser(parser)
    H = get_mbld_H_from_parser(parser)
    return H


# def get_PRDC_hparams():
#     parser = argparse.ArgumentParser("Script for calculating PRDC on trained samplers")
#     add_PRDC_args(parser)
#     parser = set_up_sampler_parser(parser)
#     H = get_sampler_H_from_parser(parser)
#     return H


# def get_sampler_FID_hparams():
#     parser = argparse.ArgumentParser("Script for calculating FID on trained samplers")
#     add_sampler_FID_args(parser)
#     parser = set_up_sampler_parser(parser)
#     H = get_sampler_H_from_parser(parser)
#     return H


# def get_big_samples_hparams():
#     parser = argparse.ArgumentParser("Script for generating larger-than-training samples")
#     add_big_sample_args(parser)
#     parser = set_up_sampler_parser(parser)
#     H = get_sampler_H_from_parser(parser)
#     return H
