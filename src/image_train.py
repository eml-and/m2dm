"""
Train a diffusion model on images.
"""

import argparse
import cProfile
import json
import pstats
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import torch as th

from improved_diff import logger
from improved_diff.image_datasets import load_data
from improved_diff.resample import create_named_schedule_sampler
from improved_diff.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    create_mu2model_and_diffusion,
    model_and_diffusion_defaults,
)
from improved_diff.train_util import TrainLoop


def load_args_from_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        args = json.load(json_file)
    return args


def create_argparser():
    defaults = dict(
        data_dir="datasets/cifar_train",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_kl=True,
        learn_sigma=True,
        class_cond=True,
        num_channels=128,
        noise_schedule="cosine",
        num_res_block=3,
        image_size=32,
        diffusion_steps=10,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = "cuda"
    # Namespace
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)  # n_params: 27.223.686
    mu2model, diffusion = create_mu2model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # input_tensor = th.randn(2, 3, 36, 36).to(device)
    # ts, y = th.arange(2).to(device), th.tensor([0, 1]).to(device)
    mu2model.to(device)  # n_params: 25.874.310
    # this breaks down at conv1d
    # RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [2, 1024, 25, 25]
    # mu2model.get_feature_vectors(x=input_tensor,

    #                              timesteps=ts,
    #                              y=y)
    # # TODO: Remove
    # flop_count = FlopCountAnalysis(model, input_tensor)
    # flops = flop_count.total()
    # flop_m2 = FlopCountAnalysis(mu2model, input_tensor)
    # flops_m2 = flop_m2.total()
    # print(f"Flops model: {flops}, flops m2: {flops_m2}")
    # breakpoint()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    print(
        f"Number of params: U {count_parameters(model)}, M2 {count_parameters(mu2model)}"
    )

    logger.log("training...")
    TrainLoop(
        model=mu2model,
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
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def debug_main():
    device = "cuda"
    # Namespace
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    x = th.randn(4, 3, 64, 64).to(device)
    timesteps = th.tensor([2000]).to(device)
    # dict containing down, middle and up feature vectors
    result = model.get_feature_vectors(x, timesteps)
    print(result)


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profiling_info.prof")

    # p = pstats.Stats("profiling_data.prof")
    p.sort_stats("cumulative").print_stats(10)
    # debug_main()
