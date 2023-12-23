# file for running the training of the binaryae
import torch
import numpy as np
import copy
import random
import datetime
import time
import os
os.environ["OMP_NUM_THREADS"] = "4"

from tqdm import tqdm

from models.encoding.mbae import MotionBinaryAutoEncoder
from models.encoding.utils import load_mbinaryae_from_checkpoint
from models.log_utils import log, log_stats, save_model, save_stats, config_log, start_training_log, save_motion
from data_loaders.get_data import DataModule
from train.utils import EMA

from hparams import get_mbae_hparams

import wandb

def main(args, project, namenow):
    """
    Main function for training binary diffusion model.
    """
    # Set wand logger
    print("Initializing wandb logger..")
    print("Project: {}".format(project))
    print("Name: {}".format(namenow))

    run = wandb.init(project=project, name=namenow)

    print("Wandb logger initialized.")
    print()

    # Log the hyperparameters as config
    wandb.config.update(args)

    # Set cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set up the device
    if args.gpu is not None:
        print("Using GPU: {}".format(args.gpu))
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        print("Using CPU")
        device = torch.device("cpu")

    device = torch.device("cuda:0")
    print("Device: {}".format(device))

    # Initialize the model at device
    mbinaryae = MotionBinaryAutoEncoder(args).cuda(device)

    # Initialize the data module
    data_module = DataModule(batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             motions_path=args.motions_path,
                             texts_path=args.texts_path,
                             name=args.dataset,
                             train=args.train,
                             val=args.val,
                             test=args.test,
                             debug=args.debug,
                             )

    # Get the data loaders
    if args.train:
        train_loader = data_module.train_dataloader()
    if args.val:
        val_loader = data_module.val_dataloader()
    if args.test:
        test_loader = data_module.test_dataloader()

    if args.ema:
        ema = EMA(args.ema_beta)
        ema_mbinaryae = copy.deepcopy(mbinaryae)
    else:
        ema_mbinaryae = None
    
    # Initialize the optimizer
    optim = torch.optim.Adam(mbinaryae.parameters(), lr=args.lr)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # Set up logging
    losses = np.array([])
    l1_losses = np.array([])
    mse_losses = np.array([])
    val_losses = np.array([])
    latent_ids = []
    fids = np.array([])
    best_fid = float('inf')

    start_step = 0
    log_start_step = 0
    eval_start_step = args.steps_per_eval

    # Load from checkpoint if specified
    if args.load_step > 0:
        start_step = args.load_step + 1  # don't repeat the checkpointed step
        mbinaryae, optim, ema_mbinaryae, train_stats = load_mbinaryae_from_checkpoint(args, mbinaryae, optim, ema_mbinaryae)

        # stats won't load for old models with no associated stats file
        if train_stats is not None:
            losses = train_stats["losses"]
            l1_losses = train_stats["l1_losses"]
            mse_losses = train_stats["mse_losses"]
            val_losses = train_stats["val_losses"]
            # latent_ids = train_stats["latent_ids"]
            fids = train_stats["fids"]
            best_fid = train_stats["best_fid"]
            args.steps_per_log = train_stats["steps_per_log"]
            args.steps_per_eval = train_stats["steps_per_eval"]

            log_start_step = 0
            eval_start_step = args.steps_per_eval
            log('Loaded stats')
        else:
            log_start_step = start_step
            if args.steps_per_eval:
                if args.steps_per_eval == 1:
                    eval_start_step = start_step
                else:
                    eval_start_step = start_step + args.steps_per_eval - start_step % args.steps_per_eval


    log(f"Number of model parameters: {(sum(p.numel() for p in mbinaryae.parameters())/1e6)}M")

    step = start_step - 1
    num_epochs = 2000
    return_input = True

    if args.dataset_type in ['smplx', 'joints']:
        from visualize.tomato_representation.motion_representation import J2NJConverter
        # Required to convert to set the offset of the skeleton
        example_data_path = '/home/apujol/mbld/datasets/MotionX/datasets/motion_data/joint/humanml/000010.npy'
        j2nj_converter = J2NJConverter(example_data_path=example_data_path)
    else:
        j2nj_converter = None

    if args.dataset_type == 'smplx':
        from visualize.tomato_representation.raw_pose_processing import S2JConverter
        s2j_converter = S2JConverter(device)
    else:
        s2j_converter = None

    print("Starting training loop..")
    # Train the model
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Epoch {}/{}".format(epoch+1, num_epochs), total=len(train_loader)):
            step += 1
            step_start_time = time.time()

            motion, cond = batch
            lengths = cond['y']['lengths']

            # print('motion shape: ', motion.shape)

            if args.amp:
                optim.zero_grad()
                with torch.cuda.amp.autocast():
                    x_hat, stats, x_input = mbinaryae.training_step(motion, device, return_input=return_input, lengths=lengths)
                scaler.scale(stats['loss']).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x_hat, stats, x_input = mbinaryae.training_step(motion, device, return_input=return_input, lengths=lengths)
                optim.zero_grad()
                stats['loss'].backward()
                optim.step()

            torch.cuda.synchronize()

            if step == 1500000:
                for param_group in optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print('lr decay')
            
            # collect latent ids
            # log codebook usage

            code = stats['latent_ids'].cpu().contiguous()
            # code = code.to(torch.bool)
            code = code > 0.5
            latent_ids.append(code)
            # pdb.set_trace()
            if step % 200 == 0 and step > 0:
                # pdb.set_trace()
                # print('shape: ', latent_ids[0].shape)
                latent_ids = torch.cat(latent_ids, dim=0).permute(1,0,2).reshape(args.codebook_size, -1)
                codesample_size = latent_ids.shape[1]
                latent_ids = latent_ids * 1.0

                latent_ids = latent_ids.sum(-1)

                odd_idx = ((latent_ids == 0) * 1.0).sum() + ((latent_ids == codesample_size) * 1.0).sum()
                
                if int(args.codebook_size - odd_idx) != args.codebook_size:
                    log(f'Codebook size: {args.codebook_size}   Unique Codes Used in Epoch: {args.codebook_size - odd_idx}')
                latent_ids = []
                
            if step % args.steps_per_log == 0:
                losses = np.append(losses, stats['loss'].item())
                step_time = time.time() - step_start_time
                stats['step_time'] = step_time
                mse_losses = np.append(mse_losses, stats['mse_loss'])
                l1_losses = np.append(l1_losses, stats['l1_loss'])
                losses = np.array([])

                log_stats(step, stats)

                run.log(stats)

            if args.ema and step % args.steps_per_update_ema == 0 and step > 0:
                ema.update_model_average(ema_mbinaryae, mbinaryae)

            if step % args.steps_per_save_output == 0:
                print()
                print("Saving output for step {}".format(step))  
                if args.ema:
                    x_hat, _, x_input = ema_mbinaryae.training_step(motion, device, return_input=return_input, lengths=lengths)
                save_motion(dataset_type=args.dataset_type, 
                            motion=x_hat.permute(0,2,1).detach().cpu().numpy(), 
                            lengths=lengths, 
                            mot_name=cond['y']['name'], 
                            desc=cond['y']['text'],  
                            step=step, 
                            j2nj=j2nj_converter,
                            s2j=s2j_converter,
                            log_dir=args.log_dir,
                            device=device)         
                if return_input:
                    print("Saving input for step {}".format(step))
                    save_motion(dataset_type=args.dataset_type, 
                                motion=x_input.permute(0,2,1).detach().cpu().numpy(), 
                                name='input', 
                                lengths=lengths,
                                mot_name=cond['y']['name'], 
                                desc=cond['y']['text'],  
                                step=step, 
                                log_dir=args.log_dir,
                                j2nj=j2nj_converter,
                                s2j=s2j_converter,
                                device=device)
        
            if step % args.steps_per_checkpoint == 0 and step > args.load_step:

                save_model(mbinaryae, 'mbinaryae', step, args.log_dir)
                save_model(optim, 'optim', step, args.log_dir)
                if args.ema:
                    save_model(ema_mbinaryae, 'mbinaryae_ema', step, args.log_dir)

                train_stats = {
                    'losses': losses,
                    'l1_losses': l1_losses,
                    'mse_losses': mse_losses,
                    'val_losses': val_losses,
                    'fids': fids,
                    'best_fid': best_fid,
                    'steps_per_log': args.steps_per_log,
                    'steps_per_eval': args.steps_per_eval,
                }
                save_stats(args, train_stats, step)
            
            if step == args.train_steps:
                exit()

if __name__ == '__main__':
    args = get_mbae_hparams()

    # Set project name
    project = "training-motion-bvae"

    # Save the current time for logging
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Set the name of current run
    name = "mbvae" + "_" + args.dataset
    namenow = name + "_" + now

    print("----------------------------------------")
    print("----------------------------------------")
    print("Starting training at {}".format(now))
    print()

    args.log_dir = os.path.join(args.log_dir, namenow)

    config_log(args.log_dir)
    log('---------------------------------')
    log(f'Setting up training for mbinaryae on {args.dataset}')
    start_training_log(args)
    main(args, project, namenow)