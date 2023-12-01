import torch
import numpy as np
import copy
import time
import os
import pdb
import datetime
from tqdm import tqdm

from models.encoding.mbae import MotionBinaryAutoEncoder
from models.encoding.modules.encoder import Generator
from models.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_motion, \
    MovingAverage
from data_loaders.get_data import DataModule
from train.utils import EMA, NativeScalerWithGradNormCount
from train.lr_sched import adjust_lr, lr_scheduler

from models.diffusion_modules.utils import retrieve_mbae_components_state_dicts,\
    get_mbld, get_online_motions, get_online_motions_guidance

from hparams import get_mbld_hparams

import wandb 

def main(H, project, namenow):
    # Set project name
    project = 'training-bdiff'

    # Save the current time for logging
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Set the name of the current run
    name ='bdiff'
    namenow = name + '_' + now

    print("----------------------------------------")
    print("----------------------------------------")
    print("Starting training at {}".format(now))
    print()

    # Set wand logger
    print("Initializing wandb logger..")
    print("Project: {}".format(project))
    print("Name: {}".format(namenow))

    run = wandb.init(project=project, name=namenow)

    # Retrieve the autoencoder components
    mbae_state_dict = retrieve_mbae_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    # Log ae_state_dict as config
    wandb.config.update(mbae_state_dict)

    # Set up the device
    if H.gpu is not None:
        print("Using GPU: {}".format(H.gpu))
        device = torch.device("cuda:{}".format(H.gpu))
    else:
        print("Using CPU")
        device = torch.device("cpu")
    print("Device: {}".format(device))

    # Load the autoencoder
    mbinaryae = MotionBinaryAutoEncoder(H)
    mbinaryae.load_state_dict(mbae_state_dict, strict=True)
    mbinaryae = mbinaryae.cuda(device)
    del mbae_state_dict

    # Initialize the sampler
    sampler = get_mbld(H, mbinaryae.quantize.embed.weight).cuda(device)

    # Initialize the data module
    data_module = DataModule(batch_size=H.batch_size,
                             num_workers=H.num_workers,
                             motions_path=H.motions_path,
                             texts_path=H.texts_path,
                             name=H.dataset,
                             train=H.train,
                             val=H.val,
                             test=H.test,
                             debug=H.debug,
                             )

    # Get the data loaders
    if H.train:
        train_loader = data_module.train_dataloader()
    if H.val:
        val_loader = data_module.val_dataloader()
    if H.test:
        test_loader = data_module.test_dataloader()

    # Initialize the EMA
    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # Initialize the optimizer
    optim_eps = H.optim_eps
    optim = torch.optim.AdamW(sampler.parameters(), lr=H.lr, weight_decay=H.weight_decay, betas=(0.9, 0.95), eps=optim_eps)

    # Set up logging
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0

    loss_ma = MovingAverage(100)

    # Load the model
    if H.load_model_step > 0:
        device = sampler.device
        sampler = load_model(sampler, H.sampler, H.load_model_step, H.load_model_dir, device=device).cuda()

    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)

    # Load from checkpoint if specified
    if H.load_step > 0:
        start_step = H.load_step + 1

        device = sampler.device

        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
        
        if not allow_mismatch:
            if H.load_optim:
                optim = load_model(
                    optim, f'{H.sampler}_optim', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr
        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if not H.reset_step:
            if not H.reset_scaler:
                try:
                    scaler.load_state_dict(torch.load(os.path.join(H.load_dir, 'saved_models', f'absorbingbnl_scaler_{H.load_step}.th')))
                except Exception:
                    print('Failing to load scaler.')
        else:
            H.load_step = 0

        
        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = train_stats["losses"],
            mean_losses = train_stats["mean_losses"],
            val_losses = train_stats["val_losses"],
            val_elbos = train_stats["val_elbos"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            val_elbos = torch.Tensor([0])

        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

        if H.reset_step:
            start_step = 0

    log(f"Sampler params total: {(sum(p.numel() for p in sampler.parameters())/1e6)}M")
    
    # Initialize the lr scheduler and warmup
    H.train_steps = H.train_steps * H.update_freq
    H.warmup_iters = H.warmup_iters * H.update_freq
    H.steps_per_log = H.steps_per_log * H.update_freq
    lr_sched = lr_scheduler(base_value=H.lr, final_value=1e-6, iters=H.train_steps+1, warmup_steps=H.warmup_iters,
                     start_warmup_value=1e-6, lr_type='constant')
    
    step = start_step - 1
    epoch = -1
    num_epochs = 2000

    # Train the sampler
    optim.zero_grad()
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        #train_loader.sampler.set_epoch(epoch)
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Epoch {}/{}".format(epoch+1, num_epochs), total=len(train_loader)):
            step += 1
            adjust_lr(optim, lr_sched, step)
            step_start_time = time.time()

            motions = batch['motion'].cuda()
            # print("motions: ", motions.shape)
            # print("motions: ", motions)
            texts = batch['texts'].cuda()
            # print("texts: ", texts.shape)
            # print("texts: ", texts)

            with torch.no_grad():
                code = mbinaryae(motions, code_only=True).detach()
                # print("code: ", code.shape)
                # print("code: ", code)
                b,c,w = code.shape
                x = code.permute(0,2,1).contiguous()
                # print("x: ", x.shape)
                # print("x: ", x)

            with torch.cuda.amp.autocast(enabled=H.amp):
                if H.conditioned = True:
                    # print("Text conditioned")
                    stats = sampler(x, texts)
                    # print("stats: ", stats)
                else:
                    # print("No text conditioning")
                    stats = sampler(x)
                    # print("stats: ", stats)
                loss = stats['loss']
                loss = loss / H.update_freq

                run.log({"train/loss": loss.item()})

            if step == 0:
                print()
                print("Saving output for step {}".format(step))  
                motions = get_online_motions(H, mbinaryae, ema_sampler if H.ema else sampler, x=x)
                # print("images: ", images.shape)
                # print("images: ", images)
                save_motion(motion=motions.detach().cpu().numpy(), mot_name=batch['name'], desc=batch['text'],  step=step, log_dir=args.log_dir)
                # save to test the reconstruction quality

            grad_norm = scaler(loss, optim, clip_grad=H.grad_norm,
                                parameters=sampler.parameters(), create_graph=False,
                                update_grad=(step + 1) % H.update_freq == 0)

            if (step + 1) % H.update_freq == 0:
                optim.zero_grad()
            loss_ma.update(loss.item())
            if H.ema and step % (H.steps_per_update_ema * H.update_freq) == 0 and step > 0:
                ema.update_model_average(ema_sampler, sampler)

            torch.cuda.synchronize()

            if step % H.steps_per_log == 0:

                stats['lr'] = optim.param_groups[0]['lr']
                step_time_taken = time.time() - step_start_time
                stats['step_time'] = step_time_taken
                mean_loss = np.mean(losses)
                stats['mean_loss'] = loss_ma.avg()

                if "scale" in scaler.state_dict().keys():
                    stats['loss scale'] = scaler.state_dict()["scale"]
                mean_losses = np.append(mean_losses, mean_loss)
                losses = np.array([])

                log_stats(step, stats)

                # Log stats dictionary to wandb
                run.log(stats)


            if step % H.steps_per_save_output == 0:
                if H.guidance:
                    # print("guidance")
                    motions = get_online_motions_guidance(H, mbinaryae, ema_sampler if H.ema else sampler)
                else:
                    # print("no guidance")
                    motions = get_online_motions(H, mbinaryae, ema_sampler if H.ema else sampler)
                print()
                print("Saving output for step {}".format(step))
                # print("motions: ", motions.shape)
                # print("motions: ", motions)
                save_motion(motion=motions.detach().cpu().numpy(), mot_name=batch['name'], desc=batch['text'],  step=step, log_dir=args.log_dir)


            if step % H.steps_per_checkpoint == 0 and step > H.load_step:
                save_model(sampler, H.sampler, step, H.log_dir)
                save_model(optim, f'{H.sampler}_optim', step, H.log_dir)
                save_model(scaler, f'{H.sampler}_scaler', step, H.log_dir)

                if H.ema:
                    save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

                train_stats = {
                    'losses': losses,
                    'mean_losses': mean_losses,
                    'val_losses': val_losses,
                    'elbo': elbo,
                    'val_elbos': val_elbos,
                    'steps_per_log': H.steps_per_log,
                    'steps_per_eval': H.steps_per_eval,
                }
                save_stats(H, train_stats, step)
            
            if step == H.train_steps:
                exit()


if __name__ == '__main__':
    H = get_mbld_hparams()

    
    # Set project name
    project = "training-motion-bld"

    # Save the current time for logging
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Set the name of current run
    name = "mbld"
    namenow = name + "_" + now

    print("----------------------------------------")
    print("----------------------------------------")
    print("Starting training at {}".format(now))
    print()

    H.log_dir = os.path.join(H.log_dir, namenow)

    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, project, namenow)
