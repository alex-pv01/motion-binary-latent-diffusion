import argparse, os, sys, datetime, glob
import numpy as np
import torch
import torchvision
import random
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

import wandb

from utils.utils import instantiate_from_config, get_parser


def main():
    """
    Main function for training binary diffusion model.
    """
    # Save the current time for logging
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print("----------------------------------------")
    print("----------------------------------------")
    print("Starting training at {}".format(now))
    print()

    # Add cwd to path
    sys.path.append(os.getcwd())

    # Parse arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # Check name and resume contradictions
    if opt.name and opt.resume:
        raise ValueError("Cannot specify both name and resume")

    # Check if we are resuming
    if opt.resume:
        # Check if checkpoint path extist
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find checkpoint path {}".format(opt.resume))
        # Get logdir and checkpoint path
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoint", "last.ckpt")
        print("Resuming from checkpoint {}".format(ckpt))
        print()
        opt.resume_from_checkpoint = ckpt
        # Get config path
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # Get name for resumed experiment
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        # Set name for new experiment
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_name = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_name)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        # Set logdir
        logdir = os.path.join("logs", nowname)
        
    print("Logdir: {}".format(logdir))
    print()

    # Get checkpoint and config dirs
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # Set seed
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
        if opt.gpu is not None:
            torch.cuda.manual_seed(opt.seed)
            torch.cuda.manual_seed_all(opt.seed)
            torch.backends.cudnn.deterministic = True

    
    # Create logdirs and save configs
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    # Start main method
    try:
        # Initialize configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        print("Poject config")
        print(config)
        print()
        OmegaConf.save(config, os.path.join(cfgdir, "{}-project.yaml".format(now)))


        # Initialize model
        print("Initializing model..")
        print("Model config: {}".format(config.model))
        print()
        model = instantiate_from_config(config.model)
        if config.model.cond_stage_config == '__is_unconditional__':
            cond = False
        else:
            cond = True

        # Initialize data
        print("Initializing data..")
        data = instantiate_from_config(config.data)
        # data.prepare_data()
        # data.setup()
        
        train_loader = data.train_dataloader()
        print("Train loader: {}".format(train_loader))
        val_loader = data.val_dataloader()
        print("Val loader: {}".format(val_loader))
        print()

        # Set optimizer and scheduler
        print("Initializing optimizer..")
        if model.use_scheduler:
            optimizers, schedulers = model.configure_optimizers()
            print("Optimizers: {}".format(optimizers))
            print("Schedulers: {}".format(schedulers))
        else: 
            optimizers = model.configure_optimizers()
            print("Optimizers: {}".format(optimizers))
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if model.use_scheduler:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
        print()

        # Set logger
        print("Initializing wandb logger..")
        print("Project: {}".format(opt.project))
        print("Name: {}".format(nowname))
        print("Config: {}".format(config))

        run = wandb.init(project=opt.project, name=nowname)

        # Log config as artifact
        artifact = wandb.Artifact("config", type="config")
        run.log_artifact(artifact)

        print("Wandb logger initialized.")
        print()

        # Set callbacks
        # default_callbacks_cfg = {
        #     #TODO: add smplx2gif callbacks
        #     #TODO: add joint2gif callbacks
        # }
        # callbacks_cfg = OmegaConf.create()
        # callbacks_cfg = OmegaConf.merge(callbacks_cfg, default_callbacks_cfg)
        # callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # Set device
        if opt.gpu is not None:
            print("Using GPU: {}".format(opt.gpu))
            device = torch.device("cuda:{}".format(opt.gpu))
        else:
            print("Using CPU")
            device = torch.device("cpu")

        print("Moving model to device..")
        model.to(device)
        print()

    except:
        raise ValueError("Error while initializing model.")


    finally:
        pass


    # Start training
    print("----------------------------------------")
    print("Starting training..")
    num_epochs = config.trainer.num_epochs
    log_every_n_steps = config.trainer.log_every_n_steps
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Epoch {}/{}".format(epoch+1, num_epochs), total=len(train_loader)):
            # Forward pass
            batch_loss = model.training_step(batch, batch_idx, device=device)

            # Backward pass
            batch_loss.backward()

            # Update weights
            for optimizer in optimizers:
                optimizer.step()

            # Update scheduler
            if model.use_scheduler:
                for scheduler in schedulers:
                    scheduler.step()

            # Log losses to wandb
            run.log({"train/loss": batch_loss})
            #print("Batch loss: {}".format(batch_loss))
            
            # Log images and save checkpoints
            if batch_idx % log_every_n_steps == 0:
                # Save checkpoint
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                torch.save(model.state_dict(), ckpt_path)
                # Save config
                cfg_path = os.path.join(cfgdir, "last.yaml")
                OmegaConf.save(config, cfg_path)
                # Save best checkpoint
                # if batch_loss < best_loss:
                #     best_loss = batch_loss
                #     ckpt_path = os.path.join(ckptdir, "best.ckpt")
                #     torch.save(model.state_dict(), ckpt_path)
                #     cfg_path = os.path.join(cfgdir, "best.yaml")
                #     OmegaConf.save(config, cfg_path)
                    #print("Saved best checkpoint..")
                # Call callbacks
                # for callback in callbacks:
                #     callback.on_train_batch_end(batch_idx, batch)
                
        # Validation
        for batch_idx, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader)):
            # Forward pass
            batch_loss = model.validation_step(batch, batch_idx, device)

            # Log
            if batch_idx % log_every_n_steps == 0:
                run.log({"val_loss": batch_loss})
                #print("Batch loss: {}".format(batch_loss))
                # Save checkpoint
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                torch.save(model.state_dict(), ckpt_path)
                # Save config
                cfg_path = os.path.join(cfgdir, "last.yaml")
                OmegaConf.save(config, cfg_path)
                # Save best checkpoint
                if batch_loss < best_loss:
                    best_loss = batch_loss
                    ckpt_path = os.path.join(ckptdir, "best.ckpt")
                    torch.save(model.state_dict(), ckpt_path)
                    cfg_path = os.path.join(cfgdir, "best.yaml")
                    OmegaConf.save(config, cfg_path)
                    print("Saved best checkpoint..")
                # Call callbacks
                # for callback in callbacks:
                #     callback.on_val_batch_end(batch_idx, batch)


if __name__ == "__main__":
    sys.exit(main())