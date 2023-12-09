import logging
import numpy as np
import os
import torch
import torchvision

import torch.distributed as dist
import pdb

from visualize.tomato_representation.plot_3d_global import draw_to_batch_smplh


class MovingAverage(object):
    def __init__(self, length):
        self.length = length
        self.count = 0
        self.pointer = 0
        self.values = np.zeros(length)
        # self.avg = 0

    def update(self, val):
        self.values[self.pointer] = val
        self.pointer += 1
        if self.pointer == self.length:
            self.pointer = 0
        self.count += 1
        self.count = np.minimum(self.count, self.length)

    def avg(self):
        return self.values.sum() / float(self.count)

    def reset(self):
        self.count = 0
        self.pointer = 0
        # self.avg = 0
        self.values.fill(0)


def config_log(log_dir, filename="log.txt"):
    # log_dir = "logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def log(output):
    logging.info(output)
    print(output)


def log_stats(step, stats):
    log_str = f"Step: {step}  "
    for stat in stats:
        if "latent_ids" not in stat:
            if stat == 'lr':
                # log_str += f"{stat}: {stats[stat]:.8f}  "
                log_str += f"{stat}: {stats[stat]:.1E}  "
                # log_str += f"{stat}: "
                # log_str += format(stats[stat],'.1E')
                # log_str += " "
            elif stat == 'acc':
                log_str += f"{stat}: {int(stats[stat]*100):02d}  "
            elif stat == 'loss scale':
                log_str += f"{stat}: {int(stats[stat]):d}  "
            else:
                try:
                    log_str += f"{stat}: {stats[stat]:.4f}  "
                except TypeError:
                    log_str += f"{stat}: {stats[stat].mean().item():.4f}  "

    log(log_str)


def start_training_log(hparams):
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def save_model(model, model_save_name, step, log_dir):
    # log_dir = "logs/" + log_dir + "/saved_models"
    log_dir = os.path.join(log_dir, 'saved_models')
    print(f'log_dir: {log_dir}')
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{model_save_name}_{step}.th"
    log(f"Saving {model_save_name} to {model_save_name}_{str(step)}.th")
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir, strict=False, device=None, allow_mismatch=False):
    log(f"Loading {model_load_name}_{str(step)}.th")
    # log_dir = "logs/" + log_dir + "/saved_models"
    log_dir = os.path.join(log_dir, 'saved_models')
    try:
        state_dict = torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th"),  map_location=device)
        # pdb.set_trace()
        if allow_mismatch:
            # state_dict={k:v if v.size()==model.state_dict()[k].size() else  model.state_dict()[k] for k,v in zip(model.state_dict().keys(), state_dict.values())}
            for k,v in model.state_dict().items():
                if k not in state_dict.keys() or v.shape != state_dict[k].shape:
                    print(f'mismatch {k}')
                    state_dict[k] = v
        # pdb.set_trace()
        model.load_state_dict(
            state_dict,
            strict=strict,
        )
        del state_dict
    except TypeError:  # for some reason optimisers don't like the strict keyword
        state_dict = torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th"),  map_location=device)
        if allow_mismatch:
            state_dict={k:v if v.size()==model[k].size() else  model[k] for k,v in zip(model.keys(), state_dict.values())}
        model.load_state_dict(
            state_dict,
        )
        del state_dict

    return model


def display_images(vis, images, H, win_name=None):
    if win_name is None:
        win_name = f"{H.model}_images"
    images = torchvision.utils.make_grid(images.clamp(0, 1), nrow=int(np.sqrt(images.size(0))), padding=0)
    vis.image(images, win=win_name, opts=dict(title=win_name))


def save_images(images, im_name, step, log_dir, save_individually=False, name='images'):
    # log_dir = "logs/" + log_dir + "/images"
    log_dir = os.path.join(log_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    if save_individually:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/{im_name}_{step}_{idx}.png")
    else:
        torchvision.utils.save_image(
            torch.clamp(images, 0, 1),
            f"{log_dir}/{im_name}_{step:09}.jpg",
            # nrow=int(np.sqrt(images.shape[0])),
            nrow=10,
            padding=0
        )


def save_motion(dataset_type, motion, mot_name, step, log_dir, lengths, j2nj=None, s2j=None, name='motion', desc=None, device=None):
    log_dir_motion = os.path.join(log_dir, name)
    os.makedirs(log_dir_motion, exist_ok=True)

    # body,hand joint idx
    # 2*3*5=30, left first, then right
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints

    t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [
        0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [
        20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
    t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [
        21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
    t2m_body_hand_kinematic_chain = t2m_kinematic_chain + t2m_left_hand_chain + t2m_right_hand_chain

    if dataset_type == 'newjoints':
        if motion.shape[1] != 52:
            # print(f'motion.shape: {motion.shape}')
            motion = motion[:, body_joints_id+hand_joints_id, :]
            # print(f'motion.shape: {motion.shape}')
        # print(f'motion.shape: {motion.shape}')
        xyz = motion.reshape(1, -1, 52, 3)
        # print(f'xyz.shape: {xyz.shape}')
        xyzs = []
        for i in range(len(lengths)):
            # print('lengths[i]: ', lengths[i])
            xyzs.append(xyz[:, :int(lengths[i].item()), :, :])
            xyz = xyz[:, int(lengths[i].item()):, :, :]
            # print(f'xyzs[i].shape: {xyzs[i].shape}')
        pose_vis = draw_to_batch_smplh(xyzs[0], t2m_body_hand_kinematic_chain, title_batch=desc, outname=mot_name, log_dir=log_dir_motion, step=step)

    elif dataset_type == 'smplx':
        assert j2nj is not None, 'j2nj is None'
        assert s2j is not None, 's2j is None'

        log_dir_smplx = os.path.join(log_dir, 'smplx_322')
        os.makedirs(log_dir_smplx, exist_ok=True)
    
        # print(f'motion.shape: {motion.shape}')
    
        # Get the first smplx motion
        print("Saving smplx motion...")
        smplx = motion[:int(lengths[0].item()), :, :].squeeze(-1)
        # print(f'smplx type: {type(smplx)}')
        # print(f'smplx.shape: {smplx.shape}')
        # Save the first smplx motion as .npy file
        save_path = os.path.join(log_dir_smplx, f'{mot_name[0]}_{step:09}.npy')
        np.save(save_path, smplx)
        # Preprocess smplx motion
        # print(f'smplx.shape: {smplx.shape}')
        smplx = torch.from_numpy(smplx).unsqueeze(0).to(device)
        print("Converting smplx motion to joint representation...")
        joint = s2j.convert(smplx, save_path)
        # print(f'joint.shape: {joint.shape}')
        # Convert joint to newjoint's format
        print("Converting joint representation to newjoint representation...")
        newjoint = j2nj.convert(save_path, joint, d_type='smplx_322')
        # print(f'newjoint.shape: {newjoint.shape}')
        # Create 3D motion representation
        print("Creating 3D motion representation...")
        pose_vis = draw_to_batch_smplh([newjoint], t2m_body_hand_kinematic_chain, title_batch=desc, outname=mot_name, log_dir=log_dir_motion, step=step)

        
def save_results(images, im_name, step, log_dir, temp, save_individually=False):
    log_dir = os.path.join(log_dir, f'results')
    os.makedirs(log_dir, exist_ok=True)
    if save_individually:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/{im_name}_{temp}_{step}_{idx}.jpg")
    else:
        torchvision.utils.save_image(
            torch.clamp(images, 0, 1),
            f"{log_dir}/{im_name}_{temp}_{step}.jpg",
            nrow=10,
            padding=2
        )


def save_results_all(images, im_name, step, log_dir, temp, save_individually=False):
    # log_dir = "logs/" + log_dir + "/images"
    log_dir = os.path.join(log_dir, f'allsteps_{temp}')
    os.makedirs(log_dir, exist_ok=True)
    if save_individually:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/{im_name}_r{dist.get_rank()}_{step}_{idx}.jpg")


def save_results_t2i(images, cls_idx, log_dir, temp, num_steps, name='', text=None, save_individually=False):
    # log_dir = "logs/" + log_dir + "/images"
    log_dir = os.path.join(log_dir, f'results_t2i_s{num_steps}_{temp}_{name}')
    os.makedirs(log_dir, exist_ok=True)
    if save_individually:
        log_dir = os.path.join(log_dir, f'results_t2i_s{num_steps}_{temp}_{name}/{cls_idx:04d}')
        os.makedirs(log_dir, exist_ok=True)
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/img_r{dist.get_rank()}_{idx}.jpg")
    else:
        torchvision.utils.save_image(
            torch.clamp(images, 0, 1),
            f"{log_dir}/{text}.jpg",
            nrow=5,
            padding=0
        )


def save_latents(H, train_latent_ids, val_latent_ids, save_dir="latents"):

    os.makedirs(f'{H.ae_load_dir}/latents/', exist_ok=True)

    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""
    # train_latents_fp = f"{save_dir}/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}"
    # val_latents_fp = f"{save_dir}/{H.dataset}_{H.latent_shape[-1]}_val_latents{latents_fp_suffix}"
    train_latents_fp = f'{H.ae_load_dir}/latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}_{H.ae_load_step}'
    val_latents_fp = f'{H.ae_load_dir}/latents/{H.dataset}_{H.latent_shape[-1]}_val_latents{latents_fp_suffix}_{H.ae_load_step}'

    torch.save(train_latent_ids, train_latents_fp)
    torch.save(val_latent_ids, val_latents_fp)


def save_stats(H, stats, step):
    save_dir = f"{H.log_dir}/saved_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{H.log_dir}/saved_stats/stats_{step}"
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)


def load_stats(H, step):
    load_path = f"{H.load_dir}/saved_stats/stats_{step}"
    stats = torch.load(load_path)
    return stats


# def set_up_visdom(H):
#     server = H.visdom_server
#     try:
#         if server:
#             vis = visdom.Visdom(server=server, port=H.visdom_port)
#         else:
#             vis = visdom.Visdom(port=H.visdom_port)
#         return vis

#     except Exception:
#         log_str = "Failed to set up visdom server - aborting"
#         log(log_str, level="error")
#         raise RuntimeError(log_str)