import os
import pdb
import numpy as np 
#import misc
import time
import torch
from tqdm import tqdm
#from pyrsistent import l

from models.log_utils import save_latents, log
from models.diffusion_modules.binary_diffusion import BinaryDiffusion
from models.diffusion_modules.transformer import TransformerBD
from models.diffusion_modules.mdm import MDM


def get_mbld(H, embedding_weight):

    if H.sampler == 'mdm':
        denoise_fn = MDM(H).cuda()
        print(denoise_fn)
        sampler = BinaryDiffusion(H, denoise_fn, H.codebook_size, embedding_weight)
    elif H.sampler == 'trans':
        if H.gpu is not None:
            denoise_fn = TransformerBD(H).cuda()
        else:
            denoise_fn = TransformerBD(H)
        print(denoise_fn)
        sampler = BinaryDiffusion(H, denoise_fn, H.codebook_size, embedding_weight)
    else:
        raise NotImplementedError

    return sampler


@torch.no_grad()
def get_samples_temp(H, generator, sampler, x=None, ee=False):

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        t0 = time.time()
        if ee:
            for t in np.linspace(0.2, 1.0, num=5):
                for f in [False, True]:
                    latents = sampler.sample(sample_steps=H.sample_steps, temp=t, full=f)
                    latents = latents[:10]
                    latents_all.append(latents)
        else:
            for t in np.linspace(0.2, 1.0, num=10):
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, full=f)
                latents = latents[:10]
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()

        print('Sampling done at %.1fs' %((time.time() - t0)))
    else:
        latents = x

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_samples_test(H, generator, sampler, x=None, t=1.0, n_samples=20, return_all=False, label=None, mask=None, guidance=None):
    generator.eval()
    sampler.eval()
    latents = sampler.sample(sample_steps=H.sample_steps, temp=t, b=n_samples, return_all=return_all, label=label, mask=mask, guidance=guidance)
    

    if mask is not None:
        latents = torch.cat([mask['latent'].unsqueeze(0), latents], 0)

    with torch.cuda.amp.autocast():
        size = min(25, latents.shape[0])
        if H.latent_shape[-1] == 32:
            size = 5
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            # latents = latents / (latents.sum(dim=-1, keepdim=True)+1e-6)
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_samples_guidance(H, generator, sampler, x=None):

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        for g in [None, 0.1, 0.5, 1.0, 2.0, 5.0]:
            for t in [0.5, 0.9]:
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, guidance=g)
                
                latents = latents[:10]
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()

    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            # latents = latents / (latents.sum(dim=-1, keepdim=True)+1e-6)
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_t2i_samples_guidance(H, generator, sampler, label, x=None,):
    
    if isinstance(label, list):
        batch_size = label[0].shape[0]
    else:
        batch_size = label.shape[0]

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        for g in [None, 0.1, 0.5, 1.0, 3.0, 10.0]:
            for t in [0.6, 1.0]:
                latents = sampler.sample(sample_steps=H.sample_steps, b=batch_size, temp=t, label=label, guidance=g)
                
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()
    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_t2i_samples_guidance_test(H, generator, sampler, label, x=None, g=None, t=1.0, return_latent=False):

    if isinstance(label, list):
        batch_size = label[0].shape[0]
    else:
        batch_size = label.shape[0]

    if x is None:
        latents_all = []
        sampler.eval()
        print('Sampling')
        t0 = time.time()
        latents = sampler.sample(sample_steps=H.sample_steps, b=batch_size, temp=t, label=label, guidance=g)
        
        latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)

        print('Sampling done at %.1fs' %((time.time() - t0)))
        sampler.train()
    else:
        latents = x


    with torch.cuda.amp.autocast():
        # latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    if return_latent:
        return images, latents
    else:
        return images

@torch.no_grad()
def get_online_motions(H, generator, sampler, x=None, lengths=None, label=None):
    # print('GET ONLINE MOTIONS')

    if x is None:
        # latents_all = []
        sampler.eval()
        # print('SAMPLING')
        # for t in np.linspace(0.55, 1.0, num=2):
        t = np.random.uniform(0.55, 1.0)
        if H.conditioned:
            assert label is not None, 'label must be provided for conditioned sampling'
            if H.guidance:
                g = 1.0
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, label=label, guidance=g)
            else:
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, label=label)
        else:
            latents = sampler.sample(sample_steps=H.sample_steps, temp=t)
            # print('latents', latents.shape)
        #     latents_all.append(latents)
        # latents = torch.cat(latents_all, dim=1)
        # print('latents', latents.shape) # if mdm torch.Size([1, 13, 32, 200]) if trans torch.Size([1, 2600, 32])
        
        lengths = label['lengths']

        # latents = generator.get_input(latents, lengths=[latents.shape[-1]])
        sampler.train()
        # print('Sampling done')

    else:
        # print('LATENTS GIVEN')
        latents = x
    
    print('latents', latents.shape) # if mdm torch.Size([bs, 13, 32, 200]) if trans torch.Size([bs, 2600, 32])

    with torch.cuda.amp.autocast():
        if H.sampler_type == 'trans':
            # print("Permute to fit generator input")
            # raise NotImplementedError
            latents = latents.permute(0,2,1)
            # print("permuted latents", latents.shape) # torch.Size([bs, 32, 2600])
            latents = latents.reshape(*latents.shape[:-1], H.latent_shape[2], H.max_length)
            # print("reshaped latents", latents.shape) # torch.Size([bs, 32, 13, 200])
        elif H.sampler_type == 'mdm':
            latents = latents.permute(0,2,1,3)
            # print('permuted latents', latents.shape) # torch.Size([bs, 32, 13, 200])
        # print('get input')
        latents = generator.get_input(latents, lengths)
        # print('latents', latents.shape) # torch.Size([bs*sum(l_i), 32, 13])

        latents = latents * 1.0
        if H.use_tanh:
            latents = (latents - 0.5) * 2.0
        if not H.norm_first:
            latents = latents / float(H.codebook_size)
        # print('normalized latents', latents.shape) # torch.Size([bs*sum(l_i), 32, 13])
        # print("Generate motions")
        motions, _, _, _ = generator(None, code=latents.float())
        # print('motions generated shape', motions.shape) # torch.Size([bs*sum(l_i), 3, 52])
        motions = motions.permute(0,2,1)
        # print('permuted motions', motions.shape) # torch.Size([bs*sum(l_i), 52, 3])

    return motions

@torch.no_grad()
def get_online_motions_guidance(H, generator, sampler, x=None, label=None):


    if x is None:
        sampler.eval()
        t = np.random.uniform(0.55, 1.0)
        g = 1.0
        if H.conditioned:
            assert label is not None, 'label must be provided for conditioned sampling'
            latents = sampler.sample(sample_steps=H.sample_steps, temp=t, label=label, guidance=g)
        else:
            latents = sampler.sample(sample_steps=H.sample_steps, temp=t, guidance=g)
        lengths = label['lengths']
        sampler.train()

    else:
        # print('LATENTS GIVEN')
        latents = x
    if x is None:
        latents_all = []

        sampler.eval()

        # print('Sampling')
        for g in [None, 1.0, 2.0, 5.0, 10.0]:
            for t in [0.5, 0.9]:
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, guidance=g)
                print('latents', latents.shape)
                latents_all.append(latents)
                # print('done')
        latents = torch.cat(latents_all, dim=0)
        # print('latents', latents.shape)
        sampler.train()
    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        print('size', size)
        motions = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]
            print('latent', latent.shape)
            print('latent', latent)
            latent = (latent * 1.0) 
            print('latent', latent)

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent.permute(0,2,1)
            print('latent', latent.shape)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            print('latent', latent.shape)
            mot, _, _ = generator(None, code=latent.float())
            print('mot', mot.shape)
            motions.append(mot)
        motions = torch.cat(motions, 0)
        print('motions', motions.shape)

    return motions


def get_samples_idx(H, generator, sampler, idx):

    latents_one_hot = latent_ids_to_onehot(idx, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)

# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_mbae_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"{H.ae_load_dir}/saved_models/mbinaryae_ema_{H.ae_load_step}.th"
    if not os.path.exists(ae_load_path):
        ae_load_path = f"{H.ae_load_dir}/saved_models/mbinaryae_{H.ae_load_step}.th"
    log(f"Loading Binary Autoencoder from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")
    # print('full_vqgan_state_dict', full_vqgan_state_dict.keys())

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                # new_key = key[3:]  # remove "ae."
                new_key = key
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]
    del full_vqgan_state_dict
    return state_dict
