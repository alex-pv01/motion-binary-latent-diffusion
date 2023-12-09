import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
import pdb
from torch import nn


class Sampler(nn.Module):
    def __init__(self, H, embedding_weight):
        super().__init__()
        self.latent_shape = H.latent_shape
        self.emb_dim = H.emb_dim
        self.codebook_size = H.codebook_size
        self.embedding_weight = embedding_weight
        self.embedding_weight.requires_grad = False
        self.n_samples = H.n_samples

    def train_iter(self, x, x_target, step):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def class_conditional_train_iter(self, x, y):
        raise NotImplementedError()

    def class_conditional_sample(n_samples, y):
        raise NotImplementedError()

    def embed(self, z):
        with torch.no_grad():
            z_flattened = z.view(-1, self.codebook_size)  # B*H*W, codebook_size
            embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                z.size(0),
                self.latent_shape[1],
                self.latent_shape[2],
                self.emb_dim
            ).permute(0, 3, 1, 2).contiguous()

        return embedded


class BinaryDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule

        self.loss_final = H.loss_final
        self.use_softmax = H.use_softmax

        self.scheduler = noise_scheduler(self.num_timesteps, beta_type=H.beta_type)
        self.p_flip = H.p_flip
        self.focal = H.focal
        self.aux = H.aux
        self.dataset = H.dataset
        self.guidance = H.guidance

    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        return t

    def q_sample(self, x_0, t):
        x_t = self.scheduler(x_0, t) # t >= 1 <=T#
        return x_t

    def _train_loss(self, x_0, label=None, x_ct=None):
        # statss = []
        # for x_0 in x_0s:
        # print('x_0', x_0.shape)
        x_0 = x_0 * 1.0
        b, device = x_0.size(0), x_0.device
        # print('b', b)
        # print('device', device)

        # choose what time steps to compute loss at
        t = self.sample_time(b, device)
        # print('t', t.shape)

        # make x noisy and denoise
        if x_ct is None:
            x_t = self.q_sample(x_0, t)
            # print('x_t', x_t.shape)
        else:
            # print('x_ct', x_ct.shape)
            x_t = self.scheduler.sr_forward(x_0, x_ct, t)
            # print('x_t', x_t.shape)

        x_t_in = torch.bernoulli(x_t)
        # print('x_t_in', x_t_in.shape)
        if label is not None:
            # print('label', label.shape)
            if self.guidance and np.random.random() < 0.1:
                label = None
            x_0_hat_logits = self._denoise_fn(x_t_in, label=label, time_steps=t-1) 
            # print('x_0_hat_logits', x_0_hat_logits.shape)
        else:
            x_0_hat_logits = self._denoise_fn(x_t_in, time_steps=t-1)
            # print('x_0_hat_logits', x_0_hat_logits.shape)


        if self.p_flip:
            if self.focal >= 0:
                x_0_ = torch.logical_xor(x_0, x_t_in)*1.0
                # print('x_0_', x_0_.shape)
                kl_loss = focal_loss(x_0_hat_logits, x_0_, gamma=self.focal)
                # print('kl_loss', kl_loss.shape)
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
                # print('x_0_hat_logits', x_0_hat_logits.shape)
            else:
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
                # print('x_0_hat_logits', x_0_hat_logits.shape)
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')
                # print('kl_loss', kl_loss.shape)

        else:
            if self.focal >= 0:
                kl_loss = focal_loss(x_0_hat_logits, x_0, self.focal, gamma=self.focal)
                # print('kl_loss', kl_loss.shape)
            else:
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')
                # print('kl_loss', kl_loss.shape)

        if torch.isinf(kl_loss).max():
            pdb.set_trace()

        if self.loss_final == 'weighted':
            weight = (1 - ((t-1) / self.num_timesteps)).view(-1, 1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        with torch.no_grad():
            if self.use_softmax:
                acc = (((x_0_hat_logits[..., 1] > x_0_hat_logits[..., 0]) * 1.0 == x_0.view(-1)) * 1.0).sum() / float(x_0.numel())
            else:
                acc = (((x_0_hat_logits > 0.0) * 1.0 == x_0) * 1.0).sum() / float(x_0.numel())

        if self.aux > 0:
            ftr = (((t-1)==0)*1.0).view(-1, 1, 1)
            # print('ftr', ftr.shape)

            x_0_l = torch.sigmoid(x_0_hat_logits)
            # print('x_0_l', x_0_l.shape)
            x_0_logits = torch.cat([x_0_l.unsqueeze(-1), (1-x_0_l).unsqueeze(-1)], dim=-1)
            # print('x_0_logits', x_0_logits.shape)
            x_t_logits = torch.cat([x_t_in.unsqueeze(-1), (1-x_t_in).unsqueeze(-1)], dim=-1)
            # print('x_t_logits', x_t_logits.shape)

            p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t-1)
            # print('p_EV_qxtmin_x0', p_EV_qxtmin_x0.shape)

            q_one_step = self.scheduler.one_step(x_t_logits, t)
            # print('q_one_step', q_one_step.shape)
            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            # print('unnormed_probs', unnormed_probs.shape)
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            # print('unnormed_probs', unnormed_probs.shape)
            unnormed_probs = unnormed_probs[...,0]
            # print('unnormed_probs', unnormed_probs.shape)
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_0_l * ftr
            # print('x_tm1_logits', x_tm1_logits.shape)
            x_0_gt = torch.cat([x_0.unsqueeze(-1), (1-x_0).unsqueeze(-1)], dim=-1)
            # print('x_0_gt', x_0_gt.shape)
            p_EV_qxtmin_x0_gt = self.scheduler(x_0_gt, t-1)
            # print('p_EV_qxtmin_x0_gt', p_EV_qxtmin_x0_gt.shape)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            # print('unnormed_gt', unnormed_gt.shape)
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            # print('unnormed_gt', unnormed_gt.shape)
            unnormed_gt = unnormed_gt[...,0]
            # print('unnormed_gt', unnormed_gt.shape)

            x_tm1_gt = unnormed_gt
            # print('x_tm1_gt', x_tm1_gt.shape)

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
            aux_loss = F.binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0), reduction='none')
            # print('aux_loss', aux_loss.shape)

            aux_loss = (weight * aux_loss).mean()
            loss = self.aux * aux_loss + loss

        stats = {'loss': loss, 'bce_loss': kl_loss, 'acc': acc}

        if self.aux > 0:
            stats['aux loss'] = aux_loss
            
        #     statss.append(stats)

        # # Sum stats over all x_0s
        # stats = {}
        # for k in statss[0].keys():
        #     stats[k] = torch.stack([s[k] for s in statss], 0).sum(0)

        return stats
    
    def sample(self, temp=1.0, sample_steps=None, return_all=False, label=None, mask=None, guidance=None, length=100, full=False):
        device = 'cuda'
        # print('SAMPLING')
        # print('self.shape', self.shape)
        #x_t = torch.bernoulli(0.5 * torch.ones((length, np.prod(self.shape), self.codebook_size), device=device))
        x_t = torch.bernoulli(0.5 * torch.ones((self.shape[-1], self.codebook_size, length), device=device)).unsqueeze(0)
        # print('x_t', x_t.shape)
        # print('x_t', x_t)
        if mask is not None:
            # print('MASK')
            m = mask['mask'].unsqueeze(0)
            latent = mask['latent'].unsqueeze(0)
            x_t = latent * m + x_t * (1-m)
        sampling_steps = np.array(range(1, self.num_timesteps+1))
        # print('sampling_steps', sampling_steps)

        if sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, sample_steps)
            idx = np.array(idx * (self.num_timesteps-1), int)
            sampling_steps = sampling_steps[idx]

        if return_all:
            x_all = [x_t]

        # if self.dataset == 'motionx':
            # if label is None:
            #     print('LABEL IS NONE')
            #     label = torch.arange(100, device=device) * 100
            #     label = label.long()
            #     print('label', label.shape)
            #     print('label', label)
            #     label = None

            #     text = 'no label ' * 10 
            #     print('text', text)
            #     label = {'lengths': length, 'text': label}
            # else:
            #     print('LABEL IS NOT NONE')
            #     label = torch.full((length,), label, device=device, dtype=torch.long)
            #     print('label', label.shape)
            #     print('label', label)
        
        sampling_steps = sampling_steps[::-1]


        for i, t in enumerate(sampling_steps):
            # t = torch.full((length,), t, device=device, dtype=torch.long)
            t = torch.tensor([t], device=device, dtype=torch.long)
            # print('t', t.shape)

            if self.dataset.startswith('motionx'):
                # print('motionx')
                # print('t', t.shape)
                # print('x_t', x_t.shape)
                # print('t', t)
                x_0_logits = self._denoise_fn(x=x_t, label=label, time_steps=t-1)
                x_0_logits = x_0_logits / temp
                # print('x_0_logits', x_0_logits.shape)
                if guidance is not None:
                    x_0_logits_uncond = self._denoise_fn(x=x_t, label=None, time_steps=t-1)
                    x_0_logits_uncond = x_0_logits_uncond / temp

                    x_0_logits = (1 + guidance) * x_0_logits - guidance * x_0_logits_uncond
            else:
                x_0_logits = self._denoise_fn(x_t, time_steps=t-1)
                x_0_logits = x_0_logits / temp
                # scale by temperature

            x_0_logits = torch.sigmoid(x_0_logits)
            # print('x_0_logits', x_0_logits.shape)


            if self.p_flip:
                # print('p_flip')
                x_0_logits =  x_t * (1 - x_0_logits) + (1 - x_t) * x_0_logits
                # print('x_0_logits', x_0_logits.shape)

            if not t[0].item() == 1:
                t_p = torch.tensor(sampling_steps[i+1], device=device, dtype=torch.long)
                # print('t_p', t_p.shape)
                
                x_0_logits = torch.cat([x_0_logits.unsqueeze(-1), (1-x_0_logits).unsqueeze(-1)], dim=-1)
                # print('x_0_logits', x_0_logits.shape)
                x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)
                # print('x_t_logits', x_t_logits.shape)


                p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t_p)
                # print('p_EV_qxtmin_x0', p_EV_qxtmin_x0.shape)
                q_one_step = x_t_logits

                for mns in range(sampling_steps[i] - sampling_steps[i+1]):
                    q_one_step = self.scheduler.one_step(q_one_step, t - mns)

                unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                unnormed_probs = unnormed_probs[...,0]
                
                x_tm1_logits = unnormed_probs
                x_tm1_p = torch.bernoulli(x_tm1_logits)
            
            else:
                x_0_logits = x_0_logits
                x_tm1_p = (x_0_logits > 0.5) * 1.0

            x_t = x_tm1_p

            if mask is not None:
                m = mask['mask'].unsqueeze(0)
                latent = mask['latent'].unsqueeze(0)
                x_t = latent * m + x_t * (1-m)

            # print('x_t', x_t.shape)


            if return_all:
                x_all.append(x_t)
        if return_all:
            # print('return all')
            return torch.cat(x_all, 0)
        else:
            # print('return last')
            # print('x_t', x_t.shape)
            return x_t
    
    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label, x_t)


class noise_scheduler(nn.Module):
    def __init__(self, steps=40, beta_type='linear'):
        super().__init__()


        if beta_type == 'linear':

            beta = 1 - 1 / (steps - np.arange(1, steps+1) + 1) 

            k_final = [1.0]
            b_final = [0.0]

            for i in range(steps):
                k_final.append(k_final[-1]*beta[i])
                b_final.append(beta[i] * b_final[-1] + 0.5 * (1-beta[i]))

            k_final = k_final[1:]
            b_final = b_final[1:]


        elif beta_type == 'cos':

            k_final = np.linspace(0.0, 1.0, steps+1)

            k_final = k_final * np.pi
            k_final = 0.5 + 0.5 * np.cos(k_final)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]
        
        elif beta_type == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + np.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return np.clip(output, clip_min, 1.)
            
            k_final = np.linspace(0.0, 1.0, steps+1)
            k_final = sigmoid_schedule(k_final, 0, 3, 0.8)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]


        else:
            raise NotImplementedError
        
        k_final = np.hstack([1, k_final])
        b_final = np.hstack([0, b_final])
        beta = np.hstack([1, beta])
        self.register_buffer('k_final', torch.Tensor(k_final))
        self.register_buffer('b_final', torch.Tensor(b_final))
        self.register_buffer('beta', torch.Tensor(beta))  
        self.register_buffer('cumbeta', torch.cumprod(self.beta, 0))  
        # pdb.set_trace()

        print(f'Noise scheduler with {beta_type}:')

        print(f'Diffusion 1.0 -> 0.5:')
        data = (1.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print(f'Diffusion 0.0 -> 0.5:')
        data = (0.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print(f'Beta:')
        print(' '.join([f'{d:0.4f}' for d in self.beta.data.numpy()]))

    
    def one_step(self, x, t):
        dim = x.ndim - 1
        k = self.beta[t].view(-1, *([1]*dim))
        x = x * k + 0.5 * (1-k)
        return x

    def forward(self, x, t):
        dim = x.ndim - 1
        k = self.k_final[t].view(-1, *([1]*dim))
        b = self.b_final[t].view(-1, *([1]*dim))
        out = k * x + b
        return out
    

def focal_loss(inputs, targets, alpha=-1, gamma=1):
    # print('focal loss')
    # print('inputs', inputs.shape)
    # print('targets', targets.shape)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    p_t = (1 - p_t)
    p_t = p_t.clamp(min=1e-6, max=(1-1e-6)) # numerical safety
    loss = ce_loss * (p_t ** gamma)
    if alpha == -1:
        neg_weight = targets.sum((-1, -2, -3))
        neg_weight = neg_weight / targets[0].numel()
        neg_weight = neg_weight.view(-1, 1, 1, 1)
        # print('neg_weight', neg_weight.shape)
        alpha_t = (1 - neg_weight) * targets + neg_weight * (1 - targets)
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss






