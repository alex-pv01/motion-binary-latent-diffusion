import os
import torch


def findAllFile(base, debug=False):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    print("Searching for files in {}...".format(base))
    file_path = []

    # Limiting the number of files for debugging purposes
    if debug > 0:
        i = 1
        for root, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                fullname = os.path.join(root, f)
                file_path.append(fullname)
                i += 1
                if i > debug:
                    return file_path
                
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def collate_tensors(batch):
    # Function for collating a batch of PyTorch tensors
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate(batch):
    # Adapter function for collating batches in the MotionDatasetV2 class
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "name": [b[1] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
    }

    return adapted_batch

def mx_collate(batch):
    # Adapter function for collating batches in the Text2MotionDataset class
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "motion":[torch.tensor(b[0]).float() for b in notnone_batches],
        #collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "motion_path": [b[1] for b in notnone_batches],
        "text": [b[2] for b in notnone_batches],
        "text_path": [b[3] for b in notnone_batches],
        "name": [b[4] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[5]).float() for b in notnone_batches]),
        "word_embs": [b[6] for b in notnone_batches],
        "pos_one_hots": [b[7] for b in notnone_batches],
    }

    return adapted_batch




import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['motion'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    namebatch = [b['name'] for b in notnone_batches]
    cond['y'].update({'name': namebatch})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'motion': torch.tensor(b[4].T).float(),#.unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'lengths': b[5],
        'tokens': b[6],
        'name': b[7],
    } for b in batch]
    return collate(adapted_batch)


