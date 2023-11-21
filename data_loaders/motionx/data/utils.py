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
    }

    return adapted_batch