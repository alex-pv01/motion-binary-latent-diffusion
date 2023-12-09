class HparamsBase(dict):
    def __init__(self, dataset, dataset_type, sampler=None):
        self.dataset = dataset
        self.dataset_type = dataset_type
        if sampler is not None:
            self.sampler = sampler

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value
