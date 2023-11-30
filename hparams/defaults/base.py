class HparamsBase(dict):
    def __init__(self, dataset, dataset_type):
        self.dataset = dataset
        self.dataset_type = dataset_type

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value
