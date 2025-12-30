import json






class Hyperargs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k ,v)


def read_json_args(args_path):
    with open(args_path, 'r', encoding='utf-8') as f:
        args_dict = json.load(f)
    return args_dict