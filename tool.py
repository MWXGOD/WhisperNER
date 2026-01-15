import json






class Hyperargs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k ,v)


def read_json_args(args_path):
    with open(args_path, 'r', encoding='utf-8') as f:
        args_dict = json.load(f)
    return args_dict





def add_special_tokens(tokenizer, task_tokens):
    # 关键：先取出原有的 additional_special_tokens
    old_specials = tokenizer.additional_special_tokens

    # 再合并（去重）
    new_specials = old_specials + [
        t for t in task_tokens if t not in old_specials
    ]

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": new_specials
    })
    return tokenizer, num_added

# def del_surpress_tokens(tokenizer, white_list=None):
#     if white_list is None:
#         white_list = ['<', '>', '(', ')', '[', ']', '-', '$', '$$', '#', '##', ]
#     for wl in white_list:
#         if tokenizer.convert_tokens_to_ids(wl) in tokenizer.generation_config.suppress_tokens:
#             tokenizer.generation_config.suppress_tokens.remove(tokenizer.convert_tokens_to_ids(wl))
#     return tokenizer
