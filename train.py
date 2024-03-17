from copy import deepcopy
from hedgehog import HedgehogAttention
from torch import nn
from transformers import AutoModel
from transformers.models.bart.modeling_bart import BartSdpaAttention
import torch
import transformers

hf_name = "sshleifer/distilbart-cnn-6-6"

base_model = AutoModel.from_pretrained(hf_name)

print(base_model)


def replace_layer(module):
    if isinstance(module, BartSdpaAttention):
        target_state_dict = deepcopy(module.state_dict())
        new_module = HedgehogAttention(module)
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module


def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


for name, module in tuple(base_model.named_modules()):
    if name:
        recursive_setattr(base_model, name, replace_layer(module))

print(base_model)

optim = torch.optim.AdamW(base_model.parameters())

loss_fn = nn.CrossEntropyLoss()

# for data in dataloader:
#     outputs = base_model(**data, output_attentions=True)
#     outputs = outputs.get("attentions")

#     loss = torch.tensor(0)
#     for attns in enumerate(outputs):
#         pred_attn, true_attn = attns
#         loss += loss_fn(pred_attn, true_attn)

#     loss.backward()
#     optim.step()
