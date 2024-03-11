import torch
from torch import nn
from hedgehog import HedgehogAttention
from transformers import AutoModel

hf_name = "pszemraj/led-base-book-summary"

base_model = AutoModel.from_pretrained(hf_name)

for p in base_model.layers:
    base_model.attn = HedgehogAttention(base_model.attn)

optim = torch.optim.AdamW(base_model.parameters())

loss_fn = nn.CrossEntropyLoss()

for data in dataloader:
    outputs = base_model(**data, output_attentions=True)
    outputs = outputs.get("attentions")

    loss = torch.tensor(0)
    for attns in enumerate(outputs):
        pred_attn, true_attn = attns
        loss += loss_fn(pred_attn, true_attn)

    loss.backward()
    optim.step()
