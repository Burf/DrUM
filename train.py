import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from drum import set_seed
set_seed(42)

import functools
import gc
import tqdm
import numpy as np
import torch
import torchvision
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from drum import Dataset, backbone, collate_fn, AdamW, cosine_lr, sim_loss, save_pickle


batch_size = 256
epoch = 20
lr = 5e-4
accumulation_steps = 256 // batch_size

model_name = "/weight/L"

ds = load_dataset("pixparse/cc3m-wds")
train_ds = Dataset(ds["train"]["txt"])

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

collate_fn_partial = functools.partial(collate_fn, processor = processor)

tr_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = collate_fn_partial)

drum = backbone(model, processor, n_layer = 10, pos = False, cls_pos = False, dropout = 0.1)
optimizer = AdamW(drum, lr, weight_decay = 0.2)
scheduler = cosine_lr(optimizer, lr, 10000 // (batch_size // 64), len(tr_loader) * epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drum.to(device)
drum.train()

empty, pool = drum.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
drum.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))
#drum.adapter.set_base_query(empty) #t5

step = 0
optimizer.zero_grad()
for ep in range(epoch):
    ep_step = 0
    loss_list = []
    best = np.inf
    tbar = tqdm.tqdm(total = len(tr_loader), desc = "[{0}/{1}]".format(ep + 1, epoch), leave = True)
    for i, xy in enumerate(tr_loader):
        step += 1
        ep_step += 1
        scheduler(step)

        xy = {k:v.to(device) for k, v in xy.items()}

        pred, pool_pred = drum(xy, None, skip = -1, pooling = True, normalize = False, normalize_pool = False, training = True)
        ori, pool_ori = drum.encode_prompt(xy, skip = -1, pooling = True, normalize = False, normalize_pool = False)

        loss = sim_loss(pred, ori) + sim_loss(pool_pred, pool_ori)
        #loss = sim_loss(pred, xy) #t5

        (loss / accumulation_steps).backward()
        loss = loss.detach().item()
        loss_list += ([loss] * len(xy))
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.mean(loss_list)
        tbar.update(1)
        tbar.set_description("[{0:03d}/{1:03d}] lr: {2:.4e} avg_loss: {3:.4f} batch_loss: {4:.4f}".format(ep + 1, epoch, optimizer.param_groups[0]["lr"], avg_loss, loss))

    if True:#avg_loss <= best:
        best = avg_loss
        torch.save(drum.adapter.state_dict(), "./{0}-e{1}-{2:.4f}.pth".format(model_name, ep + 1, avg_loss))
    save_pickle(loss_list, "./{0}-e{1}-{2:.4f}.pickle".format(model_name, ep + 1, avg_loss))
    tbar.close()
