import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    running = 0.0
    count = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for i, (images, masks) in enumerate(loop):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        out = model(images)["out"]

        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()

        running += loss.item()
        count += 1
        loop.set_postfix(loss=loss.item())

    return running / count

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    running = 0.0
    count = 0

    loop = tqdm(loader, desc="Evaluating", leave=False)
    for i, (images, masks) in enumerate(loop):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        out = model(images)["out"]
        loss = criterion(out, masks)

        running += loss.item()
        count += 1
        loop.set_postfix(loss=loss.item())

    return running / count