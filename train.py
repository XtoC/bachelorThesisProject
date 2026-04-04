# from copy import deepcopy
from pathlib import Path

import numpy as np
import time
import torch
import torch.nn.functional as F
import torchmetrics

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common import read_yaml, init_log
#from dataset import MelClsDataset
from spectrogramDataset import SpectrogramDataset
#from models.resnet import ResnetWrapper, ResnetConfig
#from models.utils import model_size
#from utils import get_data_sizes
from vgglike import VGGLike
from specaugment import SpecAugment

@torch.no_grad()
def spectrogram_mixup(x, max_mixup=0.3):
    batch = x.size(0)
    perm = torch.randperm(batch, device=x.device)
    x2 = x[perm]

    u = torch.rand(batch, 1, 1, 1, device=x.device) * max_mixup
    return x * (1 - u) + x2 * u


def train_epoch(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None,
                max_mixup: float | None = None,
                log_interval: int | None = None,
                device: torch.device | None = None,
                specaugment = None) -> float:
    model.train()
    train_loss = 0.0
    batch_t0 = time.perf_counter()

    for batch, (x, y_true) in enumerate(loader):
        x = x.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()

        if specaugment is not None:
            x = specaugment(x)

        if max_mixup is not None:
            x = spectrogram_mixup(x, max_mixup)

        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y_true)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if log_interval is not None and batch % log_interval == 0:
            t_batch = int(1000 * (time.perf_counter() - batch_t0) / log_interval)
            lr = optimizer.param_groups[0]['lr']
            print(f'batch {batch:3d}/{len(loader)} - {t_batch} ms/batch - lr {lr:3g} - train loss {loss.item():.4f}')
            batch_t0 = time.perf_counter()

    return train_loss / len(loader)


@torch.inference_mode()
def test_epoch(model, loader, device):
    model.eval()
    test_loss = 0.0

    n_classes = 6
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes).to(device)

    for x, y_true in loader:
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)
        test_loss += F.cross_entropy(y_pred, y_true).item()
        acc_metric.update(y_pred, y_true)

    return test_loss / len(loader), acc_metric.compute().item()


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('train')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"{device}")
    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    data_fn = cache_dir / 'bsd10k-data.hdf'

    batch_size = cfg.get('batch_size', 64)
    num_workers = cfg.get('num_dataloader_workers', 8)

    train_segment_length = cfg.get('train_segment_length')

    learning_rate = cfg.get('learning_rate', 1e-3)
    warmup_pct = cfg.get('warmup_pct', 0.0)
    n_epochs = cfg.get('n_epochs', 30)
    log_interval = cfg.get('log_interval')
    max_mixup = cfg.get('max_mixup')

    #n_files, n_mels, _ = get_data_sizes(data_fn)

    # in many cases this is a bad way to do the data splits, so you should be careful
    #train_val_idx, test_idx = train_test_split(np.arange(n_files), test_size=0.3)
    #train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=303)

    ds_train = SpectrogramDataset(cfg.get('train_folder'))
    ds_val = SpectrogramDataset(cfg.get('val_folder'))
    ds_test = SpectrogramDataset(cfg.get('test_folder'))

    logger.info(f'training data size {len(ds_train)}, validation data size {len(ds_val)}, test data size {len(ds_test)}')

    n_classes = 6

    train_loader = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)
    #test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers)

    #model_cfg = ResnetConfig(n_classes=n_classes)
    #model = ResnetWrapper(model_cfg)
    model = VGGLike(n_classes)

    #logger.info(f'model size {model_size(model) / 1e6:.1f}M ({n_classes} classes)')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=learning_rate,
        total_steps=n_epochs * len(train_loader),
        pct_start=warmup_pct)

    best_acc = 0.0
    # or: best_loss = float('inf')

    # this is where  the weights are stored
    checkpoint_path = 'model_checkpoint.pt'

    train_losses = []
    val_losses = []

    specaugment = SpecAugment(
        n_time_mask=2,
        n_freq_mask=2,
        time_mask_param=50,
        freq_mask_param=16
    ).to(device)

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            max_mixup=max_mixup,
            log_interval=log_interval,
            device=device,
            specaugment=specaugment
        )

        val_loss, val_acc = test_epoch(model, val_loader, device)
        logger.info(f'epoch {epoch + 1} - training loss {train_loss:.3f} - validation loss {val_loss:.3f} - accuracy {val_acc:.3f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            #best_model = deepcopy(model.state_dict())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss.pdf")
    plt.show()

    # todo: store train_losses and val_losses

    # this goes to other file: load state dict from file (do it before moving model to gpu)
    # ckpt = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(ckpt)

    # evaluating after training:
    #model.load_state_dict(best_model)

    #test_loss, test_acc = test_epoch(model, test_loader, device)
    #logger.info(f'test loss {test_loss:.4f} - accuracy {test_acc:.4f}')


if __name__ == '__main__':
    main()