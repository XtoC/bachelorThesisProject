import torch
from train import test_epoch
from common import read_yaml, init_log
from vgglike import VGGLike
from spectrogramDataset import SpectrogramDataset

def main():
    logger = init_log("test")
    cfg = read_yaml("settings.yaml")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    checkpoint_path = 'model_checkpoint.pt'
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = VGGLike(6)
    model = model.to(device)
    model.load_state_dict(ckpt)

    batch_size = cfg.get('batch_size', 64)
    num_workers = cfg.get('num_dataloader_workers', 8)

    ds_test = SpectrogramDataset(cfg.get('test_folder'))
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers)

    # evaluating after training:

    test_loss, test_acc = test_epoch(model, test_loader, device)
    logger.info(f'test loss {test_loss:.4f} - accuracy {test_acc:.4f}')

if __name__ == "__main__":
    main()