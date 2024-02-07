import torch

from models.UGDC import UGDC, UNet, UGDC_Enhance, UNet_Enhance


def get_model(num_classes, name, ckpt=None, stem_channels=32):
    model_names = ['ugdc', 'unet', 'ugdc_enhance', 'unet_enhance']

    if name not in model_names:
        raise NotImplementedError(f'model name should be one of: {model_names}, but got {name}.')

    if name == 'ugdc':
        model = UGDC(n_channels=3, n_classes=num_classes,
                        in_shapes=[(400, 640), (200, 320), (100, 160), (50, 80), (25, 40)])

    elif name == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes)

    elif name == 'unet_enhance':
        model = UNet_Enhance(n_channels=3, n_classes=num_classes)

    elif name == 'ugdc_enhance':
        model = UGDC_Enhance(n_channels=3, n_classes=num_classes,
                        in_shapes=[(400, 640), (200, 320), (100, 160), (50, 80), (25, 40)], stem_channels=stem_channels)

    if ckpt:
        print(f'loading ckpt from {ckpt}...')
        checkpoint = torch.load(ckpt)
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k[7:]] = v

        model.load_state_dict(new_state_dict)
        model.eval()

    return model