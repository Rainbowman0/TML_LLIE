import argparse
import os


import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import save_image

from data.mydataset import MyDS
from utils.train_utils import get_model


parser = argparse.ArgumentParser(description='PyTorch TML Evaluation')

# Data configuration
parser.add_argument('data', metavar='DIR', nargs='?', help='path to dataset')
parser.add_argument('--save_dir', type=str, default="./save", help="path to save info")

# Model configuration
parser.add_argument('-a', '--arch', metavar='ARCH', default='uldc')
parser.add_argument('--resume-Pred', default='', type=str, metavar='PATH',
                    help='path to latest Pred model checkpoint (default: none)')
parser.add_argument('--resume-Enhance', default='', type=str, metavar='PATH',
                    help='path to latest Enhance model checkpoint (default: none)')

# Training configuration
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


def main():
    args = parser.parse_args()
    evaluate(args)


def evaluate(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MyDS(args.data)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    trans = transforms.Normalize(mean=-mean / std, std=1 / std)

    model_Pred = get_model(3, 'ugdc', args.resume_Pred).to(device)
    model_Enhance = get_model(3, 'ugdc_enhance', args.resume_Enhance).to(device)

    for low, low_path in dataloader:
        print(f'processing {low_path} ...')

        low = low.to(device)
        course = model_Pred(low)
        noise = model_Enhance(course)
        output = course - noise



        for i in range(output.shape[0]):
            output = trans(output[i])
            save_image(output, os.path.join(args.save_dir, low_path[0].split('/')[-1]))


    print(f'normal light images are saved in {args.save_dir}')



if __name__ == '__main__':
    main()
    print('done!')