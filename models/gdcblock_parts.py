from timm.models.layers import DropPath

from .unet_parts import *


class SABlock(nn.Module):
    r"""
    Args:
        in_shape (int): (H, W), height and width of input.
        dim (int): Number of input channels.
        ps (int): Patch size. H and W should be divisible by ps.
        ks (int): Dynamic Conv kernel size. Default: 1
        reduction (int): Channel reduction ratio.
        drop_path (float): Stochastic depth rate. Default: 0.2
    """

    def __init__(self, in_shape, dim, ps=7, ks=1, reduction=4, drop_path=0.2):
        super().__init__()
        if isinstance(in_shape, int):
            h, w = in_shape, in_shape
            self.patch = nn.Conv2d(dim, dim, kernel_size=ps, stride=ps, groups=dim)

        else:
            h, w = in_shape[0], in_shape[1]
            ps = 5
            self.patch = nn.Conv2d(dim, dim, kernel_size=ps, stride=ps, groups=dim)

        self.ks = ks
        self.lnnorm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.bias = nn.Parameter(torch.zeros(1))
        d = torch.normal(mean=0, std=1.0, size=(dim, h // ps, w // ps))
        self.diff = nn.Parameter(d)

        self.convk = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(h * w // (ps * ps), dim, kernel_size=1)

        self.act = nn.GELU()

        self.ln = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        n, c, h, w = x.shape
        k = x
        q = self.patch(x) + self.diff  # (N, C, H, W) --> (N, C, H//ps, W//ps)
        q = self.act(self.lnnorm(q))  # (N, C*ks*ks, H//ps, W//ps), ks = 1
        q = q.permute(0, 2, 3, 1)
        q = q.reshape(-1, c, self.ks, self.ks)  # (N*H*W/(ps*ps), C, ks, ks)
        k = self.convk(k)  # (N, C, H, W)
        k = k.reshape(1, n * c, h, w)  # (1, N*C, H, W)
        # get a1
        a1 = F.conv2d(k, q, self.bias.repeat(q.shape[0]), padding=(self.ks - 1) // 2,
                      groups=n)  # (1, N*H*W/(ps*ps), H, W)
        a1 = a1.reshape(n, -1, h, w)  # (N, H*W/(ps*ps), H, W)
        res = self.conv1(a1)

        # res = input + res
        res = input + self.drop_path(res)
        return res



class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DoubleConv_SA(nn.Module):

    def __init__(self, in_channels, out_channels, in_shape):
        super().__init__()
        self.double_conv = nn.Sequential(
            SABlock(in_shape=in_shape, dim=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_SA(nn.Module):

    def __init__(self, in_channels, out_channels, in_shape):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_SA(in_channels, out_channels, in_shape)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_SA(nn.Module):

    def __init__(self, in_channels, out_channels, in_shape, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv_SA(in_channels, out_channels, in_shape)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)