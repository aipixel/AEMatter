from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect') -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape
    kernel_numel: int = height * width
    if kernel_numel > 81:
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


def make_lp(img, kernel, max_levels, pad_type):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = filter2D(current, kernel, pad_type)
        diff = current - filtered
        pyr.append(diff)
        current = torch.nn.functional.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr

class inverse_huber_loss(nn.Module):
    def __init__(self,):
        super(inverse_huber_loss, self).__init__()
    def forward(self, input, target):
        absdiff = torch.abs(input-target)
        C = 0.2*torch.max(absdiff).item()
        return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))

class lap_loss(nn.Module):
    def __init__(self, max_levels=5, k_size=(5, 5), sigma=(1.5, 1.5), board_type='reflect', loss_type='L1',
                 loss_multiplier=2,clip=True,clipmin=0.,clipmax=1.,reduction='mean'):
        super(lap_loss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self.board_type = board_type
        self._gauss_kernel = torch.unsqueeze(get_gaussian_kernel2d(k_size, sigma), dim=0)
        self.clip=clip
        self.clipmin = clipmin
        self.clipmax = clipmax
        loss_list: List[str] = ['L1', 'L2','IHuber']
        self.loss_multiplier = loss_multiplier
        if loss_type not in loss_list:
            raise ValueError("Invalid loss_type, we expect the following: {0}."
                             "Got: {1}".format(loss_list, loss_type))
        self.loss_type = loss_type
        if self.loss_type == 'L1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif self.loss_type == 'L2':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.loss_type=='IHuber':
            self.loss = inverse_huber_loss()

    def forward(self, input, target):
        if self.clip:
            input=torch.clamp(input,self.clipmin,self.clipmax)
        pyr_input = make_lp(input, self._gauss_kernel, self.max_levels, self.board_type)
        pyr_target = make_lp(target, self._gauss_kernel, self.max_levels, self.board_type)
        losses = []
        mul = 1
        for x in range(self.max_levels):
            losses.append(mul * self.loss(pyr_input[x], pyr_target[x]))
            mul *= self.loss_multiplier
        return sum(losses)

class grad_loss(nn.Module):
    def __init__(self, loss_type='L1',clip=False,clipmin=0.,clipmax=1.,reduction='mean'):
        super(grad_loss, self).__init__()
        loss_list: List[str] = ['L1', 'L2']
        if loss_type not in loss_list:
            raise ValueError("Invalid loss_type, we expect the following: {0}."
                             "Got: {1}".format(loss_list, loss_type))
        self.loss_type = loss_type
        if self.loss_type == 'L1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif self.loss_type == 'L2':
            self.loss = nn.MSELoss(reduction=reduction)
        self.clip=clip
        self.clipmin = clipmin
        self.clipmax = clipmax

    def forward(self, input, target):
        if self.clip:
            input=torch.clamp(input,self.clipmin,self.clipmax)
        inputx_=input[:,:,0:-1,:]-input[:,:,1:,:]
        inputy_=input[:,:,:,0:-1]-input[:,:,:,1:]
        targetx_=target[:,:,0:-1,:]-target[:,:,1:,:]
        targety_=target[:,:,:,0:-1]-target[:,:,:,1:]
        loss=self.loss(inputx_,targetx_)+self.loss(inputy_,targety_)
        return loss


if __name__ == '__main__':
    l = lap_loss().cuda()
    a = torch.randn(5, 1, 128, 128).cuda()
    b = torch.randn(5, 1, 128, 128).cuda()
    c = l(a, a/4)
    print(c)
    c = l(a, a/2)
    print(c)
    c = l(a, a*3/4)
    print(c)