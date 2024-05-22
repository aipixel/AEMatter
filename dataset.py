import math
import numbers
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa
mattingpath= '/disk2/adb/'
backgroundpath= '/disk2/coco/'
realworldaug=False
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

RWA = iaa.SomeOf((1, None), [
    iaa.LinearContrast((0.6, 1.4)),
    iaa.JpegCompression(compression=(0, 60)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
], random_order=True)


seqv3gf = iaa.Sequential(
    [
        iaa.SomeOf((0, 5),
                   [   iaa.GammaContrast((0.9, 1.2), per_channel=True),
                       iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-30, 30)), # change hue and saturation
                       iaa.Multiply((0.8, 1.25), per_channel=0.5),
                       iaa.LinearContrast((0.8, 1.25), per_channel=0.5), # improve or worsen the contrast
                       ],
                   random_order=True
                   )
    ],
    random_order=True
)

seqv3gb = iaa.Sequential(
    [
        iaa.SomeOf((0, 5),
                   [   iaa.GammaContrast((0.9, 1.1), per_channel=True),
                       iaa.Add((-3, 3), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-3, 3)), # change hue and saturation
                       iaa.Multiply((0.95, 1.05), per_channel=0.5),
                       iaa.LinearContrast((0.9, 1.1), per_channel=0.5), # improve or worsen the contrast
                       ],
                   random_order=True
                   )
    ],
    random_order=True
)

class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """

        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.shape)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.shape)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))
        rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=rs + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=rs + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix





class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample



class NToTensor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image):
        image = image[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        image = torch.from_numpy(image).sub_(self.mean).div_(self.std)
        return image





class CropAroundG(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        fg = results['fg']
        alpha = results['alpha']
        trimap = results['Temptrimap']
        bg = results['bg']


        h, w = fg.shape[:2]
        assert bg.shape == fg.shape, (f'shape of bg {bg.shape} should be the '
                                      f'same as fg {fg.shape}.')

        crop_h, crop_w = random.choice(self.crop_size)
        # Make sure h >= crop_h, w >= crop_w. If not, rescale imgs

        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:
            if random.random()>0.9:
                new_h = max(int(h * rescale_ratio), crop_h)
                new_w = max(int(w * rescale_ratio), crop_w)
                rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
                fg = cv2.resize(fg, (new_w, new_h), interpolation=rs)
                alpha = cv2.resize(
                    alpha, (new_w, new_h), interpolation=rs)
                trimap = cv2.resize(
                    trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (new_w, new_h), interpolation=rs)
                h, w = new_h, new_w
            else:
                new_h = max(int(h * rescale_ratio), crop_h)
                new_w = max(int(w * rescale_ratio), crop_w)
                ph=new_h-h
                pw=new_w-w
                hwhw=[0,0,0,0]
                hh=random.choice([0,1])
                ww = random.choice([2, 3])
                hwhw[hh]=ph
                hwhw[ww]=pw
                ph1=hwhw[0]
                ph2=hwhw[1]
                pw1=hwhw[2]
                pw2=hwhw[3]
                fg = cv2.copyMakeBorder(fg,ph1,ph2,pw1,pw2,cv2.BORDER_REFLECT)
                alpha = cv2.copyMakeBorder(alpha,ph1,ph2,pw1,pw2,cv2.BORDER_REFLECT)
                trimap =  cv2.copyMakeBorder(trimap,ph1,ph2,pw1,pw2,cv2.BORDER_REFLECT)
                bg =  cv2.copyMakeBorder(bg,ph1,ph2,pw1,pw2,cv2.BORDER_REFLECT)

                h, w = new_h, new_w

        if random.random() > 0.6:
            fg = cv2.copyMakeBorder(fg, 32, 32, 32, 32, cv2.BORDER_REFLECT)
            alpha = cv2.copyMakeBorder(alpha, 32, 32, 32, 32, cv2.BORDER_REFLECT)
            trimap = cv2.copyMakeBorder(trimap, 32, 32, 32, 32, cv2.BORDER_REFLECT)
            bg = cv2.copyMakeBorder(bg, 32, 32, 32, 32, cv2.BORDER_REFLECT)

            h, w = alpha.shape
        #
        # tritemp = np.zeros([*trimap.shape, 2], np.float32)
        # tritemp[:, :, 0] = (trimap == 0)
        # tritemp[:, :, 1] = (trimap == 255)
        #
        # sixc = trimap_transform(tritemp)
        small_trimap = cv2.resize(
            trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        margin_h, margin_w = crop_h // 2, crop_w // 2
        sample_area = small_trimap[margin_h // 4:(h - margin_h) // 4,
                      margin_w // 4:(w - margin_w) // 4]
        unknown_xs, unknown_ys = np.where(sample_area == 128)
        unknown_num = len(unknown_xs)
        if unknown_num < 10:
            # too few unknown area in the center, crop from the whole image
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
        else:
            idx = np.random.randint(unknown_num)
            top = unknown_xs[idx] * 4
            left = unknown_ys[idx] * 4
        bottom = top + crop_h
        right = left + crop_w

        results['fg'] = fg[top:bottom, left:right]
        results['alpha'] = alpha[top:bottom, left:right]
        results['bg'] = bg[top:bottom, left:right]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'





class CropAroundBG(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):

        bg = results['bg2']
        h, w = bg.shape[:2]
        crop_h, crop_w = random.choice(self.crop_size)


        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:

            new_h = max(int(h * rescale_ratio), crop_h)
            new_w = max(int(w * rescale_ratio), crop_w)
            rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            bg = cv2.resize(bg, (new_w, new_h), interpolation=rs)
            h, w = new_h, new_w

        if random.random() > 0.6:
            bg = cv2.copyMakeBorder(bg, 32, 32, 32, 32, cv2.BORDER_REFLECT)
            h, w,c = bg.shape

        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        bottom = top + crop_h
        right = left + crop_w
        results['bg2'] = bg[top:bottom, left:right]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'





class RandomJitterDualBG(object):
    def __init__(self, hue_range=40):
        if isinstance(hue_range, numbers.Number):
            assert hue_range >= 0, ('If hue_range is a single number, '
                                    'it must be positive.')
            self.hue_range = (-hue_range, hue_range)
        else:
            assert isinstance(hue_range, tuple) and len(hue_range) == 2, \
                'hue_range should be a tuple and it must be of length 2.'
            self.hue_range = hue_range

    def __call__(self, results):
        fg, alpha, bg ,bg2= results['fg'], results['alpha'], results['bg'],results['bg2']
        fg = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2RGB)
        fg = fg[np.newaxis, :, :, :]
        fg = seqv3gf(images=fg)[0]
        fg = fg[:, :, ::-1]
        results['fg'] = fg
        fg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2RGB)
        fg = fg[np.newaxis, :, :, :]
        fg = seqv3gb(images=fg)[0]
        fg = fg[:, :, ::-1]
        results['bg'] = fg
        fg= cv2.cvtColor(bg2.astype(np.uint8), cv2.COLOR_BGR2RGB)
        fg = fg[np.newaxis, :, :, :]
        fg = seqv3gb(images=fg)[0]
        fg = fg[:, :, ::-1]
        results['bg2'] = fg
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'hue_range={self.hue_range}'


class RandomBlur(object):
    def __init__(self,mb=True):
        self.mb=mb
    def __call__(self, results):
        fg, alpha, bg,bg2 = results['fg'], results['alpha'], results['bg'], results['bg2']
        blurfg=fg
        blurbg=bg
        blurbg2=bg2
        bluralpha=alpha

        if random.random()>0.95:
            sigma=random.random()*2+0.00001
            blurfg=cv2.GaussianBlur(blurfg,(5,5), sigmaX=sigma)
            bluralpha=cv2.GaussianBlur(bluralpha,(5,5),sigmaX=sigma)
        results['blurfg'] = blurfg
        results['blurbg'] = blurbg
        results['blurbg2'] = blurbg2
        results['bluralpha'] = bluralpha
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'hue_range={self.hue_range}'


class MergeFgAndBgDual(object):
    def __call__(self, results):
        alpha = results['bluralpha'][..., None].astype(np.float32) / 255.
        fg = results['blurfg']
        bg = results['blurbg']
        bg2 = results['blurbg2']
        merged = fg * alpha + (1. - alpha) * bg
        merged2 = fg * alpha + (1. - alpha) * bg2
        results['merged'] = merged
        results['merged2'] = merged2
        return results



class BasicData(Dataset):
    def __init__(self,returnhog=False):
        self.bglist = []
        self.returnhog=returnhog
        for x in os.listdir(backgroundpath + '/train2014/'):
            self.bglist.append(backgroundpath + '/train2014/' + x)
        self.alpha = []
        self.fg = []

        for file in os.listdir(os.path.join(mattingpath , 'fg')):
            self.alpha.append(os.path.join(mattingpath, 'alpha', file))
            self.fg.append(os.path.join(mattingpath, 'fg', file))

        self.mfb = MergeFgAndBgDual()
        self.randa = RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5)
        self.jt = RandomJitterDualBG()
        self.blur=RandomBlur(False)
        self.C = Composite()

        self.crop = CropAroundG([(1024, 1024)])
        self.crop2 = CropAroundBG([(1024, 1024)])

        self.l = len(self.alpha)
        self.l2=len(self.bglist)
        self.idx = 0

    def _composite_fg2(self, fg, alpha, idx):
        alpha = alpha.astype(np.float32) / 255.
        if np.random.rand() < 0.5:
            choice = random.choice([0, 1, 2])
            if choice == 0:
                idx2 = np.random.randint(len(self.fg)) + idx
                fg2 = cv2.imread(self.fg[idx2 % len(self.fg)], cv2.IMREAD_COLOR)
                alpha2 = cv2.imread(self.alpha[idx2 % len(self.fg)], cv2.IMREAD_GRAYSCALE)

                h, w = alpha.shape
                rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
                ww=random.randint(w//4*3,w)
                hh=random.randint(h//4*3,h)
                fg2 = cv2.resize(fg2, (ww, hh), interpolation=rs)
                alpha2 = cv2.resize(alpha2, (ww, hh), interpolation=rs).astype(np.float32) / 255.
                pw1=w-ww
                pw1a=pw1//2
                pw1b=pw1-pw1a
                ph1=h-hh
                ph1a=ph1//2
                ph1b=ph1-ph1a
                fg2 = cv2.copyMakeBorder(fg2, ph1a, ph1b, pw1a, pw1b, borderType=cv2.BORDER_CONSTANT)
                alpha2 = cv2.copyMakeBorder(alpha2, ph1a, ph1b, pw1a, pw1b, borderType=cv2.BORDER_CONSTANT)
                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                if np.any(alpha_tmp < 1):
                    # composite fg with fg2
                    fg = fg.astype(np.float32) * alpha[..., None] \
                         + fg2.astype(np.float32) * (1 - alpha[..., None])
                    alpha = alpha_tmp
                    fg = np.clip(fg, 0., 255.)
            if choice == 1:
                iw = fg.shape[1]
                pw = iw // 16
                pw = random.randint(2, pw + 2) * 2
                h, w = alpha.shape
                fg = cv2.copyMakeBorder(fg, 0, 0, pw, 0, borderType=cv2.BORDER_CONSTANT)
                alpha = cv2.copyMakeBorder(alpha, 0, 0, pw, 0, borderType=cv2.BORDER_CONSTANT)

                idx2 = np.random.randint(len(self.fg)) + idx
                fg2 = cv2.imread(self.fg[idx2 % len(self.fg)], cv2.IMREAD_COLOR)
                alpha2 = cv2.imread(self.alpha[idx2 % len(self.fg)], cv2.IMREAD_GRAYSCALE)
                rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
                ww=random.randint(w//4*3,w)
                hh=random.randint(h//4*3,h)
                fg2 = cv2.resize(fg2, (ww, hh), interpolation=rs)
                alpha2 = cv2.resize(alpha2, (ww, hh), interpolation=rs).astype(np.float32) / 255.
                pw1=w-ww
                pw1a=pw1//2
                pw1b=pw1-pw1a
                ph1=h-hh
                ph1a=ph1//2
                ph1b=ph1-ph1a
                fg2 = cv2.copyMakeBorder(fg2, ph1a, ph1b, pw1a, pw+pw1b, borderType=cv2.BORDER_CONSTANT)
                alpha2 = cv2.copyMakeBorder(alpha2, ph1a, ph1b, pw1a, pw+pw1b, borderType=cv2.BORDER_CONSTANT)
                # the overlap of two 50% transparency will be 75%
                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                # if the result alpha is all-one, then we avoid composition
                if np.any(alpha_tmp < 1):
                    # composite fg with fg2
                    fg = fg.astype(np.float32) * alpha[..., None] \
                         + fg2.astype(np.float32) * (1 - alpha[..., None])
                    alpha = alpha_tmp
                    fg = np.clip(fg, 0., 255.)
                    # fg = fg.astype(np.uint8)
                fg = fg[:, pw // 2:pw // 2 + w]
                alpha = alpha[:, pw // 2:pw // 2 + w]
            if choice == 2:
                iw = fg.shape[1]
                pw = iw // 16
                pw = random.randint(2, pw) * 2
                h, w = alpha.shape
                fg = cv2.copyMakeBorder(fg, 0, 0, 0, pw, borderType=cv2.BORDER_CONSTANT)
                alpha = cv2.copyMakeBorder(alpha, 0, 0, 0, pw, borderType=cv2.BORDER_CONSTANT)
                idx2 = np.random.randint(len(self.fg)) + idx
                fg2 = cv2.imread(self.fg[idx2 % len(self.fg)], cv2.IMREAD_COLOR)
                alpha2 = cv2.imread(self.alpha[idx2 % len(self.fg)], cv2.IMREAD_GRAYSCALE)
                rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
                fg2 = cv2.resize(fg2, (w, h), interpolation=rs)
                alpha2 = cv2.resize(alpha2, (w, h), interpolation=rs).astype(np.float32) / 255.
                fg2 = cv2.copyMakeBorder(fg2, 0, 0, pw, 0, borderType=cv2.BORDER_CONSTANT)
                alpha2 = cv2.copyMakeBorder(alpha2, 0, 0, pw, 0, borderType=cv2.BORDER_CONSTANT)
                # the overlap of two 50% transparency will be 75%
                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                # if the result alpha is all-one, then we avoid composition
                if np.any(alpha_tmp < 1):
                    # composite fg with fg2
                    fg = fg.astype(np.float32) * alpha[..., None] \
                         + fg2.astype(np.float32) * (1 - alpha[..., None])
                    alpha = alpha_tmp
                    fg = np.clip(fg, 0., 255.)
                    # fg = fg.astype(np.uint8)

                fg = fg[:, pw // 2:pw // 2 + w]
                alpha = alpha[:, pw // 2:pw // 2 + w]

        alpha = np.clip(alpha, 0, 1)
        fg=fg/255.
        fg =(np.clip(fg, 0., 1.)*255).astype(np.uint8)

        alpha = (alpha * 255).astype(np.uint8)

        return fg, alpha

    def gen_trimap_(self, alpha):
        if random.random()>0.99:
            min_kernel, max_kernel = (10, 40)
        else:
            min_kernel, max_kernel = (1, 30)
        iterations = (1, 2)
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            for size in range(min_kernel, max_kernel)
        ]
        kernel_num = len(kernels)
        erode_kernel_idx = np.random.randint(kernel_num)
        dilate_kernel_idx = np.random.randint(kernel_num)
        min_iter, max_iter = iterations
        erode_iter = np.random.randint(min_iter, max_iter)
        dilate_iter = np.random.randint(min_iter, max_iter)
        eroded = cv2.erode(
            alpha, kernels[erode_kernel_idx], iterations=erode_iter)
        dilated = cv2.dilate(
            alpha, kernels[dilate_kernel_idx], iterations=dilate_iter)
        trimap = np.zeros_like(alpha)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        trimap = trimap.astype(np.float32)
        return trimap


    def __getitem__(self, i):
        ii = random.randint(0,self.l-1)
        img = self.fg[ii]
        alpha = self.alpha[ii]
        img = cv2.imread(img)
        alpha = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)
        h, w = alpha.shape
        rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,cv2.INTER_LANCZOS4])
        if max(h, w) < 1500:
            if random.random() > 0.6:
                rr=random.choice([2,1.5])
                img = cv2.resize(img, (0, 0), None, rr, rr, rs)
                alpha = cv2.resize(alpha, (0, 0), None, rr, rr, rs)
        fg, alpha = self._composite_fg2(img, alpha, ii)
        bg = self.bglist[i]
        bg = cv2.imread(bg, cv2.IMREAD_COLOR)
        i2=    random.randint(0,self.l2-1)
        bg2 = self.bglist[i2]
        bg2 = cv2.imread(bg2, cv2.IMREAD_COLOR)
        rs = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        if random.random() > 0.9:
            if random.random() > 0.5:
                bg = cv2.resize(bg, (alpha.shape[1], alpha.shape[0]), interpolation=rs)
                bg2=cv2.resize(bg2, (alpha.shape[1], alpha.shape[0]), interpolation=rs)
            else:
                ih, iw = alpha.shape[0], alpha.shape[1]
                sh = max(ih / bg.shape[0], iw / bg.shape[1])
                bg = cv2.resize(bg, (0, 0), None, sh, sh, interpolation=rs)
                bgh, bgw, _ = bg.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg = bg[oh:oh + ih, ow:ow + iw]
                sh = max(ih / bg2.shape[0], iw / bg2.shape[1])
                bg2 = cv2.resize(bg2, (0, 0), None, sh, sh, interpolation=rs)
                bgh, bgw, _ = bg2.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg2 = bg2[oh:oh + ih, ow:ow + iw]
        else:
            if random.random() > 0.5:
                bg = cv2.resize(bg, (int( alpha.shape[1]*1.1) , int( alpha.shape[0]*1.1)    ), interpolation=rs)
                ih, iw = alpha.shape[0], alpha.shape[1]
                bgh, bgw, _ = bg.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg = bg[oh:oh + ih, ow:ow + iw]
                bg2=cv2.resize(bg2, (int( alpha.shape[1]*1.1) , int( alpha.shape[0]*1.1)    ), interpolation=rs)
                bgh, bgw, _ = bg2.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg2 = bg2[oh:oh + ih, ow:ow + iw]


            else:
                ih, iw = alpha.shape[0], alpha.shape[1]
                sh = max(ih / bg.shape[0], iw / bg.shape[1])
                bg = cv2.resize(bg, (0, 0), None, sh*1.1, sh*1.1, interpolation=rs)
                bgh, bgw, _ = bg.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg = bg[oh:oh + ih, ow:ow + iw]
                sh = max(ih / bg2.shape[0], iw / bg2.shape[1])
                bg2 = cv2.resize(bg2, (0, 0), None, sh*1.1, sh*1.1, interpolation=rs)
                bgh, bgw, _ = bg2.shape
                oh = (bgh - ih) // 2
                ow = (bgw - iw) // 2
                bg2 = bg2[oh:oh + ih, ow:ow + iw]

        sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'bg2': bg2}
        sample = self.randa(sample)
        sample['Temptrimap'] = self.gen_trimap_(sample['alpha'])
        sample = self.crop(sample)
        sample=self.crop2(sample)
        sample=self.jt(sample)
        sample=self.blur(sample)

        sample['trimap'] = self.gen_trimap_(sample['bluralpha'])
        h_,w_=sample['trimap'].shape
        trimap=sample['trimap']
        tritemp = np.zeros([*sample['trimap'].shape, 2], np.float32)
        tritemp[:, :, 0] = (trimap == 0)
        tritemp[:, :, 1] = (trimap == 255)

        sample = self.mfb(sample)
        merged = sample['merged']
        merged2 = sample['merged2']
        alpha = sample['bluralpha']
        bg2 = sample['blurbg2']
        fg = sample['blurfg']
        tri = sample['trimap']
        bgt2 = bg2
        mgt = merged
        if realworldaug and (random.random()>0.5):
            mgt = cv2.cvtColor(mgt.astype(np.uint8), cv2.COLOR_BGR2RGB)
            mgt = mgt[np.newaxis, :, :, :]
            mgt = RWA(images=mgt)[0]
            mgt = mgt[:, :, ::-1]
            mgt = mgt.astype(np.float32)


        mgt2 = merged2
        fgt = fg[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        bgt2 = bgt2[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        mgt = mgt[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        mgt2 = mgt2[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        Talpha = np.array(alpha, np.float32) / 255.
        Talpha = Talpha[np.newaxis, :, :]
        Tfseg = np.zeros([3, alpha.shape[0], alpha.shape[1]], np.float32)
        Tfseg[0] = (tri == 0)
        Tfseg[1] = (tri == 128)
        Tfseg[2] = (tri == 255)
        return bgt2 ,mgt,mgt2,Tfseg,Talpha,fgt

    def __len__(self):
        return len(self.bglist)






if __name__ == '__main__':
    from torch.utils import data
    a = BasicData(False)
    print(len(a))
    trainloader = data.DataLoader(a, batch_size=1, num_workers=0, shuffle=True)
    for x, y in enumerate(trainloader):
        xxx = y
        for _ in xxx:
            if type(_) == torch.Tensor:
                print(_.shape)
            else:
                print(_)
        # print('1')
        print('skkkkkkkkkkkkkr')
        # print(n,d.shape)
