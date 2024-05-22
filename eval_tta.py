import cv2
import numpy as np
import torch
import model
import os

p1 = './tri/'
p2 = './img/'
p3a = './'

os.makedirs(p3a, exist_ok=True)

if __name__ == '__main__':

    matmodel = model.AEMatter()
    matmodel.load_state_dict(torch.load('*.ckpt', map_location='cpu')['model'])
    matmodel = matmodel.cpu()
    matmodel.eval()

    for idx, file in enumerate(os.listdir(p1)):
        rawimg = p2 + file[:-4] + '.jpg'
        trimap = p1 + file
        rawimgf = cv2.imread(rawimg)
        trimapf = cv2.imread(trimap, cv2.IMREAD_GRAYSCALE)
        alphaall = np.array(trimapf * 0, dtype=np.float32)

        for rotate1, rotate2 in zip([-1, 2, 1, 0], [-1, 0, 1, 2]):
            for flip in range(0, 2):
                rawimg = rawimgf.copy()
                trimap = trimapf.copy()
                if rotate1 != -1:
                    rawimg = cv2.rotate(rawimgf, rotate1)
                    trimap = cv2.rotate(trimapf, rotate1)
                if flip == 1:
                    rawimg = cv2.flip(rawimg, 1)
                    trimap = cv2.flip(trimap, 1)

                trimap_nonp = trimap.copy()
                h, w, c = rawimg.shape
                nonph, nonpw, _ = rawimg.shape
                newh = (((h - 1) // 32) + 1) * 32
                neww = (((w - 1) // 32) + 1) * 32
                padh = newh - h
                padh1 = int(padh / 2)
                padh2 = padh - padh1
                padw = neww - w
                padw1 = int(padw / 2)
                padw2 = padw - padw1
                rawimg_pad = cv2.copyMakeBorder(rawimg, padh1, padh2, padw1, padw2, cv2.BORDER_REFLECT)
                trimap_pad = cv2.copyMakeBorder(trimap, padh1, padh2, padw1, padw2, cv2.BORDER_REFLECT)
                h_pad, w_pad, _ = rawimg_pad.shape
                tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
                tritemp[:, :, 0] = (trimap_pad == 0)
                tritemp[:, :, 1] = (trimap_pad == 128)
                tritemp[:, :, 2] = (trimap_pad == 255)
                tritemp2 = np.transpose(tritemp, (2, 0, 1))
                tritemp2 = tritemp2[np.newaxis, :, :, :]
                img = np.transpose(rawimg_pad, (2, 0, 1))[np.newaxis, ::-1, :, :]
                img = np.array(img, np.float32)
                img = img / 255.
                img = torch.from_numpy(img).cpu()
                tritemp2 = torch.from_numpy(tritemp2).cpu()

                with torch.no_grad():
                    pred = matmodel(img, tritemp2)
                    pred = pred.detach().cpu().numpy()[0]
                    pred = pred[:, padh1:padh1 + h, padw1:padw1 + w]
                    preda = pred[0:1, ] * 255
                    preda = np.transpose(preda, (1, 2, 0))
                    preda = preda * (trimap_nonp[:, :, None] == 128) + (trimap_nonp[:, :, None] == 255) * 255
                preda = np.array(preda[:, :, 0], np.uint8)
                if flip == 1:
                    preda = cv2.flip(preda, 1)
                if rotate2 != -1:
                    preda = cv2.rotate(preda, rotate2)
                alphaall = alphaall * 1. + preda * 1.
        alphaall = alphaall / 8.
        alphaall = np.array(alphaall, np.uint8)
        cv2.imwrite(p3a + file, alphaall)