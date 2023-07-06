import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
import model

ptrimap='./trimaps/'
pimgs='./img/'
p_outs='./alphas/'

os.makedirs(p_outs,exist_ok=True)

if __name__ == '__main__':
    matmodel = model.AEMatter()
    matmodel.load_state_dict(torch.load('./model.ckpt',map_location='cpu')['model'])
    matmodel=matmodel.cuda()
    matmodel.eval()
    for idx,file in enumerate(os.listdir(ptrimap)) :
        print(idx)
        rawimg=pimgs+file
        trimap=ptrimap+file
        rawimg=cv2.imread(rawimg)
        trimap=cv2.imread(trimap,cv2.IMREAD_GRAYSCALE)
        trimap_nonp=trimap.copy()
        h,w,c=rawimg.shape
        nonph,nonpw,_=rawimg.shape
        newh= (((h-1)//32)+1)*32
        neww= (((w-1)//32)+1)*32
        padh=newh-h
        padh1=int(padh/2)
        padh2=padh-padh1
        padw=neww-w
        padw1=int(padw/2)
        padw2=padw-padw1
        rawimg_pad=cv2.copyMakeBorder(rawimg,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
        trimap_pad=cv2.copyMakeBorder(trimap,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
        h_pad,w_pad,_=rawimg_pad.shape
        tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
        tritemp[:, :, 0] = (trimap_pad == 0)
        tritemp[:, :, 1] = (trimap_pad == 128)
        tritemp[:, :, 2] = (trimap_pad == 255)
        tritempimgs=np.transpose(tritemp,(2,0,1))
        tritempimgs=tritempimgs[np.newaxis,:,:,:]
        img=np.transpose(rawimg_pad,(2,0,1))[np.newaxis,::-1,:,:]
        img=np.array(img,np.float32)
        img=img/255.
        img=torch.from_numpy(img).cuda()
        tritempimgs=torch.from_numpy(tritempimgs).cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                pred=matmodel(img,tritempimgs)
                pred=pred.detach().cpu().numpy()[0]
                pred=pred[:,padh1:padh1+h,padw1:padw1+w]
                preda=pred[0:1,]*255
                preda=np.transpose(preda,(1,2,0))
                preda=preda*(trimap_nonp[:,:,None]==128)+(trimap_nonp[:,:,None]==255)*255
        preda=np.array(preda,np.uint8)
        cv2.imwrite(p_outs+file,preda)
