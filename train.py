import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim
import dataset
import torch.nn as nn
from torch.utils import data
import laploss
import model
import radam

if __name__ == '__main__':
    def weighted_loss(pd, gt, wl=0.9, epsilon=1e-6,tri=None):
        bs, _, h, w = pd.shape
        alpha_gt =gt.view(bs, 1, h, w)
        tri=tri.view(bs, 1, h, w)
        diff_alpha0 = (pd - alpha_gt).float()*(tri==1)
        loss_alpha2 = torch.sqrt(diff_alpha0 * diff_alpha0 + epsilon*epsilon)
        sums=((tri==1).sum(2).sum(2)+5)
        loss_alpha =   loss_alpha2.sum(2).sum(2) / sums
        loss_alpha=torch.mean(loss_alpha)
        return loss_alpha

    def get_param(model):
        nodecay = {'params': [], 'weight_decay': 0}
        decay = {'params': [], 'weight_decay': 1e-6}

        for name, param in model.named_parameters():
            if 'start_conv' in name:
                nodecay['params'].append(param)
            elif 'bias' in name:
                nodecay['params'].append(param)
            elif 'convo' in name:
                nodecay['params'].append(param)
            elif 'conv5' in name:
                nodecay['params'].append(param)
            elif 'conv4' in name:
                nodecay['params'].append(param)
            elif 'conv3' in name:
                nodecay['params'].append(param)
            else:
                decay['params'].append(param)
        return [nodecay, decay]

    laploss=laploss.lap_loss()
    mseloss = nn.MSELoss().cuda()
    absloss = nn.L1Loss().cuda()
    matmodel = model.AEMatter().cuda()
    matmodel.train()
    bs=2
    l1loss=nn.L1Loss().cuda()
    globalstep=0
    groups=get_param(matmodel)
    optim_g =radam.RAdam(groups, lr= bs*2.5*1e-5, betas=[0.5,0.999])
    sl=torch.optim.lr_scheduler.CosineAnnealingLR(optim_g,150)
    idx = 0
    h_dataset = dataset.BasicData(1024)
    h_trainloader = data.DataLoader(h_dataset, batch_size=bs,num_workers=4, shuffle=True,drop_last=True,pin_memory=True)
    temps=1.
    for epoch in range(150):
        print('Train_Start', epoch)
        id = 0
        L = 0
        L_tri=0
        L_alpha1=0
        L_alpha2=0
        L_fg=0
        L_bg=0
        L_img2=0
        L_img=0
        for _, datas in enumerate(h_trainloader):
            globalstep+=1
            bgt2 ,mgt,mgt2,Tfseg,Talpha,fgt= datas
            _,_,h,w=mgt.shape
            mgt=mgt.cuda(non_blocking=True)
            fgt=fgt.cuda(non_blocking=True)
            bgt2=bgt2.cuda(non_blocking=True)
            mgt2=mgt2.cuda(non_blocking=True)
            Talpha=Talpha.cuda(non_blocking=True)
            Tfseg=Tfseg.cuda(non_blocking=True)
            Tfseg2=torch.cat([Tfseg[:,0:1],Tfseg[:,2:3]],1)
            optim_g.zero_grad()
            lastpred=matmodel(mgt,Tfseg)
            alpha=lastpred[:,0:1]*Tfseg[:,1:2]+Tfseg[:,2:3]
            lossm = laploss(alpha, Talpha)
            loss_alpha=l1loss(alpha,Talpha)
            loss_i=weighted_loss(alpha,Talpha,tri=Tfseg[:,1:2])
            loss=loss_alpha*0.5+loss_i*0.5+lossm*0.5
            loss.backward()
            nn.utils.clip_grad_norm(matmodel.parameters(),10.)
            optim_g.step()
            id += 1
            L += loss.item()
            L_alpha1+= lossm.item()
            L_alpha2 +=loss_alpha.item()
            L_img+=loss_alpha.item()
            L_img2 += loss_alpha.item()
            if id % 100 == 0 and id > 0:
                print('Epoch', epoch, 'Total_Los', L / 100.,'Alpha1Loss',L_alpha1/100,'Alpha2Loss',L_alpha2/100)
                L = 0
                id = 0
                L_tri = 0
                L_alpha1 = 0
                L_alpha2 = 0
                L_fg = 0
                L_bg = 0
                L_img2 = 0
                L_img = 0
        checkpoint = {"model": matmodel.state_dict(),
                      }
        torch.save(checkpoint, './ckpt/' + str(epoch//1) + 'aem.ckpt')
        sl.step()


