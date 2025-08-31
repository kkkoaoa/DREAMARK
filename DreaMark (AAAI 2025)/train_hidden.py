import torch
import torch.nn.functional as F
from torch import optim
import argparse
import time
import tqdm

from nerf.utils import *
from hidden import *

def _message_loss(fts, targets, m=1, loss_type='bce'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')
    
def _image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

def log(*args, **kwargs):
    print(*args, file=log_ptr)
    log_ptr.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit_length", default=32, help="bit length")
    parser.add_argument('--workspace', type=str, default='exp-hidden/')
    parser.add_argument("--channel", default=64)
    parser.add_argument("--decoder_num_block", default=8)
    parser.add_argument("--encoder_num_block", default=4)
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--seed', default=42)
    parser.add_argument('--init_ckpt', type=str, default='', help="ckpt to init dmtet")
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--epoch", default=100)
    parser.add_argument('--img_size', default=256)
    parser.add_argument("--test", action="store_true")

    opt = parser.parse_args()
    
    opt.workspace += str(time.strftime('%Y-%m-%d', time.localtime()))+"-"+str(opt.bit_length).replace(" ", "-")
    
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = HiddenEncoder(num_blocks=opt.encoder_num_block, num_bits=opt.bit_length, channels=opt.channel)
    decoder = HiddenDecoder(num_blocks=opt.decoder_num_block, num_bits=opt.bit_length, channels=opt.channel)
    augmentation = HiddenAug(img_size=opt.img_size, p_crop=1.0, p_blur=0, p_jpeg=1, p_rot=0, p_color_jitter=0, p_res=1.0)

    model = EncoderDecoder(encoder=encoder,
                           decoder=decoder,
                           attenuation=None,
                           augmentation=augmentation,
                           num_bits=opt.bit_length).to(device)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = opt.lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    train_loader = COCODataLoader(batch_size=16, img_size=opt.img_size, split="train", root="/data/Public/Dataset/COCO/COCO2017/train")
    valid_loader = COCODataLoader(batch_size=16, img_size=opt.img_size, split="val", root="/data/Public/Dataset/COCO/COCO2017/val/")

    if opt.init_ckpt != '':
        checkpoint = torch.load(opt.init_ckpt, map_location=device)
        new_ckpt = {}
        for k, v in checkpoint['encoder_decoder'].items():
            if k.startswith('module.'):
                new_ckpt[k[7:]] = v
            else:
                new_ckpt[k] = v
        model.load_state_dict(new_ckpt)

    os.makedirs(opt.workspace, exist_ok=True)
    log_path = os.path.join(opt.workspace, f"log_df.txt")
    log_ptr = open(log_path, "a+")
    ckpt_path = os.path.join(opt.workspace, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)

    if opt.test:
        progress = tqdm.tqdm(train_loader)
        for imgs in progress:
            B = imgs.shape[0]
            imgs = imgs.to(device)
            msgs = torch.FloatTensor(np.random.choice([0, 1], (B, opt.bit_length))).to(device)

            fts, imgs_w, imgs_aug = model(imgs, msgs, eval_mode=True)
            acc = (fts.round().clip(0,1)==msgs).float().mean().item()
            
            progress.set_description_str(
                f"accuracy: {acc}"
            )
    else:
        best = torch.inf
        for ep in range(opt.epoch):
            log(f"[INFO] Epoch={ep}, lr={optimizer.param_groups[0]['lr']}")
            # train
            model.train()
            progress = tqdm.tqdm(train_loader)
            loss_value = 0
            for idx, imgs in enumerate(progress):
                optimizer.zero_grad()

                B = imgs.shape[0]
                imgs = imgs.to(device)
                msgs = torch.FloatTensor(np.random.choice([0, 1], (B, opt.bit_length))).to(device)
                msgs = 2 * msgs - 1

                fts, imgs_w, imgs_aug = model(imgs, msgs)

                loss_w = _message_loss(fts, msgs, loss_type='bce')
                
                loss_w.backward()
                optimizer.step()
                
                ori_msgs = torch.sign(msgs) > 0
                decoded_msgs = torch.sign(fts) > 0
                diff = (~torch.logical_xor(ori_msgs, decoded_msgs))
                acc = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item()
                loss_value += loss_w.item()

                progress.set_description_str(
                    f"message_loss: {loss_value/(idx+1)}, accuracy: {acc}"
                )
            # eval
            model.eval()
            progress = tqdm.tqdm(valid_loader)
            loss_value = 0
            acc_value = 0
            for idx, imgs in enumerate(progress):
                B = imgs.shape[0]
                imgs = imgs.to(device)
                msgs = torch.FloatTensor(np.random.choice([0, 1], (B, opt.bit_length))).to(device)
                msgs = 2 * msgs - 1
                
                fts, imgs_w, imgs_aug = model(imgs, msgs, eval_mode=True)

                loss_w = _message_loss(fts, msgs, loss_type='bce')
                
                ori_msgs = torch.sign(msgs) > 0
                decoded_msgs = torch.sign(fts) > 0
                diff = (~torch.logical_xor(ori_msgs, decoded_msgs))
                acc = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item()

                loss_value += loss_w.item()
                acc_value += acc

                progress.set_description_str(
                    f"message_loss: {loss_value/(idx+1)}, accuracy: {acc}"
                )
            log(f"[INFO] Epoch={ep}, message_loss={loss_value/(idx+1)}, accuracy={acc_value/(idx+1)}")
            scheduler.step()
            if best > loss_value/(idx+1):
                best = loss_value/(idx+1)
                torch.save({
                    "encoder_decoder": model.state_dict()
                }, os.path.join(ckpt_path, f"best.pth"))