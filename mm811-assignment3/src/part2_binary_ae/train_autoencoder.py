import torch
import torch.nn.functional as F


def compute_psnr_torch(imgs, refs, eps=1e-8):
    mse = F.mse_loss(imgs, refs, reduction="none").mean(dim=[1, 2, 3])
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse + eps))
    return psnr


def compute_psnr_sigs(imgs, refs):
    imgs_norm = imgs + 1.0
    refs_norm = refs + 1.0
    imgs_norm *= 0.5
    refs_norm *= 0.5
    imgs_norm *= 255.0
    refs_norm *= 255.0
    psnr_vals = compute_psnr_torch(imgs_norm, refs_norm)
    return psnr_vals


def train_epoch(
    model,
    dataloader,
    device,
    optimizer,
    *,
    use_l1=True,
    reg_lambda=1e-3,
    push_margin=1.0,
):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        imgs = imgs * 2.0 - 1.0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        recon, binary_latent, latent_logits = model(imgs)

        # Reconstruction loss
        if use_l1:
            rec_loss = F.l1_loss(recon, imgs)
        else:
            rec_loss = F.mse_loss(recon, imgs)

        commit = F.relu(push_margin - latent_logits.abs()).mean()
        loss = rec_loss + reg_lambda * commit

        loss.backward()
        optimizer.step()

        total_loss += rec_loss.item()
        total_psnr += compute_psnr_sigs(recon.detach(), imgs).mean().item()

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    return avg_loss, avg_psnr


def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            imgs = imgs * 2.0 - 1.0
            recon, _, _ = model(imgs)
            total_psnr += compute_psnr_sigs(recon, imgs).mean().item()

    avg_psnr = total_psnr / len(dataloader)
    return avg_psnr
