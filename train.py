import os
import torch
import torch.nn as nn
from tqdm import tqdm

from hparams import hp
from dataset import get_loader
from model import get_model
from synthesize import gen_testset


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    # dataset
    print("Loading dataset...")
    train_loader, test_loader = get_loader()

    # model, optimizer
    print("Loading model...")
    model, optimizer = get_model(hp, device, train=True)

    # loss
    Loss = nn.CrossEntropyLoss().to(device)

    # output
    ckpt_dir = hp.ckpt_dir
    log_dir = hp.log_dir
    log_path = os.path.join(log_dir, "log.txt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # training
    step = hp.restore_step + 1
    epoch = 1

    grad_clip_thresh = hp.clip_grad_norm

    total_step = hp.total_step
    synth_step = hp.synth_step
    log_step = hp.log_step
    save_step = hp.save_step

    print("Training...")
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = hp.restore_step
    outer_bar.update()
    
    while True:
        for x, y, mel in tqdm(train_loader, desc="Epoch {}".format(epoch), position=1):
            x, y, mel = x.to(device), y.to(device), mel.to(device)

            # Forward
            y_hat = model(x, mel)
            
            # Cal loss
            y_hat = y_hat.view(-1, y_hat.size(-1))
            y = y.view(-1)
            loss = Loss(y_hat, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            optimizer.step()

            # Log
            if step % log_step == 0:
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Loss: {:.4f}.".format(loss)
                with open(log_path, "a") as f:
                    f.write(message1 + message2 + "\n")
                outer_bar.write(message1 + message2)

            # Synth
            if step % synth_step == 0:
                model.eval()
                with torch.no_grad():
                    gen_testset(model, test_loader, hp, device, step)
                model.train()

            # Save
            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        ckpt_dir,
                        "{}.pth.tar".format(step),
                    ),
                )
            
            # Quit
            if step == total_step:
                quit()

            step += 1
            outer_bar.update(1)
                
        epoch += 1
                    

if __name__ == '__main__':

    main() 
