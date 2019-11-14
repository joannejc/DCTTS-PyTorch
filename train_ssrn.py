import torch
import torch.nn.functional as F
import numpy as np

from networks import SSRN
from hparams import Hparams
from tqdm import tqdm
from data import DataLoader, LJDataset, ssrn_collate_fn

def train():
    print("training begins...")
    # training and validation relative directories
    train_directory = "../../ETTT/Pytorch-DCTTS/LJSpeech_data/"
    val_directory = "../../ETTT/Pytorch-DCTTS/LJSpeech_val/"
    t_data = LJDataset(train_directory)
    v_data = LJDataset(val_directory)
    train_len = len(t_data.bases)
    val_len = len(v_data.bases)

    # training parameters
    batch_size = 40
    epochs = 500
    save_every = 5
    learning_rate = 1e-4
    max_grad_norm = 1.0

    # create model and optim
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SSRN(hp, device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # main training loop
    for ep in tqdm(range(epochs)):
        total_loss = 0 # epoch loss
        t_loader = DataLoader(t_data, batch_size=batch_size, shuffle=True, drop_last = False, collate_fn=ssrn_collate_fn)
        for data in tqdm(t_loader):
            # initialize batch_loss
            batch_loss = model.compute_batch_loss(data)
            # batch update
            optim.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(max_norm=max_grad_norm, parameters=model.parameters())
            optim.step()
            total_loss += batch_loss.detach().cpu().numpy()
        # one epoch complete, add to total loss and print
        print("epoch {}, total loss:{}, average total loss:{}, validating now...".format(ep, float(total_loss), float(total_loss)/train_len))
        # if time to save, we save model
        if ep % save_every == 0:
            torch.save(model.state_dict(),"save_stuff/checkpoint/epoch_"+str(ep)+"_ssrn_model.pt")
        
        # Validation phase
        with torch.no_grad():
            total_loss = 0
            v_loader = DataLoader(v_data, batch_size=batch_size//10, shuffle=True, drop_last=False, collate_fn=ssrn_collate_fn)
            for data in tqdm(v_loader):
                loss = model.compute_batch_loss(data)
                total_loss += loss.detach().cpu().numpy()
            # printing
            print("validation loss:{}, average validation loss:{}".format(float(total_loss), float(total_loss)/val_len))
            for dat in data:
                x, y = dat
            # predict
            predict, _ = model.forward((x.view(1,80,-1)).to(device))
            np.save("save_stuff/mel_pred/epoch_"+str(ep)+"_mel_pred.npy", predict.detach().cpu().numpy())
            np.save("save_stuff/mel_pred/epoch_"+str(ep)+"_ground_truth.npy", y)


if __name__=="__main__":
    train()



                

