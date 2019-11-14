import torch
import torch.nn.functional as F
import numpy as np

from networks import Text2Mel
from hparams import Hparams
from tqdm import tqdm
from data import DataLoader, LJDataset, text2mel_collate_fn

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
    batch_size = 350
    epochs = 500
    save_every = 5
    learning_rate = 1e-4
    max_grad_norm = 1.0

    # create model and optim
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Text2Mel(hp, device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # main training loop
    for ep in tqdm(range(epochs)):
        total_loss = 0 # epoch loss
        t_loader = DataLoader(t_data, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=text2mel_collate_fn)
        for dat in tqdm(t_loader):
            # batch update
            batch_loss = 0
            for data in dat:
                text, mel_in, mel_target = data
                loss, _, _ = model.compute_loss(text.to(device), mel_in.to(device), mel_target.to(device))
                batch_loss += loss
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
            torch.save(model.state_dict(),"save_stuff/text2mel/checkpoint/epoch_"+str(ep)+"_text2mel_model.pt")
        
        # Validation phase
        with torch.no_grad():
            total_loss = 0
            v_loader = DataLoader(v_data, batch_size=batch_size//10, shuffle=True, drop_last=False, collate_fn=text2mel_collate_fn)
            for dat in tqdm(v_loader):
                for data in dat:
                    text, mel_in, mel_target = data
                    loss, y, a = model.batch_generate(text.to(device), mel_target.to(device))
                    total_loss += loss.detach().cpu().numpy()
            # printing
            print("validation loss:{}, average validation loss:{}\n".format(float(total_loss), float(total_loss)/val_len))
            # saving prediciton and attention
            np.save("save_stuff/text2mel/attention/epoch_"+str(ep)+"_attention.npy", a)
            np.save("save_stuff/text2mel/pred/epoch_"+str(ep)+"_y_prediction.npy", y)

if __name__=="__main__":
    train()
        



    
