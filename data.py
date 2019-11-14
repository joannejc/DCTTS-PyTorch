import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from hparams import Hparams

class LJDataset(Dataset):
    """LJDataset Class.
    
    """
    def __init__(self, directory):
        self.directory = directory
        hp = Hparams()
        self.F = hp.F
        self.Fprime = hp.Fprime
        allfiles = os.listdir(directory)
        self.bases= []
        for f in allfiles:
            if f[-8:] == '_mel.npy':
                self.bases.append(f[:-8])
                
    def __len__(self):
        return len(self.bases)
    
    def __getitem__(self, index):
        mel = np.load(os.path.join(self.directory, self.bases[index] + '_mel.npy')).astype('f')
        text = np.load(os.path.join(self.directory, self.bases[index] + '_txt.npy')).astype('f')
        fft = np.load(os.path.join(self.directory, self.bases[index] + '_fft.npy')).astype('f')
        return text, mel, fft


def ssrn_collate_fn(data):
    """collate function for training SSRN

    """
    output = []
    batch_size = len(data)
    
    for dat in data:
        _, mel, fft = dat
        mel_coarse = torch.from_numpy(mel)
        mel_target = torch.from_numpy(fft)
        output.append([mel_coarse, mel_target])
    
    return output

def text2mel_collate_fn(data):
    """collate function for training text2mel

    """
    outputs = []
    batch_size = len(data)

    for dat in data:
        text, mel, _ = dat
        text = (torch.from_numpy(text).long()).unsqueeze(0)
        mel = (torch.from_numpy(mel)).unsqueeze(0)
        mel_in = torch.cat((torch.zeros(1, 80,1), mel[:,:,:-1]),dim=-1)
        outputs.append([text, mel_in, mel])


        return outputs



def test_ssrn():
    val_directory = "../../ETTT/Pytorch-DCTTS/LJSpeech_data/"
    dataset = LJDataset(val_directory)
    loader = DataLoader(dataset, batch_size=300, shuffle=False, collate_fn=text2mel_collate_fn)
    print(len(loader))

