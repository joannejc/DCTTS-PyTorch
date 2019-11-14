import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from module import NonCausalConv1d, CausalConv1d, Highway

# Tex2Mel Network
class TextEncoder(nn.Module):
    """Text encoder
    
    Note Text encoder uses Non-causal Conv1d
    
    """
    def __init__(self, d, e, vocab):
        """
        Inputs:
            d: hidden dimension
            e: embedding dimension
            vocab: vocabulary size

        Note call convolutiosn in text encoder are non-causal
            
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, e)
        self.l1 = nn.Sequential(NonCausalConv1d(e, 2*d, 1, 1),
                                nn.ReLU(),
                                NonCausalConv1d(2*d, 2*d, 1, 1),
                                nn.ReLU(),
                                
                                Highway(2*d, 3, 1),
                                Highway(2*d, 3, 3),
                                Highway(2*d, 3, 9),
                                Highway(2*d, 3, 27), 
                                Highway(2*d, 3, 1),
                                Highway(2*d, 3, 3),
                                Highway(2*d, 3, 9),
                                Highway(2*d, 3, 27),
                                
                                Highway(2*d, 3, 1),
                                Highway(2*d, 3, 1),
                                
                                Highway(2*d, 1, 1),
                                Highway(2*d, 1, 1))

    def forward(self, text):
        """
        Inputs:
            text (batch_size x N): batch of character sequence, where N is sequence length
            
        """
        text = self.embed(text)
        # switch axis, since text has shape batch_size x seq_len x embedding_dim, switch last two axis
        return self.l1(text.transpose(2,1))


# audio encoder
class AudioEncoder(nn.Module):
    """Audio encoder that encodes the past generated sequences.
    
    Note Audio Encoder uses Causal Conv1d
    
    """
    def __init__(self, d, F):
        """
        Inputs:
            d: hidden size
            F: mel spectrogram size

        Note all convolutions in Audio Encoder have stride 1 and are causal. (as mentioned in paper)
            
        """
        super().__init__()
        self.l1 = nn.Sequential(CausalConv1d(F, d, 1, 1),
                                nn.ReLU(),
                                CausalConv1d(d, d, 1, 1),
                                nn.ReLU(),
                                CausalConv1d(d, d, 1, 1),
                                
                                Highway(d, 3, 1, True),
                                Highway(d, 3, 3, True),
                                Highway(d, 3, 9, True),
                                Highway(d, 3, 27, True),
                                Highway(d, 3, 1, True),
                                Highway(d, 3, 3, True),
                                Highway(d, 3, 9, True),
                                Highway(d, 3, 27, True),
                                
                                Highway(d, 3, 3, True),
                                Highway(d, 3, 3, True))
        
    def forward(self, audio):
        """audio has shape batch_size x F x T, where T is the output audio length
        
        """
        return self.l1(audio)


class Decoder(nn.Module):
    """Audio Decoder
    
    """
    def __init__(self, d, F):
        """
        """
        super().__init__()
        self.l1 = nn.Sequential(CausalConv1d(2*d, d, 1, 1),
                                
                                Highway(d, 3, 1, True),
                                Highway(d, 3, 3, True),
                                Highway(d, 3, 9, True),
                                Highway(d, 3, 27, True),
                                
                                Highway(d, 3, 1, True),
                                Highway(d, 3, 1, True),
                                
                                CausalConv1d(d, d, 1, 1),
                                nn.ReLU(),
                                CausalConv1d(d, d, 1, 1),
                                nn.ReLU(),
                                CausalConv1d(d, d, 1, 1),
                                nn.ReLU(),
                                
                                CausalConv1d(d, F, 1, 1))
        
    def forward(self, x):
        """x is of shape batch_size x 2*d x T
        
        """
        return self.l1(x)

class Text2Mel(nn.Module):
    """Actual text2mel network

    """
    def __init__(self, hparams, device):
        self.hp = hparams
        self.device = device
        super().__init__()
        self.t_encoder = TextEncoder(self.hp.d, self.hp.e, self.hp.vocab)
        self.a_encoder = AudioEncoder(self.hp.d, self.hp.F)
        self.decoder = Decoder(self.hp.d, self.hp.F)
        self.to(device)

    def forward(self, text, prev_mel):
        """Takes a forward step given a batch of text and previous mel output

        Inputs:
            text(batch_size x N): text can also be a tuple where the first item is tensor of shape batch_size x N, and the second a list of sequence length
                                  Note that batched text input should be arranged in order from longest to shortest from top to bottom to utilize Pytorch's packed sequence

            prev_mel(batch_size x F x T):

        Returns:
            y (batch_size x F x T+1):
            logit (batch_size x F T+1):
            a (batch_size x N x T):

        """
        # forward pass through text embedding and get k and v
        kv = self.t_encoder(text)
        k = kv[:,:self.hp.d,:]
        v = kv[:,self.hp.d:,:]
        # forward pass through audio encoding and get Q
        q = self.a_encoder(prev_mel)
        
        # compute attention
        a = (k.transpose(2,1)).matmul(q)/np.sqrt(self.hp.d)
        a = F.softmax(a, dim=1)
        r = v.matmul(a)
        
        # create R' and forward pass through decoder
        # note that the decoder does not have sigmoid transform at the end, so we are actually getting 
        # ylogit
        rprime = torch.cat((r, q), dim=1)
        ylogit = self.decoder(rprime)
        y = F.sigmoid(ylogit)
        return y, ylogit, a


    def generate(self, text, prev_mel):
        """Generate, no grad.

        Note text should be a single text, i.e of batch_size of 1


        """
        # forward pass through text embedding and get k and v
        kv = self.t_encoder(text)
        k = kv[:,:self.hp.d,:]
        v = kv[:,self.hp.d:,:]
        # forward pass through audio encoding and get Q
        q = self.a_encoder(prev_mel)
        
        # compute attention and use forcible incremental attention (section 4.2)
        a = (k.transpose(2,1)).matmul(q)/np.sqrt(self.hp.d)
        a = F.softmax(a, dim=1)
        """
        # get argmax
        argmax = a[0].argmax(dim=0) # argmax on the N dimension
        # forcibly incremental attention
        preva = -1
        for i in range(a.shape[-1]):
            if argmax[i] < preva -1 or preva + 3 < argmax[i]:
                # force the ith column to be zero
                a[:,:,i] = 0
                # find correct position
                position = min(a.shape[1]-1, preva + 1)
                a[:,position,i] = 1.0
            # update preva
            preva = a[0,:,i].argmax()"""

        # finish computing y and a
        r = r = v.matmul(a)

        rprime = torch.cat((r, q), dim=1)
        ylogit = self.decoder(rprime)
        y = F.sigmoid(ylogit)
        return y, ylogit, a

    def batch_train(self, text, mel_in, mel_target):
        """
        """


    def batch_generate(self, text, mel_target):
        """Given a text embedding and mel_target, generate until we have enough.

        Note this is for reference only.

        """
        mel_in = torch.zeros(1,80,1).to(self.device)
        for i in range(mel_target.shape[2]):
            # forward pass
            y, ylogit, a = self.generate(text, mel_in)
            # create new mel_in
            mel_in = torch.cat((mel_in, y[:,:,-1].view(1,80,1)), dim=-1)

        # compute L1 loss to target and return target and prediction
        loss = F.l1_loss(y, mel_target)
        return loss, y, a


    def compute_loss(self, text, mel_in, mel_target):
        """compute loss given target

        """
        y, ylogit , a = self.forward(text, mel_in)

        # compute loss
        l1_loss = F.mse_loss(y, mel_target)
        bin_loss = F.binary_cross_entropy_with_logits(ylogit, mel_target)
        # now attention loss
        N = text.shape[-1]
        T = mel_in.shape[-1]
        def w_fun(n, t):
            return 1 - np.exp(-((n/(N-1) - t/(T-1))**2) / (2 * self.hp.g**2))
        w = np.fromfunction(w_fun, (a.shape[1], T), dtype='f')
        w = torch.from_numpy(w).to(self.device)
        loss_att = (w * a[:, :, :T]).mean()
        loss = l1_loss + bin_loss + loss_att
        return loss, y, a

    def count_parameters(self):
        """Count total parameters in millions
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6


        

# SSRN network
class SSRN(nn.Module):
    """SSRN network, maps from coarse-mel spectrogram to fine mel-spectrogram via super-resolution

    """
    def __init__(self, hparams, device):
        """Initialize the network

        """
        self.hp = hparams
        self.device = device
        super().__init__()
        F = self.hp.F
        Fprime = self.hp.Fprime
        c = self.hp.c
        self.layers = nn.Sequential(NonCausalConv1d(F, c, 1, 1),
                                Highway(c,3,1),
                                Highway(c,3,3),
                                
                                nn.ConvTranspose1d(c,c,2,dilation=1,stride=2),
                                Highway(c,3,1),
                                Highway(c,3,3),
                                nn.ConvTranspose1d(c,c,2,dilation=1,stride=2),
                                Highway(c,3,1),
                                Highway(c,3,3),
                                
                                NonCausalConv1d(c,2*c,1,1),
                                
                                Highway(2*c,3,1),
                                Highway(2*c,3,1),
                                
                                NonCausalConv1d(2*c,Fprime,1,1),
                                
                                NonCausalConv1d(Fprime,Fprime,1,1),
                                nn.ReLU(),
                                NonCausalConv1d(Fprime,Fprime,1,1),
                                nn.ReLU(),
                                
                                NonCausalConv1d(Fprime,Fprime,1,1))
        # put onto device
        self.to(device)

    def forward(self, mel_coarse):
        """given a corase mel spectrogram, forward pass through layer and return the predicted fine mel spectrogram.

        Input:
            mel_coarse(batch_size x F x T):

        Return:
            y (batch_size x Fprim x 4*T): y output after applying sigmoid
            logit(batch_size x Fprime x 4*T): logit before applying sigmoid

        """
        logit = self.layers(mel_coarse)
        y = F.sigmoid(logit)
        return y, logit

            
    def compute_loss(self, mel_coarse, mel_fine):
        """Given a batch of coarse mel and target mel_fine, forward pass through the network compute the loss.
        INput:
            mel_coarse(batch_size x F x T):
            mel_fine (batch_size x Fprim x 4*T):
            mellens (list of size batch_size):

        """
        y, logit = self.forward(mel_coarse)
        # compute loss
        l1_loss = F.l1_loss(y, mel_fine)
        bin_loss = F.binary_cross_entropy_with_logits(logit, mel_fine)
        loss = l1_loss + bin_loss
        return loss


    def compute_batch_loss(self, batch_data):
        """given a list of batch data, iterate through and compute loss

        """
        loss = 0
        for data in batch_data:
            x, y = data
            x = x.view(-1,x.shape[0],x.shape[1])
            y = y.view(-1,y.shape[0], y.shape[1])
            loss += self.compute_loss(x.to(self.device), y.to(self.device))
        
        return loss



    def count_parameters(self):
        """Count total parameters in millions
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6


def no_test_SSRN():
    from hparams import Hparams
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SSRN(hp, device)
    print("SSRN model has {} million parameters".format(model.count_parameters()))
    out = model.forward(torch.rand(5,hp.F,100).to(device))
    for i in out:
        print(i.shape)


def test_text2mel():
    from hparams import Hparams
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Text2Mel(hp, device)
    print("SSRN model has {} million parameters".format(model.count_parameters()))
    text = torch.rand(1,40).long()
    mel_in = torch.rand(1,80,120)
    mel_target = torch.rand(1,80,120)
    out = model.compute_loss(text.to(device), mel_in.to(device), mel_target.to(device))
    loss, y, a = out
    print(loss)
    print(y.shape, a.shape)
    out = model.batch_generate(text.to(device), mel_target.to(device))
    loss, y, a = out
    print(loss, y.shape, a.shape)
    for i in range(a.shape[2]):
        print(i, a[0,:,i].argmax())



