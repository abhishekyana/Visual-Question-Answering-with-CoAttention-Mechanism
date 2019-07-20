import torch.nn as nn
import torch

class OneGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(1),padding=(0))
    def forward(self,X,form='ncl'):
        if form=='nlc':
            X = X.permute(0,2,1)
        elif form=='ncl':
            pass
        else:
            print("IVALID FORMAT nlc or ncl is accepted")
        N,v,n = X.shape
        out = self.conv(X)[:,:,:n]
        if form=='nlc':
            out = out.permute(0,2,1)
        return out 

class BiGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(2),padding=(1))
    def forward(self,X):
        N,v,n = X.shape
        return self.conv(X)[:,:,:n]

class TriGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(3),padding=(1))
    def forward(self,X):
        N,v,n = X.shape
        return self.conv(X)[:,:,:n]
    
class PhraseLevel(nn.Module):
    def __init__(self,nE=300,nH=128,p=0.5):
        super().__init__()
        self.onegram = OneGram(nE,nH)
        self.bigram = BiGram(nE,nH)
        self.trigram = TriGram(nE,nH)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=p)
    def forward(self,X,form='ncl'):
        if form=='nlc':
            X = X.permute(0,2,1)
        elif form=='ncl':
            pass
        else:
            print("IVALID FORMAT nlc or ncl is accepted")
        oe = self.onegram(X)
        be = self.bigram(X)
        te = self.trigram(X)
        out = torch.stack([oe,be,te])
        out = torch.max(out,dim=0)[0]
        out = self.tanh(out)
        out = self.dropout(out)
        if form=='nlc':
            out = out.permute(0,2,1)
        return out

class QuestionLevel(nn.Module):
    def __init__(self, nE=512, nH=512, nL=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(nE,nH,nL,batch_first=True)
    def forward(self,X,*kwargs):
        out,(h,c) = self.lstm(X,*kwargs)
        return out

class AttentionNet(nn.Module):
    def __init__(self, nH):
        super().__init__()
        self.Wb = torch.nn.Parameter(torch.autograd.Variable(torch.randn(nH,nH)))
        self.tanh = torch.nn.Tanh()
    def forward(self,V,Q):
        # print(V.shape,Q.shape,self.Wb.shape)
        out = Q @ (self.Wb @ V)
        C = self.tanh(out)
        return C
    
class ParallelAttention(nn.Module):
    def __init__(self, nH=512):
        super().__init__()
        self.attention = AttentionNet(nH)
        self.tanh = torch.nn.Tanh()
        self.Wq  = torch.nn.Parameter(torch.autograd.Variable(torch.randn(nH,nH)))
        self.Wv  = torch.nn.Parameter(torch.autograd.Variable(torch.randn(nH,nH)))
        self.whv = torch.nn.Parameter(torch.autograd.Variable(torch.randn(1,nH)))
        self.whq = torch.nn.Parameter(torch.autograd.Variable(torch.randn(1,nH)))
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self,V,Q):
        C = self.attention(V,Q)
        WqQ = (Q @ self.Wq).permute(0,2,1)
        WvV = self.Wv @ V
#         print("Inputs: ",V.shape,Q.shape)
#         print("Weights:",WvV.shape,WqQ.shape,C.shape)
        Hv = self.tanh(WvV + WqQ @ C)
        Hq = self.tanh(WvV @ C.permute(0,2,1) + WqQ)
#         print("H s:    ",Hv.shape, Hq.shape)
        av = self.softmax(self.whv @ Hv)
        aq = self.softmax(self.whq @ Hq)
#         print("Attented",av.shape,aq.shape)
        Vhat = torch.squeeze((V*av).sum(dim=2))
        Qhat = torch.squeeze((Q.permute(0,2,1)*aq).sum(dim=2))
        # print("Hats:   ",Vhat.shape,Qhat.shape)
        return Vhat,Qhat    

class CoattentionNet(nn.Module):
    def __init__(self,nE=300, nH=512, nO=1001):
        super().__init__()
        self.parallelattention = ParallelAttention(nH)
        self.WL = OneGram(nE, nH)
        self.PL = PhraseLevel(nE, nH)
        self.QL = QuestionLevel(nE, nH)
        self.tanh = torch.nn.Tanh()
        self.HwL = torch.nn.Linear(nH, 1024)
        self.HpL = torch.nn.Linear(1024 + 512, 1024)
        self.HsL = torch.nn.Linear(1024 + 512, 1024)
        self.final = torch.nn.Linear(1024,nO)
    
    def forward(self,V,Q):
        Qwl = self.WL(Q,form='nlc')
        Vwhat,Qwhat = self.parallelattention(V, Qwl)
        # print(Vwhat.shape,Qwhat.shape)
        hwin = Vwhat+Qwhat
        hw = self.tanh(self.HwL(hwin))
        Qpl = self.PL(Q,form='nlc')
        Vphat,Qphat = self.parallelattention(V, Qpl)
        # print(Vphat.shape,Qphat.shape,(Vphat+Qphat).shape,hw.shape)
        hpin = torch.cat([Vphat+Qphat,hw],1)
        # print(hpin.shape)
        hp = self.tanh(self.HpL(hpin))
        Qsl = self.QL(Q)
        Vshat,Qshat = self.parallelattention(V, Qsl)
        hsin = torch.cat([Vwhat+Qwhat,hw],1)
        hs = self.tanh(self.HsL(hsin))
        out = self.final(hs)
        return out
