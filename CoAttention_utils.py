import torch
import numpy as np
import pickle


def filterit(s,W2ID):
    s=s.lower()
    S=''
    for c in s:
        if c in ' abcdefghijklmnopqrstuvwxyz0123456789':
        	S+=c
    S = " ".join([x  if x and x in W2ID else "<unk>" for x in S.split()])
    return S

def Sentence2Embeddings(sentence,W2ID,EMB):
    if type(sentence)==str:
        sentence = filterit(sentence, W2ID)
        #print(sentence)
        IDS = torch.tensor([W2ID[i] for i in sentence.split(" ")])
        return EMB(IDS)
    if type(sentence)==list:
        sembs = []
        for sent in sentence:
            sent = filterit(sent,W2ID)
            IDS = torch.tensor([W2ID[i] for i in sent.split(" ")])
            sembs.append(EMB(IDS))
        sembs = torch.nn.utils.rnn.pad_sequence(sembs,batch_first=True)
        return sembs

def GetEmbeddings(path='./student_code/supportfiles/GloVe300.d'):
    GloVe = pickle.load(open(path,'rb'))
    W2ID = {w:i for i,w in enumerate(sorted(list(GloVe.keys())))}
    EMB = torch.nn.Embedding(len(W2ID),300)
    EMB.weight.requires_grad=False
    GloVeW = np.vstack([GloVe[w] for w in W2ID])
    EMB.weight.data.copy_(torch.from_numpy(GloVeW))
    return W2ID, EMB

def getAnsWords(path='./student_code/supportfiles/CoAttAns.d'):
    with open(path,'rb') as file:
        data = pickle.load(file)
    return data


def Answer2OneHot1(answers,AW):
    A=[]
    for answer in answers:
        Aembs = torch.zeros(len(AW))
        for w in answer.split(" "):
            if w in AW: 
            	Aembs[AW[w]]=1
            	break
            else:
                Aembs[0]=1
                break
        A.append(Aembs)
    A = torch.stack(A)
    return A

def Answer2OneHot(answers,AW):
    A=[]
    for answer in answers:
        Aembs = torch.zeros(len(AW))
        w = answer.split(" ")[0]
        if w in AW:Aembs[AW[w]]=1
        else:Aembs[-1]=1
        A.append(Aembs)
    A = torch.stack(A)
    return A