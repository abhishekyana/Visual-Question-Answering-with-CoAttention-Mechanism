from operator import itemgetter
import os
import torch
import numpy as np
import torchvision.models as models

from torch.utils.data import Dataset
from vqa import VQA
from PIL import Image
from scipy.misc import imresize
import torch
import numpy as np
import pickle
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader

def filterAnswer(s):
    s=s.lower()
    S=''
    for c in s:
        if c in ' abcdefghijklmnopqrstuvwxyz0123456789': S+=c
    return S

def MakeTopWords(D,top=1000,onlywords=True):
    Aall = ""
    for i in D['annotations']:
        Aall += filterAnswer(i['answers'][0]['answer'])+' '
    Aall = Aall.strip()
    W2C = {}
    for w in S.split(" "):
        if not w: continue
        if w not in W2C: W2C[w]=1
        else:W2C[w]+=1
    sortedWords = sorted(W2C.items(), key=itemgetter(1), reverse = True)
    SW = sorted(sortedWords[:top])
    if onlywords:
        W = [w for w,i in SW]
        return W
    return SW

class VqaDataset1(Dataset):
    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern):
        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.idx2key = {i:self.vqa.qa[v]['question_id'] for i,v in enumerate(self.vqa.qa)}
        self.imgdict = {idt:image_dir+'/'+image_filename_pattern.format(str(idt).zfill(12)) for idt in self.vqa.imgToQA}
        self.imgkeys = sorted(list(self.imgdict.keys()))
        pass

    def __len__(self):
        return len(self.imgkeys)

    def __getitem__(self, idx):
        key = self.imgkeys[idx]
        t = Image.open(self.imgdict[key])
        t = imresize(t,(2*224,2*224,3))
        if len(t.shape)==2:
            t = np.stack((t,)*3, axis=-1)
        t = np.asarray(t)
        t = t.transpose(2,0,1)
        img = torch.Tensor(t)
        return key,self.imgdict[key],img


def makeGloVe():
    with open('./supportfiles/glove.6B/glove.6B.300d.txt','r') as f:
        data=f.read()
    GloVe = {}
    for i,v in enumerate(tqdm(data.split('\n'))):
        if v:
            V = v.split(" ")
            word,vec = V[0],np.array(V[1:]).astype(np.float32)
            GloVe[word] = vec
        else:
            print("Not done for",i,v)
    pickle.dump(GloVe,open('./supportfiles/GloVe300.d','wb'))
    return True

if __name__=='__main__':
    ResNet = models.resnet18(pretrained=True)
    ResNet = ResNet.cuda()
    newmodel = torch.nn.Sequential(*(list(ResNet.children())[:-2]))
    train_D = VqaDataset1(image_dir='Data/train2014/',
                    question_json_file_path='./Data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json',
                    annotation_json_file_path='./Data/Annotations_Train_mscoco/mscoco_train2014_annotations.json',
                    image_filename_pattern="COCO_train2014_{}.jpg")
    val_D = VqaDataset1(image_dir='Data/val2014/',
                        question_json_file_path='./Data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json',
                        annotation_json_file_path='./Data/Annotations_Val_mscoco/mscoco_val2014_annotations.json',
                        image_filename_pattern="COCO_val2014_{}.jpg")
    traindata = DataLoader(train_D,batch_size=32,shuffle=False, num_workers=10)
    validdata = DataLoader(val_D,batch_size=32,shuffle=False, num_workers=10)
    path = './supportfiles/ResNetData/'
    pathval = './supportfiles/ResNetDataVal/'
    for i,bdata in enumerate(tqdm(traindata)):
        k,ik,img = bdata
        outs = newmodel(img.cuda())
        for ii,kk in enumerate(k):
            # D[kk.item()]=outs[ii].cpu().detach().numpy()
            nm = pathval+'feat_'+str(kk.item())
            np.save(nm, outs[ii].cpu().detach().numpy())
    for i,bdata in enumerate(tqdm(validdata)):
        k,ik,img = bdata
        outs = newmodel(img.cuda())
        for ii,kk in enumerate(k):
            # D[kk.item()]=outs[ii].cpu().detach().numpy()
            nm = path+'feat_'+str(kk.item())
            np.save(nm, outs[ii].cpu().detach().numpy())
    print("Done")
    if not os.path.isfile('./supportfiles/GloVe300.d'):
        makeGloVe()
    if not os.path.isfile('./supportfiles/CoAttAns.d'):
        with open('./supportfiles/CoAttAns.d','wb') as file:
            print("Making the top 1000 words")
            WORDS = MakeTopWords(train_D.vqa.dataset)
            pickle.dump(WORDS,file)