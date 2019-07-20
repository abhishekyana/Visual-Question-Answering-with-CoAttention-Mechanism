from torch.utils.data import Dataset
from vqa import VQA
from PIL import Image
from scipy.misc import imresize
import torch
import numpy as np
import pickle


def filterit(s):
    """
    A functoin to filter out unnecessary characters from a sentece, helper for nBOW
    Args: 
        s = Sentence
    returns: filtered Sentence
    """
    s=s.lower()
    S=''
    for c in s:
        if c in ' abcdefghijklmnopqrstuvwxyz0123456789':
            if c.isdigit():
                c=' '+c+' '
            S+=c
    return S

def nBOW(sent, setdict, mode='question'):
    """
    Implementation of n Bag of Words model to convert from a sentence into multi hot encoding or nBagofWords
    Args:
        sent = sentence for which the actual conversion is needed.
        setdict = dictionary to activate the appropriate word index pair
        mode = 'answer' or 'question' for respective sentence
    returns: Multi Hot Encoded Torch Tensor
    """
    z = torch.zeros(len(setdict))
    if mode=='answer':
        w = filterit(sent).strip().split(" ")
        while '' in w:
            try:
                w.remove('')
            except ValueError:
                print("Broken the loop",w)
                break
        try:
            # print("Worked",sent)
            w_ = w[0]
            z[setdict[w_]] = 1.0
        except Exception as e:
            pass
            # print("Answer",e,w,w_,sent)
        return z
    elif mode=='question':
        for w in filterit(sent).split(" "):
            try:
                # print("Worked Q",sent)
                z[setdict[w]] = 1.0
            except Exception as e:
                # print("Question",e,w,sent)
                pass
        return z
    raise ValueError(f"{mode} is not a valid keyword, should be 'answer' or 'question'")


def D2Dict(D, mode='annotations'):
    """
    Data to Dict function for converting word to index and filtering out unnecessary words.
    Args:
        D = Dictionary of questions or annotations 
        mode = 'annotations' or 'questions'
    returns: A Dict that maps from word to index which helps in one hot encoding
    """
    #print(D.keys())
    if mode=='questions':
        Qall = ""
        for i in D['questions']:
            Qall += filterit(i['question'])+' '
        Qall = Qall.strip()
        setQ = sorted(list(set(Qall.split(' '))-set([''])))
        SetQdict = {w:i for i,w in enumerate(setQ)}
        return SetQdict
    if mode=='annotations':
        Aall = ""
        for i in D['annotations']:
            Aall += filterit(i['answers'][0]['answer'])+' '
        Aall = Aall.strip()
        setA = sorted(list(set(Aall.split(' '))-set([''])))
        SetAdict = {w:i for i,w in enumerate(setA)}
        return SetAdict

class VqaDataset(Dataset):
    """
    Pytorch Dataset class.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers  together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.idx2key = {i:self.vqa.qa[v]['question_id'] for i,v in enumerate(self.vqa.qa)}
        self.imgdict = {idt:image_dir+'/'+image_filename_pattern.format(str(idt).zfill(12)) for idt in self.vqa.imgToQA}
        # self.SetQdict = D2Dict(self.vqa.questions,'questions') # This will prepare the dataset everytime
        # self.SetAdict = D2Dict(self.vqa.dataset,'annotations') # So, I've saved the files into a pickle object for saving time
        with open('./supportfiles/QnA.d','rb') as f:
            file = pickle.load(f)
        self.SetQdict = file['questions']
        self.SetAdict = file['annotations']
        

    def __len__(self):
        """
        Length function that is required by the pytorch dataset class
        """
        return len(self.idx2key)

    def __getitem__(self, idx):
        key = self.idx2key[idx] # idx 2 key dictionary
        question = self.vqa.qqa[key]['question'] #using that and getting the question for that image
        gt_answer = self.vqa.qa[key]['answers'][0]['answer'] # Using that key to get the answer(Most voted answer) for the question. 
        imgid = self.vqa.qqa[key]['image_id'] # To obtain the image_id to load from Memory
        t = Image.open(self.imgdict[imgid]) #Using PIL to load/open image with path.
        t = imresize(t,(224,224,3)) #Resize the image to required dimensions, becaue of inconsistent image size.
        if len(t.shape)==2: # To catch a black n white image of 1 channel and make it to 3 channel
            t = np.stack((t,)*3, axis=-1)
        t = np.asarray(t)
        t = t.transpose(2,0,1) # Converting the HWC into CHW format, channel first format.
        img = torch.Tensor(t) # Converting np.ndarray into torch.Tensor
        answernBOW = nBOW(gt_answer,self.SetAdict,mode='answer') # Getting Bag of Words, Multi hot encoded question and answer format for easy training
        questionnBOW = nBOW(question,self.SetQdict)
        return {'gt_answer':gt_answer,'image':img,'gt_question':question,'answer':answernBOW,"question":questionnBOW}