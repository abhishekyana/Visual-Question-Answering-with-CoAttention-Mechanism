from torch.utils.data import Dataset
from vqa import VQA
from PIL import Image
from scipy.misc import imresize
import torch
import numpy as np
import pickle

# def filterit(s,W2ID):
#     s=s.lower()
#     S=''
#     for c in s:
#         if c in ' abcdefghijklmnopqrstuvwxyz0123456789':
#         	S+=c
#     S = " ".join([x  if x and x in W2ID else "<unk>" for x in S.split()])
#     return S

# def Sentence2Embeddings(sentence,W2ID,EMB):
#     if type(sentence)==str:
#         sentence = filterit(sentence, W2ID)
#         #print(sentence)
#         IDS = torch.tensor([W2ID[i] for i in sentence.split(" ")])
#         return EMB(IDS)
#     if type(sentence)==list:
#         sembs = []
#         for sent in sentence:
#             sent = filterit(sent,W2ID)
#             IDS = torch.tensor([W2ID[i] for i in sent.split(" ")])
#             sembs.append(EMB(IDS))
#         sembs = torch.nn.utils.rnn.pad_sequence(sembs,batch_first=True)
#         return sembs
def ImageID2Embeddings(path):
	return np.load(path)

# def GetEmbeddings(path='./student_code/supportfiles/GloVe300.d'):
#     GloVe = pickle.load(open(path,'rb'))
#     W2ID = {w:i for i,w in enumerate(sorted(list(GloVe.keys())))}
#     EMB = torch.nn.Embedding(len(W2ID),300)
#     EMB.weight.requires_grad=False
#     GloVeW = np.vstack([GloVe[w] for w in W2ID])
#     EMB.weight.data.copy_(torch.from_numpy(GloVeW))
#     return W2ID, EMB

# def getAnsWords(path='./student_code/supportfiles/CoAttAns.d'):
# 	with open(path,'rb') as file:
# 		data = pickle.load(file)
# 	return data

# another comment

# def Answer2OneHot(answer,AW):
# 	Aembs = torch.zeros(len(AW))
# 	for w in answer.split(" "):
# 		if w in AW: Aembs[AW[w]]=1
# 	return Aembs

class VqaDataset(Dataset):
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
		if 'val2014' in image_filename_pattern:
			self.toloadpath = './student_code/supportfiles/ResNetDataVal/feat_'
		if 'train2014' in image_filename_pattern:
			self.toloadpath = './student_code/supportfiles/ResNetData/feat_'
		# self.imgdict = {idt:image_dir+'/'+image_filename_pattern.format(str(idt).zfill(12)) for idt in self.vqa.imgToQA}
		# self.AnsWords = getAnsWords(path='./student_code/supportfiles/CoAttAns.d')
		# self.AW2ID = {w:i for i,w in enumerate(self.AnsWords)}
		# self.W2ID, self.EMB = GetEmbeddings(path='./student_code/supportfiles/GloVe300.d')
		pass

	def __len__(self):
		return len(self.idx2key)

	def __getitem__(self, idx):
		key = self.idx2key[idx]
		question = self.vqa.qqa[key]['question']
		answer = self.vqa.qa[key]['answers'][0]['answer']
		imgid = self.vqa.qqa[key]['image_id']
		imgpath = self.toloadpath+str(imgid)+'.npy'
		t = ImageID2Embeddings(imgpath).reshape(-1,14*14)
		img = torch.Tensor(t)
		return {'answer':answer,'image':img,'question':question}
