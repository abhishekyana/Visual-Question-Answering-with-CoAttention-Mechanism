from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import datetime
from CoAttention_dataset import VqaDataset
import numpy as np
from CoAttention_net import CoattentionNet
from CoAttention_utils import *

class CoattentionNetExperimentRunner(object):
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_D = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg")
        val_D = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg")
        mval=len(val_D)
        # inds30 = np.arange(0,int(0.3*mval))
        # inds70 = np.arange(int(0.3*mval),mval)
        TOT = np.random.randint(0,mval,mval)
        inds30,inds70 = TOT[:int(0.3*mval)], TOT[int(0.3*mval):]
        val70 = torch.utils.data.Subset(val_D,inds70)
        val_dataset = torch.utils.data.Subset(val_D,inds30)
        train_dataset = torch.utils.data.ConcatDataset([train_D,val70])
        # self.model = SimpleBaselineNet(len(train_D.SetQdict)+1000, len(train_D.SetAdict)).cuda()
        self._model = CoattentionNet()
        # self.BCE = torch.nn.BCELoss()
        self.CE = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        # self.optim = torch.optim.RMSprop(self._model.parameters(), lr=4e-4, weight_decay=1e-8,momentum=0.99)#
        self.optim = torch.optim.Adam(self._model.parameters(), lr=0.001)#can be changed
        # self._model = CoattentionNet()

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()
        self.softmax = torch.nn.Softmax(dim=1)
        self.modeltype = 'coattention'
        self.till=250
        self.W2ID,self.EMB = GetEmbeddings()
    	self.AnsWords = getAnsWords(path='./supportfiles/CoAttAns.d')
    	self.AW2ID = {w:i for i,w in enumerate(self.AnsWords)}
    	self.AW2ID['<unk>']=1000
    	self.till=1250
    	self.tbname = './RUNS/COATT/'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.writer = SummaryWriter(log_dir=self.tbname)
        if self._cuda:
            print("CUDA-fied")
            self._model = self._model.cuda()

    def _optimize(self, predicted_answers, true_answer_ids):
        # ll = self.BCE(self.softmax(predicted_answers), true_answer_ids)
        ll = self.CE(predicted_answers, true_answer_ids.argmax(1).long())
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()
        return ll
        raise NotImplementedError()

    def validate(self,till=1250):
        #  validation accuracy
        Acc=[]
        print("Validating...")
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            img = batch_data['image']
            qENC = batch_data['question']#_nBOW']#
            ground_truth_answer = batch_data['answer']
            if self.modeltype=='coattention':
            	qENC = Sentence2Embeddings(qENC,self.W2ID,self.EMB)
            	ground_truth_answer = Answer2OneHot(ground_truth_answer,self.AW2ID)
            predicted_ans = self._model(img.cuda(), qENC.cuda()) # TODO
            predicted_answer = self.softmax(predicted_ans)
            tpk, tpkvals = torch.topk(predicted_answer,k=1,dim=1)
            gttpk, gtpkvals = torch.topk(ground_truth_answer.cuda(),k=1,dim=1)
            # print(tpkvals,gtpkvals)
            acc = (tpkvals==gtpkvals).double().mean()
            Acc.append(acc.item())
            if batch_id>=till:
            	print("Done")
            	return sum(Acc)/len(Acc)
        return sum(Acc)/len(Acc)
        raise NotImplementedError()

    def train(self):
        # print("Hello")
        # for pg in self.optim.param_groups:
        #     print("Learning rate is ",pg['lr'])
        #     pg['lr']*=0.5
        #     print("Learning rate is ",pg['lr'])
        # return None
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            #Write code to Save the torch model
            PathforModel = './savedmodels/coattention/modelat_'+str(epoch)
            torch.save(self._model.state_dict(), PathforModel)
            print(f"Saved Simple Baseline model at epoch {epoch} at {PathforModel}")
            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id
                img = batch_data['image']
                qENC = batch_data['question']#_nBOW']
                ground_truth_answer = batch_data['answer']
                if self.modeltype=='coattention':
	                qENC = Sentence2Embeddings(qENC,self.W2ID,self.EMB)
	                ground_truth_answer = Answer2OneHot(ground_truth_answer,self.AW2ID)
                predicted_answer = self._model(img.cuda(), qENC.cuda()) # TODO
                loss = self._optimize(predicted_answer, ground_truth_answer.cuda())

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    self.writer.add_scalar('train/loss', loss, current_step)

                if current_step % self._test_freq == 0:
                    # self._model.eval()
                    val_accuracy = self.validate(till=self.till)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    self.writer.add_scalar('Val/accuracy', val_accuracy, current_step)
			#Changing the learning rate to half for every epoch:
            for pg in self.optim.param_groups:
                pg['lr']*=0.5
                try:
                    self.writer.add_scalar('learning_rate',pg['lr'],current_step)
                except Exception as e:
                    print(e)
            val_accuracy = self.validate()
            print("complete accuracy")
            print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))