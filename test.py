import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class doc_LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size,output_size):
        super(doc_LinearSeqAttn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0),x.size(1),self.output_size)
        x_mask = x_mask.unsqueeze(2).expand_as(scores)

        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class LinearSeqAttn_ques(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn_ques, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


doc_hiddens = Variable(torch.FloatTensor(32,809,768))
print('doc_hiddens     : ',doc_hiddens.size())

relu=torch.nn.ReLU(inplace=True)
maxPool=torch.nn.MaxPool1d(kernel_size=2)
adaAvgPool=torch.nn.AdaptiveAvgPool1d(output_size=30)
doc_conv1 = torch.nn.Conv1d(in_channels=768,out_channels=128,kernel_size=1)
doc_conv2 = torch.nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
doc_conv3 = torch.nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
doc_self_attn = doc_LinearSeqAttn(input_size=128, output_size=25)



# embedding size reduction
doc_hiddens = torch.transpose(relu(doc_conv1(torch.transpose(doc_hiddens,1,2))),1,2)
print('doc_hiddens     : ',doc_hiddens.size())


# document length reduction (something like summarizing the document)
doc_hiddens = torch.transpose(relu(maxPool(doc_conv2(torch.transpose(doc_hiddens, 1, 2)))), 1, 2)
print('doc_hiddens     : ',doc_hiddens.size())

doc_hiddens = torch.transpose(relu(maxPool(doc_conv3(torch.transpose(doc_hiddens, 1, 2)))), 1, 2)
print('doc_hiddens     : ',doc_hiddens.size())

x_mask = Variable(torch.ByteTensor(doc_hiddens.size(0),doc_hiddens.size(1)).fill_(0))
alpha_doc = doc_self_attn.forward(doc_hiddens, x_mask)
print('alpha_doc       : ',alpha_doc.size())

doc_hiddens = torch.bmm(alpha_doc.transpose(1, 2), doc_hiddens)
print('doc_hiddens     : ',doc_hiddens.size())

question_hiddens = Variable(torch.FloatTensor(32,20,768))
print('question_hiddens:',question_hiddens.size())
question_conv1 = torch.nn.Conv1d(in_channels=768,out_channels=128,kernel_size=1)
ques_self_attn = LinearSeqAttn_ques(input_size=128)

question_hiddens = torch.transpose(relu(question_conv1(torch.transpose(question_hiddens,1,2))),1,2)
print('question_hiddens:',question_hiddens.size())
x_mask = Variable(torch.ByteTensor(question_hiddens.size(0),question_hiddens.size(1)).fill_(0))
alpha_ques = ques_self_attn.forward(question_hiddens,x_mask)
print('alpha_ques      : ',alpha_ques.size())
ques_hidden = torch.bmm(alpha_ques.unsqueeze(1), question_hiddens)
print('ques_hidden     : ',ques_hidden.size())

'''
Relation Network
'''

#Concatenate all available relations
x_i = doc_hiddens.unsqueeze(1)
x_i = x_i.repeat(1,25,1,1)
x_j = doc_hiddens.unsqueeze(2)
x_j = x_j.repeat(1,1,25,1)
q = ques_hidden.unsqueeze(1).repeat(1,25,25,1)
relations = torch.cat([x_i,x_j,q],3)
print('relations       :',relations.size())

g_fc1 = torch.nn.Linear(128*3,128*3)
g_fc2 = torch.nn.Linear(128*3,128*3)
g_fc3 = torch.nn.Linear(128*3,128*3)
g_fc4 = torch.nn.Linear(128*3,128*3)


x_ = relations.view(-1,384)

x_ = relu(g_fc1(x_))
x_ = relu(g_fc2(x_))
x_ = relu(g_fc3(x_))
x_ = relu(g_fc4(x_))
x_g = x_.view(relations.size(0),relations.size(1)*relations.size(2),-1)
x_g = x_g.sum(1).squeeze()
print('x_g             :',x_g.size())

#reshape for passing thourgh netowrk