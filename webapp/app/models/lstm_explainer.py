
import torch
import torch.nn as nn
from torchtext import datasets, data
import torchtext
import random
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

from torchtext.data import TabularDataset
import os
from tqdm.notebook import tqdm
from lime.lime_text import LimeTextExplainer
from lime import lime_text
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 256

def bn_dropout_fc(in_features, out_features, dropout_p=0.5):
    return [nn.BatchNorm1d(in_features), nn.Dropout(dropout_p), nn.Linear(in_features, out_features)]

class NNet(nn.Module):
    def __init__(self,embeddings, embedding_dim, output_dim, pad_idx, lstm_hidden_size, fc_hidden):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=pad_idx)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_hidden_size, 
                            num_layers=2, batch_first=False,
                            bidirectional=True,dropout=0.0)

        self.fc = nn.Sequential(
            *bn_dropout_fc(lstm_hidden_size * 2, fc_hidden[0]),
            nn.ReLU(),
            *bn_dropout_fc(fc_hidden[0], fc_hidden[1]),
            nn.ReLU(),
            *bn_dropout_fc(fc_hidden[1], output_dim)
        )


    def forward(self, input, input_lengths):
        a1 = self.dropout(self.embeddings(input.to(device)))

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(a1, input_lengths,enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embeddings,) 
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        output,output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        return self.fc(hidden)



class Predict:
  def __init__(self, model, text_field):
    self.model = model
    self.text_field = text_field
  def __call__(self, examples):
    examples_new=[]
    for s in examples:
      if s.strip() == '':
        examples_new.append(self.text_field.unk_token)
      else:
        examples_new.append(s)
    ds = torchtext.data.Dataset([torchtext.data.Example.fromlist([i],fields=[('text', self.text_field)]) for i in examples_new],fields=[('text', self.text_field)])
    iterator = data.BucketIterator(ds, batch_size=bs,
                    device=device,
                    shuffle=False,
#                    sort_key=lambda x: len(x.text),
                    sort_within_batch = False)
    res = []
    self.model.eval()
    for batch in iterator:                     
      text, text_lengths = batch.text
      with torch.no_grad():
        y_hat = self.model(text, text_lengths)
      res.append(y_hat.softmax(1).detach().cpu())
    return torch.cat(res,dim=0).numpy()

  

class LSTMExplainer:
    def __init__(self, tokenizer, num_features):
        super().__init__()
        self.TEXT = torch.load(os.getcwd()+'/app/models/binaries'+'/text_field.ptz')
        embedding_size = 300


        lstm_hidden=200
        fc_hidden = [100,50]
        self.model = NNet(self.TEXT.vocab.vectors,
                   embedding_size,
                   2,
                   self.TEXT.vocab.stoi[self.TEXT.pad_token],
                   lstm_hidden,fc_hidden).to(device)
        
        self.model.load_state_dict(torch.load(os.getcwd()+'/app/models/binaries'+'/rnn_model.pt',map_location=device)) # TODO change to Path
        self.predict = Predict(self.model,self.TEXT)
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
        self.tokenizer = tokenizer
        self.num_features = num_features
    def get_lime_exp(self, text):
       
        text = ' '.join(self.tokenizer(text))
        exp = self.explainer.explain_instance(text, self.predict, num_features=self.num_features,top_labels=2, num_samples=500)
        return exp.as_html(text=True,labels=(1,))