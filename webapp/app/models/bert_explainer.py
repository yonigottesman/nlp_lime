from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.processors.utils import InputFeatures, InputExample,SingleSentenceClassificationProcessor
from transformers import Trainer, TrainingArguments, BertTokenizerFast
import os
from lime.lime_text import LimeTextExplainer
from lime import lime_text
from transformers import BertForSequenceClassification
from nlp import Dataset
import torch as torch
from tqdm import tqdm


class Predict:
  def __init__(self, model, tokenizer):
    self.model = model
    self.model.eval()
    self.tokenizer = tokenizer
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def __call__(self, examples):
    ds = Dataset.from_dict({'text':examples})
    ds = ds.map(lambda batch: self.tokenizer(batch['text'], truncation=True, padding='max_length'), batched=True, batch_size=512)
    ds.set_format('torch', columns=['input_ids','token_type_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
    res = []
    for batch in tqdm(dataloader):
      batch = {k: v.to(self.device) for k, v in batch.items()}
      outputs = self.model(**batch)
      res.append(outputs[0].softmax(1).detach().cpu())
    return torch.cat(res,dim=0).numpy()


class BertExplainer:
    def __init__(self, num_features):
        super().__init__()
        bert_model_path=os.getcwd()+'/app/models/binaries'+'/bert_saved_model/'
        model = BertForSequenceClassification.from_pretrained(bert_model_path)       
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',model_max_length=256)
        
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
        self.predict = Predict(model,tokenizer)
        self.num_features = num_features
        
    def get_lime_exp(self, text):
       
        exp = self.explainer.explain_instance(text, self.predict, num_features=self.num_features, num_samples=500)
        return exp.as_html(text=True)