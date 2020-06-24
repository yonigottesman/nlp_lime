from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.processors.utils import InputFeatures, InputExample,SingleSentenceClassificationProcessor
from transformers import Trainer, TrainingArguments, BertTokenizerFast
import os
from lime.lime_text import LimeTextExplainer
from lime import lime_text


class Predict:
  def __init__(self, model, labels, tokenizer, trainer):
    self.model = model
    self.labels = labels
    self.tokenizer = tokenizer
    self.trainer = trainer
  def __call__(self, examples):
    examples = [InputExample(guid=i, text_a=text,label=1) for
                i,text in enumerate(examples)] # just use 1 as label it will be ignored
    processor = SingleSentenceClassificationProcessor(labels=self.labels,examples=examples)
    features = processor.get_features(self.tokenizer,return_tensors=None)
    return self.trainer.predict(test_dataset=features).predictions

class BertExplainer:
    def __init__(self, num_features):
        super().__init__()
        # For sure I don't need this and can get from config
        labels=[1,2]
        bert_model_path=os.getcwd()+'/app/models/binaries'+'/bert-base-uncased/'
        config = AutoConfig.from_pretrained(bert_model_path+'/model', num_labels=len(labels))
        model = AutoModelForSequenceClassification.from_pretrained(bert_model_path+'/model', config=config)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_path+'/tokenizer')   
        
        # Not done here for training just to get be able to batch the lime examples and predict easily
        training_args = TrainingArguments(output_dir='/tmp/')
        trainer = Trainer(model=model, args=training_args)
        
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
        self.predict = Predict(model,labels,self.tokenizer,trainer)
        self.num_features = num_features
        
    def get_lime_exp(self, text):
       
        text = ' '.join(self.tokenizer.tokenize(text))
        exp = self.explainer.explain_instance(text, self.predict, num_features=self.num_features)
        return exp.as_html(text=True)