from torchtext.data.utils import get_tokenizer
from joblib import dump, load
from lime.lime_text import LimeTextExplainer
from lime import lime_text
from pathlib import Path
import os 


class TfidfModel:
    def __init__(self):
        super().__init__()
        # The pipline has the tokenizer saved in vectorizer but I still need this for lime
        self.tokenize=get_tokenizer('basic_english')
        self.pipeline = load(Path(os.getcwd()+'/app/models/binaries'+'/tfidf_pipeline.joblib')) 
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
    def get_lime_exp(self, text):
        text = ' '.join(self.tokenize(text))
        exp = self.explainer.explain_instance(text, self.pipeline.predict_proba, num_features=6)
        return exp.as_html()