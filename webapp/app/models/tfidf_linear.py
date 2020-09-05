
from joblib import dump, load
from lime.lime_text import LimeTextExplainer
from lime import lime_text
from pathlib import Path
import os 


class TfidfModel:
    def __init__(self, tokenizer, num_features):
        super().__init__()
        self.num_features = num_features
        self.tokenizer = tokenizer
        self.pipeline = load(Path(os.getcwd()+'/app/models/binaries'+'/tfidf_pipeline.joblib')) 
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
    def get_lime_exp(self, text):
        # The pipline vectorizer already contains the tokenizer but I want lime to see the same tokens to train its model
        text = ' '.join(self.tokenizer(text))
        exp = self.explainer.explain_instance(text, self.pipeline.predict_proba, num_features=self.num_features, num_samples=500)
        return exp.as_html()