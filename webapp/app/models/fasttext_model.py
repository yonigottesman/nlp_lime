import fasttext
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime import lime_text
from fasttext import load_model
import os


class Predict:
    def __init__(self, model):
        self.model = model
    def __call__(self, examples):
        all_results = []
        for example in examples:

            result = self.model.predict(example, k=2)
            labels = result[0]
            probs = result[1]
            # the resuly vecor should be (label1 prob,label2 prob), fasttext returns best first so must swap
            if (labels[0] == '__label__2'):
                tmp=probs[0]
                probs[0]=probs[1]
                probs[1]=tmp 
            all_results.append(probs.reshape(1,2))
        return np.vstack(all_results)
            

class FasttextModel:
    def __init__(self, tokenizer, num_features):
        super().__init__()
        self.num_features = num_features
        self.tokenizer = tokenizer
        
        model = load_model(os.getcwd()+'/app/models/binaries'+'/fasttext.ftz')
        
        self.predict = Predict(model)
        
        self.explainer = LimeTextExplainer(class_names=['Negative','Positive'])
    def get_lime_exp(self, text):
        # The pipline vectorizer already contains the tokenizer but I want lime to see the same tokens to train its model
        text = ' '.join(self.tokenizer(text))
        exp = self.explainer.explain_instance(text, self.predict, num_features=self.num_features, num_samples=500)
        return exp.as_html()