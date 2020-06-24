
from torchtext.data.utils import get_tokenizer
from app.models.tfidf_linear import TfidfModel
from app.models.fasttext_model import FasttextModel
from app.models.lstm_explainer import LSTMExplainer
from app.models.bert_explainer import BertExplainer


tokenizer=get_tokenizer('basic_english')
NUM_FEATURES = 6


# All these models have lots of common code but I still prefer not to use inheritance.
# Keep it more readable and simple lie this

tfidf_model = TfidfModel(tokenizer, num_features=NUM_FEATURES)
fasttext_model = FasttextModel(tokenizer, num_features=NUM_FEATURES)
lstm_explainer = LSTMExplainer(tokenizer,NUM_FEATURES)
bert_explainer = BertExplainer(NUM_FEATURES)
