
from torchtext.data.utils import get_tokenizer
from app.models.tfidf_linear import TfidfModel
from app.models.fasttext_model import FasttextModel
from app.models.lstm_model import LSTMModel

tokenizer=get_tokenizer('basic_english')
NUM_FEATURES = 6


tfidf_model = TfidfModel(tokenizer, num_features=NUM_FEATURES)
fasttext_model = FasttextModel(tokenizer, num_features=NUM_FEATURES)
lstm_model = LSTMModel(tokenizer,NUM_FEATURES)