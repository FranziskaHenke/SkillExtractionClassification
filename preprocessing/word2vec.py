import gensim 
from gensim.models import Word2Vec 
from paths import CBOW_PATH, SKIP_GRAM_PATH, MODEL_BASE_PATH
from os import path, makedirs
from get_clean_data import get_clean_data, read_clean_data_for_year
from clean_alt_corpus import read_data_i, read_corpus_file
import time
import numpy


class W2V():

  def __init__(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=100):
    self.model = None
    self.init_model(skipgram=skipgram, punctuation_marks=punctuation_marks, stoppwords=stoppwords, multiword=multiword, lower_case=lower_case, vector_dim=vector_dim)

  def load_model(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200): 
    
    model_path = self.get_model_path(skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)
    print('loading existing w2v model from ', model_path)
    self.model = Word2Vec.load(model_path)

  def get_model_path(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    model_path = MODEL_BASE_PATH  
    model_path +='/pm' if(punctuation_marks) else '/no_pm'
    model_path +='_sw' if(stoppwords) else '_no_sw'
    if(multiword): model_path += '_multi'
    if(lower_case): model_path += '_lowerCase'
    if(vector_dim != 200): model_path += '_vecDim'+str(vector_dim)
    makedirs(model_path+'/', exist_ok=True)
    model_path += SKIP_GRAM_PATH if(skipgram) else CBOW_PATH
    return model_path

  def save_model(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=True, lower_case=False, vector_dim=200):  
    model_path = self.get_model_path(skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)
    print('saving', model_path)
    self.model.save(model_path)

  def train_model(self, data, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    self.model.build_vocab(data, update=True)
    self.model.train(data, total_examples=len(data), epochs=15)
    self.save_model(skipgram, punctuation_marks, stoppwords,multiword, lower_case, vector_dim)

  def tokens_to_lower(self, data):
    lower = []
    for i in range(len(data)):
      lower.append([t.lower() for t in data[i]])
    return lower

  def create_model(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    print('skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim', skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)
    print('creating new w2v model', time.strftime("%H:%M:%S +0000", time.localtime()))
    start = time.time()
    data = []
    #2018: 200.000
    data = read_clean_data_for_year(2018, not punctuation_marks, not stoppwords)  
    data = data[len(data)-200000:]
    if(lower_case):
      data = self.tokens_to_lower(data)
    print('Initializing model')
    self.model = gensim.models.Word2Vec(data[:200000], iter=10,min_count = 1, size = vector_dim, window = 7, hs=0, sg = skipgram)# negative=0 
    self.save_model(skipgram, punctuation_marks, stoppwords,multiword, lower_case, vector_dim)    
    #2019: 100.00
    data = read_clean_data_for_year(2019, not punctuation_marks, not stoppwords)
    data = data[len(data)-100000:]
    if(lower_case):
      data = self.tokens_to_lower(data)
    self.train_model(data, skipgram=skipgram, punctuation_marks=punctuation_marks, stoppwords=stoppwords, multiword=multiword, lower_case=lower_case, vector_dim=vector_dim)
    print('saving model, took:', time.time() - start)
    self.save_model(skipgram, punctuation_marks, stoppwords,multiword, lower_case, vector_dim)

  def train_w2v(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    print('training new model, getting training data...')
    self.create_model(skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)

  def init_model(self, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    print('skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim', skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)
    model_path = self.get_model_path(skipgram, punctuation_marks, stoppwords, multiword, lower_case, vector_dim)
    print('init', model_path)
    if(path.exists(model_path)):
      self.load_model(skipgram=skipgram, punctuation_marks=punctuation_marks, stoppwords=stoppwords, multiword=multiword, lower_case=lower_case, vector_dim=vector_dim)
    else:
      self.train_w2v(skipgram=skipgram, punctuation_marks=punctuation_marks, stoppwords=stoppwords, multiword=multiword, lower_case=lower_case, vector_dim=vector_dim)

  def get_closest_word_to_vec(self,vector):
    return self.model.similar_by_vector(vector, topn=1)

  def get_closest_word_to_vec_matrix(self, vector, skipgram=True, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=False, vector_dim=200):
    words = []
    for i in range(len(vector)):
      words.append(self.model.similar_by_vector(vector[i], topn=1))
    return words  

  def is_word_in_model(self, word):
    return word in self.model.wv.vocab
    
  def get_w2v(self, word):
    return self.model.wv[word]

  def test_words(self, word1, word2, lower):
    if(not lower):
      print(word1, word2,self.model.similarity(word1, word2))
    else:
      print(word1.lower(), word2.lower(),self.model.similarity(word1.lower(), word2.lower()))

  def test_model(self, lower=False):
    words=[['C++', 'Rentabilität'],['C++', 'Mikrobiologie'],['C++', 'C#'],['C++', 'C'],['C', 'C#'],['Java', 'C#'],['C++', 'jeglicher'],['C++', 'jeglicher'],['HTML', 'CSS'],['Java', 'Javascript'],['Java', 'JavaScript'],['Javascript', 'JavaScript'], ['Powerpoint', 'PowerPoint']] 
    for i in range(len(words)):
      self.test_words(words[i][0],words[i][1], lower)

  def test_zero_vector(self):  
    zero_v = [0] * 100
    sim = self.model.similar_by_vector(numpy.array(zero_v))
    print("most similar to zero vector: ",sim)

  def test_closest_to_word(self, word, lower):  
    if(not lower):
      print("most similar to '"+word+"': ",self.model.similar_by_word(word, topn=10))
    else:
      print("most similar to '"+word.lower()+"': ",self.model.similar_by_word(word.lower(), topn=10))

  def test_similar_words(self, lower=False):
    words = ['Office', 'MS', 'Ruby', 'Java', 'C++', 'C', 'HTML', 'node.js']
    words = ['Java', 'Ruby', 'Medizin', 'Metzger', 'Photoshop', 'Datenbanken']
    for i in range(len(words)):
      self.test_closest_to_word(words[i], lower)

if __name__ == '__main__':
  w = W2V()
  #w.test_model()
  w.test_similar_words()
