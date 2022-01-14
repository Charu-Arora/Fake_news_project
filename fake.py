# %%
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# %%
np.random.seed(500)
corpus=pd.read_csv(r"https://raw.githubusercontent.com/Gunjitbedi/Text-Classification/master/corpus.csv",encoding='latin-1')

# %%
corpus['text'].dropna(inplace=True)
corpus['text']=[entry.lower() for entry in corpus['text']]
corpus['text']= [word_tokenize(entry) for entry in corpus['text']]
corpus['text']=[entry for entry in corpus['text'] if not entry in stopwords.words('english')]
tag_map=defaultdict(lambda : wn.NOUN)
tag_map['V']=wn.VERB
tag_map['A']=wn.ADJ
tag_map['R']=wn.ADV

# %%
for i,entry in enumerate(corpus['text']):
    finalset=[]
    word_lemmatization=WordNetLemmatizer()
    for word,tag in pos_tag(entry):
        if word.isalpha():
            wordset=word_lemmatization.lemmatize(word,tag_map[tag[0]])
            finalset.append(wordset)
    corpus.loc[i,'text_final']=str(finalset)