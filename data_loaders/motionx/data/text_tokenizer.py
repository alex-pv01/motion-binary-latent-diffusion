import string
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

class TextTokenizer(object):
    """
    Gets a string and returns a list of words and their corresponding POS tags
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize(self, text):
        # Tokenize
        word_list = word_tokenize(text)
        # Remove punctuation and set to lower case
        word_list = [word.lower() for word in word_list if word not in string.punctuation]
        # Lemmatize
        word_list = [self.lemmatizer.lemmatize(word) for word in word_list]
        # Get POS
        pos_list = pos_tag(word_list)
        # Convert to list of strings "word/POS"
        tokens = [word + "/" + pos for word, pos in pos_list]

        return tokens

