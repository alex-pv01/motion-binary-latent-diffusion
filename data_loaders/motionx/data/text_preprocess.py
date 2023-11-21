import string

class TextPreprocess(object):
    """
    Gets a string and returns a list of words and their corresponding POS tags
    """
    def __init__():
        pass

    def __getitem__(self, item):
        assert isinstance(item, str)
        return self.get_words(item)
    
    def get_words(self, text):
        # Get words and POS tags from a string
        # words = []
        # pos_tags = []
        # for word, pos in pos_tag(word_tokenize(text)):
        #     words.append(word)
        #     pos_tags.append(pos)
        # return words, pos_tags
        pass
    
    def word_tokenize(self, text):
        # Get words from a string
        words = text.split()

        # Remove punctuation
        words = [word.strip(string.punctuation) for word in words]