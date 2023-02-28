from collections import Counter
import nltk

class BigramAnalyzer:
    def __init__(self):
        self.bigram_counter = Counter()

    def process_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        bigrams = nltk.bigrams(tokens)
        self.bigram_counter.update(bigrams)

    def get_top_n_bigrams(self, n):
        return self.bigram_counter.most_common(n)

"""
Example:


text = "The quick brown fox jumped over the lazy dog. The lazy dog slept all day."
analyzer = BigramAnalyzer()
analyzer.process_text(text)
top_bigrams = analyzer.get_top_n_bigrams(3)
print(top_bigrams)

Output:
[(('the', 'lazy'), 2), (('lazy', 'dog'), 2), (('dog', '.'), 2)]

"""
