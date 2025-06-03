import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def train(self, corpus, vocab_size):
        # Tách văn bản thành các ký tự ban đầu
        words = [list(word) + ['</w>'] for word in corpus.split()]
        word_freq = Counter(corpus.split())

        # Tính tần suất cặp ký tự
        def get_pairs(word_list):
            pairs = defaultdict(int)
            for word, freq in word_freq.items():
                symbols = word_list[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            return pairs

        # Gộp cặp ký tự phổ biến nhất
        for _ in range(vocab_size - len(self.vocab)):
            pairs = get_pairs(words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Cập nhật danh sách từ
            new_words = {}
            for word in word_freq:
                new_word = []
                i = 0
                while i < len(words[word]):
                    if i < len(words[word]) - 1 and (words[word][i], words[word][i + 1]) == best_pair:
                        new_word.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word.append(words[word][i])
                        i += 1
                new_words[word] = new_word
            words = new_words

        # Xây dựng từ vựng
        self.vocab = {''.join(word): i for i, word in enumerate(set(''.join(merge) for merge in self.merges))}

    def encode(self, text):
        words = [list(word) + ['</w>'] for word in text.split()]
        for merge in self.merges:
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == merge:
                        new_word.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words
        return [self.vocab.get(''.join(word), 0) for word in words]

    def decode(self, tokens):
        text = ''.join([list(self.vocab.keys())[list(self.vocab.values()).index(token)] for token in tokens])
        return text.replace('</w>', ' ')

# Ví dụ sử dụng
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    corpus = "hello world hello there world"
    tokenizer.train(corpus, vocab_size=50)
    encoded = tokenizer.encode("hello world")
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)