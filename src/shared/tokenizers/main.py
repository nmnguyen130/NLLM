from .basic import BasicTokenizer
from .regex import RegexTokenizer
from .test import BPETokenizer

def visual_compare(tokenizer1, tokenizer2, text: str):
    ids1 = tokenizer1.encode(text)
    ids2 = tokenizer2.encode(text)

    def decode_tokens(tokenizer, ids):
        decoded = []
        for i in ids:
            try:
                decoded.append(tokenizer.vocab[i].decode("utf-8"))
            except:
                decoded.append(str(tokenizer.vocab[i]))
        return decoded

    decoded1 = decode_tokens(tokenizer1, ids1)
    decoded2 = decode_tokens(tokenizer2, ids2)

    print("BasicTokenizer:")
    print(" ".join(f"[{i}]{t}" for i, t in zip(ids1, decoded1)))
    print("\nHeapTokenizer:")
    print(" ".join(f"[{i}]{t}" for i, t in zip(ids2, decoded2)))

    if decoded1 != decoded2:
        print("\n⚠️ Tokenization differs!")
    else:
        print("\n✅ Tokenization matches.")

text = "the theater is there for those who think they're thinkers"
tokenizer1 = BasicTokenizer()
tokenizer1.train(text, vocab_size=270, verbose=False)
tokenizer1.save("tokenizer1")
enc1 = tokenizer1.encode(text)

# tokenizer2 = RegexTokenizer()
# tokenizer2.train(text, vocab_size=270, verbose=False)
# tokenizer2.save("tokenizer2")
# enc2 = tokenizer2.encode(text)

tokenizer3 = BPETokenizer()
tokenizer3.train(text, vocab_size=270, verbose=False)
tokenizer3.save("tokenizer3")
enc3 = tokenizer3.encode(text)

print("Basic: ", enc1)
# print("Regex: ", enc2)
print("Heap:  ", enc3)

visual_compare(tokenizer1, tokenizer3, "for those who say they're for the theater")