from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

sentences=["Today is my birthday",
            "how you doing?",
            "I love watching people surf.",
            "Exercise is good for people's health.",
            "Interesting novel!",
            "Haha! You are hilarious."]


vocab_size=10000
encoding= [one_hot(sentence,vocab_size) for sentence in sentences]
print("Encoding-",encoding)

sentence_length=10
padded_encoding = pad_sequences(encoding,padding="pre",maxlen=sentence_length)
print("Padded Encoding-",padded_encoding)


dim=10
model=Sequential()
model.add(Embedding(vocab_size,dim,input_length=sentence_length))
model.compile('adam','mse')
print(model.summary())


print(model.predict(padded_encoding))

