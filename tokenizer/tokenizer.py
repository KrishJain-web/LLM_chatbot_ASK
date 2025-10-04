! pip3 install tiktoken
import importlib
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = input()
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
