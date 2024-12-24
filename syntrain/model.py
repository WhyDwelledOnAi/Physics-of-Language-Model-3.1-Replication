from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

# define a 12-layer, 768-hidden, 12-heads, 110M parameters model
config = LlamaConfig.from_json_file("params.json")
tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")
config.vocab_size = tokenizer.vocab_size
model = LlamaForCausalLM(config)


