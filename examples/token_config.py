from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
print(tok.decode([1,29871,13,13], clean_up_tokenization_spaces=False))
