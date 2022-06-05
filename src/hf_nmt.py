from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch


class NMT:
    def __init__(self, src_lang, tar_lang):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.tokenizer.src_lang = src_lang
        self.tar_lang = tar_lang
    
    def predict(self, swa_text):
        encoded_swa = self.tokenizer(swa_text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**encoded_swa, forced_bos_token_id=self.tokenizer.get_lang_id(self.tar_lang))
        eng_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return eng_text


if __name__ == '__main__':
    # model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # tokenizer.src_lang = "sw"
    swa_text = 'yeye ni maamufu kuni liko'
    # encoded_swa = tokenizer(swa_text, return_tensors="pt")
    # generated_tokens = model.generate(**encoded_swa, forced_bos_token_id=tokenizer.get_lang_id("en"))
    # eng_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    swa_nmt = NMT(src_lang="sw", tar_lang="en")
    eng_text = swa_nmt.predict(swa_text)
    print(eng_text)

