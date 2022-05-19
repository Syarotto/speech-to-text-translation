import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])  # file is the path and speech is a vector
    batch["speech"] = speech
    return batch

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")

# translate English Speech To French Text
generated_ids = model.generate(
    input_ids=inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"]
)
translation_fr = processor.batch_decode(generated_ids)

# translate English Speech To German Text
generated_ids = model.generate(
    input_ids=inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    forced_bos_token_id=processor.tokenizer.lang_code_to_id["de"]
)
translation_de = processor.batch_decode(generated_ids, skip_special_tokens=True)


# import torch
# from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
# from datasets import load_dataset

# model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
# processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")


# ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
# generated_ids = model.generate(
#     inputs["input_features"],
#     attention_mask=inputs["attention_mask"],
#     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
# )

# transcription = processor.batch_decode(generated_ids)