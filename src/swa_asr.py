import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class swaASR:
    def __init__(self, device, processor, model, resampler):
        self.device = device
        self.processor = processor
        self.model = model
        self.resampler = resampler
    
    def load_file_to_data(self, file):
        batch = {}
        speech, _ = torchaudio.load(file)
        batch["speech"] = self.resampler.forward(speech.squeeze(0)).numpy()
        batch["sampling_rate"] = self.resampler.new_freq
        return batch

    def predict(self, data):
        features = self.processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)


from googletrans import Translator
translator = Translator()


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # processor = Wav2Vec2Processor.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw")
    # model = Wav2Vec2ForCTC.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw").to(device)
    # resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
    # predict(load_file_to_data('./demo.wav'))
    pass