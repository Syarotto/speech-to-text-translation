import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class swaASR:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw")
        self.model = Wav2Vec2ForCTC.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw").to(self.device)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
    
    def load_file_to_data(self, data_path):
        batch = {}
        speech, _ = torchaudio.load(data_path)
        batch["speech"] = self.resampler.forward(speech.squeeze(0)).numpy()
        batch["sampling_rate"] = self.resampler.new_freq
        return batch

    def predict(self, data_path):
        data = self.load_file_to_data(data_path)
        features = self.processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        swa_text = self.processor.batch_decode(pred_ids)[0]
        return swa_text


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # processor = Wav2Vec2Processor.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw")
    # model = Wav2Vec2ForCTC.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw").to(device)
    # resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
    # predict(load_file_to_data('./demo.wav'))
    swa_asr = swaASR()
    data_path = 'data/swa-eng/train/wav/1/0b710f1ba2e5dff867bd617678258c39__1573474526.6732.wav'
    text = swa_asr.predict(data_path)
    print(text)