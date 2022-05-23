from torch.utils.data import IterableDataset
from utils import get_files_from_yaml, get_transcription_from_file
from tqdm import tqdm


class iswltDataset(IterableDataset):
    def __init__(self, split):
        if split == 'train':
            self.data_path = 'data/swa-eng/train/'
        elif split == 'valid':
            self.data_path = 'data/swa-eng/valid/'
        else:
            raise Exception("Error in split option: train, valid")

    def __len__(self):
        return len(get_files_from_yaml(self.data_path))

    def __iter__(self):
        swa_texts = get_transcription_from_file(self.data_path, 'swa')
        eng_texts = get_transcription_from_file(self.data_path, "eng")
        file_list = get_files_from_yaml(self.data_path)
        assert len(swa_texts) == len(eng_texts), 'English and Swahili transcirptions not matching'
        assert len(swa_texts) == len(file_list), 'Swahili transcriptions and audio files not matching'
        for file_path, swa_text, eng_text in zip(file_list, swa_texts, eng_texts):
            yield file_path, swa_text, eng_text


if __name__ == '__main__':
    dataset = iswltDataset(split='train')
    for file_path, swa_text, eng_text in tqdm(dataset, miniters=1):
        pass