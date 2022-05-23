from swa_asr import swaASR
from swa_nmt import NMT
from dataset import iswltDataset
import os
from tqdm import tqdm


def evaluate():
    swa_asr = swaASR()
    swa_nmt = NMT(src_lang="sw", tar_lang="en")
    dataset = iswltDataset(split='valid')
    pred_swa_results = []
    pred_eng_results = []
    pred_eng_with_gold_swa_results = []
    for file_path, gold_swa, gold_eng in tqdm(dataset, miniters=1):
        pred_swa = swa_asr.predict(file_path)
        pred_eng = swa_nmt.predict(pred_swa)
        pred_eng_with_gold_swa = swa_nmt.predict(gold_swa)
        pred_swa_results.append(pred_swa)
        pred_eng_results.append(pred_eng)
        pred_eng_with_gold_swa_results.append(pred_eng_with_gold_swa)
    save_dir = 'outputs/twostage'
    with open(os.path.join(save_dir, 'pred_swa.txt'), 'w') as f:
        for sent in pred_swa_results:
            f.write(sent + '\n')
    with open(os.path.join(save_dir, 'pred_eng.txt'), 'w') as f:
        for sent in pred_eng_results:
            f.write(sent + '\n')
    with open(os.path.join(save_dir, 'pred_eng_with_gold_swa.txt'), 'w') as f:
        for sent in pred_eng_with_gold_swa_results:
            f.write(sent + '\n')


if __name__ == '__main__':
    evaluate()