from dataset import iwsltDataset
import os
from tqdm import tqdm


def evaluate_hf_pipeline():
    from hf_asr import swaASR
    from hf_nmt import NMT
    swa_asr = swaASR()
    swa_nmt = NMT(src_lang="sw", tar_lang="en")
    dataset = iwsltDataset(split='valid')
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


def evaluate_googleASR():
    from google_asr import transcribe_file
    dataset = iwsltDataset(split='valid')
    save_dir = 'outputs/GoogleCloud-KE'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for file_path, gold_swa, gold_eng in tqdm(dataset, miniters=1):
        try:
            pred_swa = transcribe_file(file_path)
        except:
            pred_swa = ''
        with open(os.path.join(save_dir, 'pred_swa.txt'), 'a') as f:
                f.write(pred_swa + '\n')


def evaluate_MT():
    from hf_nmt import NMT
    swa_nmt = NMT(src_lang="sw", tar_lang="en")
    pred_eng_results = []
    save_dir = 'outputs/GoogleCloud-KE'

    with open(os.path.join(save_dir, 'pred_swa.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pred_eng = swa_nmt.predict(line)
            pred_eng_results.append(pred_eng)
    
    with open(os.path.join(save_dir, 'pred_eng.txt'), 'w') as f:
        for sent in pred_eng_results:
            f.write(sent + '\n')


if __name__ == '__main__':
    evaluate_MT()