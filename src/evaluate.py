import nltk
import os
from jiwer import wer, cer


def read_hypo_bleu(path):
    with open(path, 'r') as f:
        output = [line.strip().split(' ') for line in f.readlines()]
    return output


def read_ref_bleu(path):
    with open(path, 'r') as f:
        output = [[line.strip().split(' ')] for line in f.readlines()]
    return output


def cal_bleu(ref_path, hypo_path):
    refs = read_ref_bleu(ref_path)
    hypos = read_hypo_bleu(hypo_path)
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs, hypos)
    return bleu_score


def read_file_wer(path):
    with open(path, 'r') as f:
        output = [line.strip() for line in f.readlines()]
    return output


def cal_wer_cer(ref_path, hypo_path):
    refs = read_file_wer(ref_path)
    hypos = read_file_wer(hypo_path)
    wer_score = wer(refs, hypos)
    cer_score = cer(refs, hypos)
    return wer_score, cer_score
    

def main():
    output_dir = 'outputs/twostage/'
    score_file = os.path.join('results/', 'twostage_scores.txt')
    f = open(score_file, 'w')
    ref_path = os.path.join(output_dir, 'gold_eng.txt')
    hypo_path = os.path.join(output_dir, 'pred_eng_with_gold_swa.txt')
    bleu_score = cal_bleu(ref_path, hypo_path)
    f.write(f'BLEU score of Swahili-English translation on golden transcription: {bleu_score}\n')
    print(bleu_score)
    hypo_path = os.path.join(output_dir, 'pred_eng.txt')
    bleu_score = cal_bleu(ref_path, hypo_path)
    f.write(f'BLEU score of Swahili-English translation on ASR transcription: {bleu_score}\n')
    print(bleu_score)
    ref_path = os.path.join(output_dir, 'gold_swa.txt')
    hypo_path = os.path.join(output_dir, 'pred_swa.txt')
    wer_score, cer_score = cal_wer_cer(ref_path, hypo_path)
    f.write(f'WER of Swahili speech-to-text transcription: {wer_score}, CER: {cer_score}\n')
    print(wer_score)
    print(cer_score)


if __name__ == '__main__':
    main()
