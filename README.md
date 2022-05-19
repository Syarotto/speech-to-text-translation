# speech-to-text-translation

## Installation
This system is tested on `python==3.8.0`, and a list of required packages can be found at [requirements.txt](https://github.com/Syarotto/hyperpartisan_news_dection/blob/main/requirements.txt). It is recommended to use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up the environment. With `conda` installed, you may run: 

```
conda create -n 575c python==3.8.0
conda activate 575c
pip install -r requirements.txt
conda install -c conda-forge libsndfile
```

The endangered language parallel dataset LORELEI can be accessed on the cluster by:
```
newgrp lorelei
cd /corpora/LORELEI/LDC2017E64_LORELEI_Swahili_Representative_Language_Pack_Translation_Annotation_Grammar_Lexicon_and_Tools_V1.0
```

References:
- [End-to-End_Speech-to-Text_Translation](https://github.com/Shivam0712/End-to-End_Speech-to-Text_Translation)
- [Huggingface/speech-to-text](https://huggingface.co/docs/transformers/model_doc/speech_to_text)
- [fairseq/speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text)
