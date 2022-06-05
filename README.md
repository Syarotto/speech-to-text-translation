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
To set up the Google Cloud Speech service, follow the instructions on [Install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk#linux) and [Setting up a Python development environment](https://cloud.google.com/python/docs/setup#linux), then install `google-cloud-speech` through `pip`. You also need to generate the credential key given [Getting started with authentication](https://cloud.google.com/docs/authentication/getting-started), where the service account can be found in [this link](console.cloud.google.com/iam-admin/serviceaccounts). Name the credential json file as `credential.json` and place it under the root directory of this project. 

## Resources:
- [End-to-End_Speech-to-Text_Translation](https://github.com/Shivam0712/End-to-End_Speech-to-Text_Translation)
- [Huggingface/speech-to-text](https://huggingface.co/docs/transformers/model_doc/speech_to_text)
- [fairseq/speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text)
- [alokmatta/wav2vec2-large-xlsr-53-sw](https://huggingface.co/alokmatta/wav2vec2-large-xlsr-53-sw)
- [MUST-C](https://ict.fbk.eu/must-c-releases/)
- [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)
- [IWSLT 2021 Swahili](https://iwslt.org/2021/low-resource)
- [Google Cloud Speech](https://cloud.google.com/speech-to-text)