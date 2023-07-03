# NorBERT
This repository contains in-house code used in training and evaluating NorBERT-1 and NorBERT-2: large-scale Transformer-based language models for Norwegian.
The models were trained by the [Language Technology Group](https://www.mn.uio.no/ifi/english/research/groups/ltg/) at the University of Oslo. 
The computations were performed on resources provided by UNINETT Sigma2 - the National Infrastructure for High Performance Computing and Data Storage in Norway.

For most of the training, [BERT For TensorFlow from NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) was used.
We made minor changes to their code, see the `patches_for_NVIDIA_BERT` subdirectory. 

NorBERT models training was conducted as a part of the NorLM project. Check this paper for more details:

_Andrey Kutuzov, Jeremy Barnes, Erik Velldal, Lilja Øvrelid, Stephan Oepen. [Large-Scale Contextualised Language Modelling for Norwegian](https://www.aclweb.org/anthology/2021.nodalida-main.4/), NoDaLiDa'21 (2021)_

- [Read about NorBERT](http://norlm.nlpl.eu)
- [Download NorBERT-1 from our repository](http://vectors.nlpl.eu/repository/20/216.zip) or [from HuggingFace](https://huggingface.co/ltgoslo/norbert)
- [Download NorBERT-2 from our repository](http://vectors.nlpl.eu/repository/20/221.zip) or [from HuggingFace](https://huggingface.co/ltgoslo/norbert2)

## NorBERT-3
In 2023, we released a new family of *NorBERT-3* language models for Norwegian. In general, we now recommend using these models:

- [NorBERT 3 xs](https://huggingface.co/ltg/norbert3-xs) (15M parameters)
- [NorBERT 3 small](https://huggingface.co/ltg/norbert3-small) (40M parameters)
- [NorBERT 3 base](https://huggingface.co/ltg/norbert3-base) (123M parameters)
- [NorBERT 3 large](https://huggingface.co/ltg/norbert3-large) (323M parameters)

NorBERT-3 is described in detail in this paper:
[NorBench – A Benchmark for Norwegian Language Models](https://aclanthology.org/2023.nodalida-1.61/) (Samuel et al., NoDaLiDa 2023)


![Logo](https://github.com/ltgoslo/NorBERT/raw/main/Norbert.png)
