# NorBERT
This repository contains in-house code used in training and evaluating NorBERT: a large-scale Transformer-based language model for Norwegian. The model was trained by the [Language Technology Group](https://www.mn.uio.no/ifi/english/research/groups/ltg/) at the University of Oslo. The computations were performed on resources provided by UNINETT Sigma2 - the National Infrastructure for High Performance Computing and Data Storage in Norway.

For general training, [BERT For TensorFlow from NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) was used. 
We made minor changes to their code, see the `patches_for_NVIDIA_BERT` subdirectory. 

NorBERT training was conducted as a part of the NorLM project. Check this paper for more details:

_Andrey Kutuzov, Jeremy Barnes, Erik Velldal, Lilja Øvrelid, Stephan Oepen. [Large-Scale Contextualised Language Modelling for Norwegian](https://www.aclweb.org/anthology/2021.nodalida-main.4/), NoDaLiDa'21 (2021)_

- [Read about NorBERT](http://norlm.nlpl.eu)
- [Download NorBERT from our repository](http://vectors.nlpl.eu/repository/216.zip) or [from HuggingFace](https://huggingface.co/ltgoslo/norbert)


![Logo](https://github.com/ltgoslo/NorBERT/raw/main/Norbert.png)
