# The MSR-Video to Text dataset with clean annotations
![Forks](https://img.shields.io/github/forks/WingsBrokenAngel/MSR-VTT-DataCleaning
) ![Stars](https://img.shields.io/github/stars/WingsBrokenAngel/MSR-VTT-DataCleaning
) ![License](https://img.shields.io/github/license/WingsBrokenAngel/MSR-VTT-DataCleaning
) ![Issues](https://img.shields.io/github/issues/WingsBrokenAngel/MSR-VTT-DataCleaning
) 

This is the source code for *The MSR-Video to Text dataset with clean annotations*. 
We found that MSR-VTT dataset contains a lot of noisy annotations. 
After analyzing the data carefully, we put some efforts on cleaning the annotations. 
We retrained some models on the cleaned dataset and found experimental results improved compared to the previous models. 

## Requirements
1. Python 3.8
2. Jupyter Notebook
3. Hunspell

##  Information
`clean_process` is the folder for cleaning MSR-VTT dataset. 

## Reproduction of Results
1. Run Jupyter Notebook in the `clean_process`.
2. Please replace the input for the models in [Semantics-Assisted Video Captioning Model Trained with Scheduled Sampling Strategy
](https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning) and [Delving Deeper into the Decoder for Video Captioning
](https://github.com/WingsBrokenAngel/delving-deeper-into-the-decoder-for-video-captioning).
3. Train the new models.

## Links
- Cleaned dataset: [GoogleDrive](https://drive.google.com/file/d/1kVgaefASHM2GP4gZBNw90KcwGs3qZXWf/view?usp=sharing)   
- The paper on Arxiv: [The MSR-Video to Text dataset with clean annotations
](https://arxiv.org/abs/2102.06448)
- The published paper: [The MSR-Video to Text dataset with clean annotations
](https://www.sciencedirect.com/science/article/abs/pii/S107731422200159X)
- Dictionary download for Hunspell: [dictionaries](https://github.com/wooorm/dictionaries)

## Citation
```
Haoran Chen, Jianmin Li, Simone Frintrop, Xiaolin Hu,
The MSR-Video to Text dataset with clean annotations,
Computer Vision and Image Understanding,
Volume 225,
2022,
103581,
ISSN 1077-3142,
https://doi.org/10.1016/j.cviu.2022.103581.
(https://www.sciencedirect.com/science/article/pii/S107731422200159X)
Abstract: Video captioning automatically generates short descriptions of the video content, usually in form of a single sentence. Many methods have been proposed for solving this task. A large dataset called MSR Video to Text (MSR-VTT) is often used as the benchmark dataset for testing the performance of the methods. However, we found that the human annotations, i.e., the descriptions of video contents in the dataset are quite noisy, e.g., there are many duplicate captions and many captions contain grammatical problems. These problems may pose difficulties to video captioning models for learning underlying patterns. We cleaned the MSR-VTT annotations by removing these problems, then tested several typical video captioning models on the cleaned dataset. Experimental results showed that data cleaning boosted the performances of the models measured by popular quantitative metrics. We recruited subjects to evaluate the results of a model trained on the original and cleaned datasets. The human behavior experiment demonstrated that trained on the cleaned dataset, the model generated captions that were more coherent and more relevant to the contents of the video clips.
Keywords: MSR-VTT dataset; Data cleaning; Data analysis; Video captioning
```