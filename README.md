# Fake news detection
This project incorporates Graph Machine Learning techniques for fake news detection. 

## Dataset
[Dataset](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/upfd.py) used for our project can be downloaded directly from PyTorch-Geometric library

The dataset consists of Twitter networks that propagate both fake and real news, constructed based on fact-check information from Politifact and Gossipcop. The retweet graphs for the news were initially obtained using FakeNewsNet. 

Please keep in mind that there are four node feature types in the dataset. The 768-dimensional BERT and 300-dimensional spaCy features are encoded using pretrained models from BERT[5] and spaCy[6] word2vec, respectively. Additionally, the 10-dimensional profile feature is derived from profiles of  Twitter accounts, and content feature that is a combination of spacy and profile features

## Models
For evaluation, we used four models that include:
* GCN
* GAT
* GraphSAGE
* GCNFN 

## Basic commands
```
!git lfs clone https://github.com/balian729/fake_news_detection.git
!cd fake_news_detection
!pip install -r requirements.txt
!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
!pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
!pip install torch-geometric
```
## Train model
```
python models/train.py
```
