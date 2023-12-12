# **User Preference-aware Fake News Detection**



* [https://github.com/balian729/fake_news_detection](https://github.com/balian729/fake_news_detection)

## **Team members**


<table>
  <tr>
   <td><strong>Name</strong>
   </td>
   <td><strong>Email</strong>
   </td>
   <td><strong>Alias</strong>
   </td>
  </tr>
  <tr>
   <td>Elaman Turekulov
   </td>
   <td>e.turekulov@innopolis.university
   </td>
   <td>@bal1an
   </td>
  </tr>
  <tr>
   <td>Dmitrii Naumov
   </td>
   <td>d.naumov@innopolis.university
   </td>
   <td>@DmitryNau
   </td>
  </tr>
  <tr>
   <td>Albert Khazipov
   </td>
   <td>a.khazipov@innopolis.university
   </td>
   <td>@drxelt
   </td>
  </tr>
</table>


## **Problem statement**

In this day and age, when people get most of the news from the internet, where the sources cannot always be credible, the spread of fake news became a significant issue. With the exponential growth of social media platforms and online news sources, identifying and mitigating the impact of false information has become a daunting challenge. Traditional methods of fake news detection often fall short in handling the dynamic and evolving nature of deceptive content.

## **Solution description**

In this context, leveraging Graph Machine Learning (Graph ML) techniques for fake news detection presents a promising avenue. Graph ML enables the representation of relationships and interactions among entities, making it well-suited for capturing the intricate patterns of information propagation and source credibility in online networks.

In our project, we tried to develop a model that will be able to detect fake news and for this, we decided to use UPFD[1] dataset and compared the performance of different models on this dataset

## **Dataset**

The dataset consists of Twitter networks that propagate both fake and real news, constructed based on fact-check information from Politifact[2] and Gossipcop[3]. The retweet graphs for the news were initially obtained using FakeNewsNet[4]. The dataset's statistics are presented below:


<table>
  <tr>
   <td><strong>Data</strong>
   </td>
   <td><strong>#Graphs</strong>
   </td>
   <td><strong>#Fake News</strong>
   </td>
   <td><strong>#Total Nodes</strong>
   </td>
   <td><strong>#Total Edges</strong>
   </td>
   <td><strong>#Avg. Nodes per Graph</strong>
   </td>
  </tr>
  <tr>
   <td>Politifact
   </td>
   <td>314
   </td>
   <td>157
   </td>
   <td>41,054
   </td>
   <td>40,740
   </td>
   <td>131
   </td>
  </tr>
  <tr>
   <td>Gossipcop
   </td>
   <td>5464
   </td>
   <td>2732
   </td>
   <td>314,262
   </td>
   <td>308,798
   </td>
   <td>58
   </td>
  </tr>
</table>


In our case, we used only Politifact as our dataset, since Gossipcop was too large and required much more computational resources.

Please keep in mind that there are **four node feature types** in the dataset. The 768-dimensional BERT and 300-dimensional spaCy features are encoded using pretrained models from BERT[5] and spaCy[6] word2vec, respectively. Additionally, the 10-dimensional profile feature is derived from profiles of  Twitter accounts, and content feature that is a combination of spacy and profile features

## **What is Graph ML and PyG?**

So, what is Graph ML? Think of Graph ML as a way to teach computers about relationships and connections. In real-world problems, things are often connected or related to each other, like friends in a social network or molecules in chemistry. Graph ML helps computers understand and make predictions based on these connections. 

A graph consists of nodes (representing entities) and edges (representing relationships between entities). Each node and edge can have associated attributes. There are many different Graph ML models, and those of them that were used in this project, I am going to explain further down below.

And what is PyG? PyG is an acronym for PyTorch Geometric, which is an extension library for PyTorch specifically tailored for dealing with graph-structured data. It provides abstractions for handling graphs, creating graph datasets, and implementing various graph neural network layers.

Key features of PyG:



* **Data Handling**: PyG provides a `Data` class to manage node and edge attributes, making it easy to work with graph data. 
* **Graph Convolutional Layers**: PyG includes various graph convolutional layers, such as Graph Convolutional Networks (GCNs), GraphSAGE, etc., simplifying the implementation of GNNs. 
* **Dataset Loading**: PyG includes datasets for common benchmarks, simplifying the process of loading and working with well-known graph datasets.

## **Node feature types**

There are node four feature types that were used, each introducing their own set of features based on the selected approach



* **bert** - 768-dimensional feature type which is encoded using pretrained BERT
* **spacy** - 300-dimensional feature type which is encoded using pretrained spaCy word2vec, respectively
* **profile** - 10-dimensional feature type which represents data taken from Twitter accounts. This data include: verified status(binary), geo-spatial positioning(binary), the amount of followers and friends, date of account creation, etc.
* **content** - a combination of two other node feature types:  
    * 300-dimensional `spacy`, capturing semantic content.
    * 10-dimensional `profile`, providing additional user-specific information.

## **Models and techniques**

In our project, we evaluated four different models: 



* **GCN**
* **GAT**
* **GraphSAGE**
* **GCNFN [7]**

## **Metrics**

## **Training results**

## **Code snippets**

## **References**

[1] [https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR](https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR)

[2] [https://www.politifact.com/](https://www.politifact.com/)

[3] [https://www.gossipcop.com/](https://www.gossipcop.com/)

[4] [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

[5] [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

[6] [https://spacy.io/models/en#en_core_web_lg](https://spacy.io/models/en#en_core_web_lg)

[7] Monti, Federico & Frasca, Fabrizio & Eynard, Davide & Mannion, Damon & Bronstein, Michael. (2019). Fake News Detection on Social Media using Geometric Deep Learning, [https://doi.org/10.48550/arXiv.1902.06673](https://doi.org/10.48550/arXiv.1902.06673)
