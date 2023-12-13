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



### **GCN**

GCN stands for Graph Convolutional Network, and it's a type of neural network architecture designed to work with graph-structured data.

#### **Graph Convolution Operation:**
**Transformation of Node Features:** 
  - Each node is associated with a feature vector. In the case of the first layer, these are the initial node features. 
  - A weight matrix is applied to these features, similar to the weight matrices in traditional neural network layers.

$Z = \sigma(A \cdot X \cdot W_1)$

where:
  - $X$ is the input node features matrix.
  - $W_1$ is the weight matrix for the first layer.
  - $A$ is the adjacency matrix.
  - $\sigma$ is the activation function.

**Aggregation of Neighboring Information:** 

* The convolutional operation aggregates information from neighboring nodes. 
* This is achieved by multiplying the adjacency matrix A with the transformed node features Z.

$Z' = \sigma(A \cdot Z \cdot W_2)$

where:
 - $W_2$ is the weight matrix for the second layer.

**Output:**
The result $Z'$ represents the new features of each node after considering information from its neighbors.

#### **Training**
Training a GCN involves optimizing the weights $W_1$ and $W_2$ to minimize a specific loss function for the task at hand. Commonly used loss functions include cross-entropy for classification tasks or mean squared error for regression tasks.

### **GAT**

The Graph Attention Network (GAT) is a type of neural network architecture designed for graph-structured data. In the context of GAT, a graph is represented as a set of nodes and edges. Each node in the graph corresponds to an entity, and edges represent relationships or connections between entities.

The core idea behind GAT is the use of attention mechanisms. Attention mechanisms allow the model to focus on different parts of the input when making predictions, mimicking the human attention process. GAT employs a self-attention mechanism, where each node can attend to its neighbors with different attention weights. This allows the model to assign different importance to different neighbors during information aggregation.

GAT computes a node's representation by aggregating information from its neighbors. The aggregation is done using attention weights, which are learned during training. The node representations are computed as a weighted sum of the neighboring node features, where the attention weights are determined by the compatibility between the node and its neighbors.

GAT extends the basic attention mechanism by using multiple attention heads. Each attention head learns a different set of attention weights, capturing different aspects of the relationships between nodes. The outputs of multiple attention heads are concatenated or averaged to produce the final node representations.

Given a graph with nodes represented by feature vectors, the GAT operation for computing the output features of a node involves computing attention coefficients, applying them to the corresponding neighbor node features, and aggregating the results.

### **GraphSAGE**

GraphSAGE, which stands for Graph Sample and Aggregated (SAGE), is a graph neural network (GNN) architecture designed for learning node embeddings in a graph. The main objective of GraphSAGE is to learn a function that generates embeddings for nodes in a graph in an inductive manner. In other words, it aims to produce embeddings for nodes that were not seen during training, enabling the model to generalize to unseen nodes.


#### **Graph Convolution**:
GraphSAGE utilizes a graph convolutional operation to aggregate information from a node's local neighborhood. Unlike traditional convolutional operations in image processing, graph convolutional operations consider the graph structure.


#### **Aggregation Strategy**

The key innovation of GraphSAGE lies in its sampling and aggregation strategy. Rather than aggregating information from all neighboring nodes, which could be computationally expensive for large graphs, GraphSAGE samples a fixed-size neighborhood around each node.

1. **Sampling**: For each node, a fixed-size sample of its neighbors is chosen. The sampling strategy can be random, but it's often guided by a more structured approach like sampling neighbors from a fixed radius or using a specific sampling algorithm.

2. **Aggregation**: The sampled node features are then aggregated to create a representative vector for the node. This aggregation process is typically a differentiable operation, and it allows the model to learn how to combine information from the neighborhood.

#### **GraphSAGE Architecture:**

The GraphSAGE model typically consists of multiple layers of the sampling and aggregation process. The output of each layer is used as the input for the next layer. The final layer's output represents the node embeddings.

### **GCNFN [7]**
GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

## **Metrics**
To evaluate the performance of our models, we used four different metrics: These metrics being accuracy, AUC, F1 score and loss. Below you can see how each metric is calculated and the reasoning behind its selection.

### Accuracy
As you know, accuracy is one of the most basic metrics and can be simply calculated as (# of correct predictions)/(total prediction number). The reason behind choosing it as it is the most simple metric that allows any person to clearly understand the performance of our models. It is also worth mentioning that our dataset is fairly balanced, with fake news making up ~40% of the dataset, and real news being ~60%, respectively.
### AUC
As it is known AUC stands for Area Under ROC Curve, ROC here meaning Receiver Operating Characteristic, which is sometimes called an “error curve”. We decided to chose that metric as it is excellent for binary classification and suits a slight imbalance in our data
### F1 score
F1 score is one of the most used evaluation metrics as it integrates precision and recall into a single metric to gain a better understanding of model performance. It’s calculated as follows:
F1 = (2 * _Precision_ * _Recall_)(_Precision_ + _Recall_)



## **References**

[1] [https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR](https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR)

[2] [https://www.politifact.com/](https://www.politifact.com/)

[3] [https://www.gossipcop.com/](https://www.gossipcop.com/)

[4] [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

[5] [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

[6] [https://spacy.io/models/en#en_core_web_lg](https://spacy.io/models/en#en_core_web_lg)

[7] Monti, Federico & Frasca, Fabrizio & Eynard, Davide & Mannion, Damon & Bronstein, Michael. (2019). Fake News Detection on Social Media using Geometric Deep Learning, [https://doi.org/10.48550/arXiv.1902.06673](https://doi.org/10.48550/arXiv.1902.06673)
