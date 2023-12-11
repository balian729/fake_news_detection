from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch_geometric.nn import global_max_pool as gmp


class GCNFN(torch.nn.Module):
	"""
GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

Paper: Fake News Detection on Social Media using Geometric Deep Learning
Link: https://arxiv.org/pdf/1902.06673.pdf


Model Configurations:

Vanilla GCNFN: args.concat = False, args.feature = content
UPFD-GCNFN: args.concat = True, args.feature = spacy
"""

	def __init__(self, args):
		super(GCNFN, self).__init__()

		self.num_features = args.num_features
		self.num_classes = args.num_classes
		self.nhid = args.nhid
		self.concat = args.concat

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.fc1 = Linear(self.nhid * 2, self.nhid)

		if self.concat:
			self.fc0 = Linear(self.num_features, self.nhid)
			self.fc1 = Linear(self.nhid * 2, self.nhid)

		self.fc2 = Linear(self.nhid, self.num_classes)


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.fc0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.fc1(x))

		x = F.log_softmax(self.fc2(x), dim=-1)

		return x
	

class SimpleGNN(torch.nn.Module):
	def __init__(self, args):
		super(SimpleGNN, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.model = args.model
		self.concat = args.concat

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'graphsage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

	def forward(self, data):

		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

		x = F.log_softmax(self.lin2(x), dim=-1)

		return x