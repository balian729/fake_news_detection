import argparse
import sys
from tqdm import trange
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv
import torch
import pandas as pd
from tabulate import tabulate
import numpy as np

from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from utils.dataloader import DropEdge, ToUndirected, FNNDataset
from models.architectures import GCNFN, SimpleGNN
from utils.train import train
from utils.eval import compute_test
	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# original model parameters
	parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')

	# hyper-parameters
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--train_size', type=float, default=0.6)
	parser.add_argument('--val_size', type=float, default=0.2)
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
	parser.add_argument('--nhid', type=int, default=128, help='hidden size')
	parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
	parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
	parser.add_argument('--feature', type=str, default='spacy', help='feature type, [profile, spacy, bert, content]')
	parser.add_argument('--model', type=str, default='gcnfn', help='model type, [gcnfn, gcn, gat, graphsage]')

	args = parser.parse_args()
	SEED = 42
	torch.manual_seed(SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(SEED)

	dataset = FNNDataset(root='data', feature=args.feature, empty=False, name='politifact', transform=ToUndirected())

	args.num_classes = dataset.num_classes
	args.num_features = dataset.num_features

	print(args)

	num_training = int(len(dataset) * args.train_size)
	num_val = int(len(dataset) * args.val_size)
	num_test = len(dataset) - (num_training + num_val)

	training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])


	train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
	
	if args.model == 'gcnfn':
		model = GCNFN(args=args)
	else:
		model = SimpleGNN(args)
	model = model.to(args.device)

	best_results = train(model, train_loader, val_loader, args)
	best_results_df = pd.DataFrame([best_results])
	print("Validation Results")
	print(tabulate(best_results_df, headers='keys', tablefmt='psql'))

	model.load_state_dict(torch.load(f'./models/weights/{args.model}_best.pth'))
	test_results = compute_test(model, test_loader, args.device, verbose=False)
	test_results_df = pd.DataFrame([test_results], columns=["Accuracy", "F1", "Precision", "Recall", "AUC", "Loss"])
	print("Test Results")
	print(tabulate(test_results_df, headers='keys', tablefmt='psql'))