import argparse
import sys
from tqdm import trange
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv

from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from utils.dataloader import *
from utils.graphic_utils import plot_figure
from utils.eval import eval_deep, compute_test
from models.architectures import GCNFN, SimpleGNN

def train(model, train_loader, val_loader, args):
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model.train()

	loop = trange(args.epochs)

	train_loss_list = []
	val_loss_list = []

	train_acc_list = []
	val_acc_list = []

	train_auc_list = []
	val_auc_list = []

	for _ in loop:
		out_log = []
		loss_train = 0.0
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			data = data.to(args.device)
			out = model(data)
			y = data.y
			loss = F.nll_loss(out, y)
			loss.backward()
			optimizer.step()
			loss_train += loss.item()
			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(model, val_loader, args.device)
		loop.set_postfix(
			loss_train=loss_train,
			acc_train=acc_train,
			loss_val=loss_val,
			acc_val=acc_val
		)
		loss_train /= len(train_loader)
		train_loss_list.append(loss_train)
		val_loss_list.append(loss_val)

		train_acc_list.append(acc_train)
		val_acc_list.append(acc_val)

		train_auc_list.append(auc_train)
		val_auc_list.append(auc_val)
	plot_figure(y_label="Loss", 
			 y1=train_loss_list, 
			 y2=val_loss_list, 
			 epochs=args.epochs, 
			 model_name=args.model,
			 file_name="Train_val_loss.png")
	plot_figure(
			 y1=train_acc_list, 
			 y2=val_acc_list,
			 y_label="Accuracy",
			 y1_label="Train accuracy", 
			 y2_label="Val accuracy", 
			 file_name="Train_val_acc.png",
			 epochs=args.epochs,
			 model_name=args.model)
	plot_figure(
			 y1=train_auc_list,
			 y2=val_auc_list,
			 y_label="AUC",
			 y1_label="Train AUC",
			 y2_label="Val AUC",
			 file_name="Train_val_auc",
			 epochs=args.epochs,
			 model_name=args.model)
	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# original model parameters
	parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

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

	train(model, train_loader, val_loader, args)

	# Model training
	
	[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(model, test_loader, args.device, verbose=False)
	print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}')
	print(f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')