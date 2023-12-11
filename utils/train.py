import torch
from tqdm import trange
import torch.nn.functional as F

from utils.graphic_utils import plot_figure
from utils.eval import eval_deep, compute_test


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
	
	train_f1_micro_list = []
	val_f1_micro_list = []

	train_f1_macro_list = []
	val_f1_macro_list = []

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
		acc_train, f1_macro_train, f1_micro_train, precision_train, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, f1_macro_val, f1_micro_val, _, recall_val, auc_val, _], loss_val = compute_test(model, val_loader, args.device)
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

		train_f1_macro_list.append(f1_macro_train)
		val_f1_macro_list.append(f1_macro_val)

		train_f1_micro_list.append(f1_micro_train)
		val_f1_micro_list.append(f1_micro_val)

	plot_figure(y_label="Loss", 
			 y1=train_loss_list, 
			 y2=val_loss_list, 
			 epochs=args.epochs, 
			 model_name=args.model,
			 feature=args.feature,
			 file_name="Train_val_loss.png")
	plot_figure(
			 y1=train_acc_list, 
			 y2=val_acc_list,
			 y_label="Accuracy",
			 y1_label="Train accuracy", 
			 y2_label="Val accuracy", 
			 file_name="Train_val_acc.png",
			 epochs=args.epochs,
			 feature=args.feature,
			 model_name=args.model)
	plot_figure(
			 y1=train_auc_list,
			 y2=val_auc_list,
			 y_label="AUC",
			 y1_label="Train AUC",
			 y2_label="Val AUC",
			 file_name="Train_val_auc",
			 epochs=args.epochs,
			 feature=args.feature,
			 model_name=args.model)
	plot_figure(
			 y1=train_f1_macro_list,
			 y2=val_f1_macro_list,
			 y_label="F1 Macro",
			 y1_label="Train f1 macro",
			 y2_label="Val f1 macro",
			 file_name="Train_val_f1_macro",
			 epochs=args.epochs,
			 feature=args.feature,
			 model_name=args.model)
	plot_figure(
			 y1=train_f1_micro_list,
			 y2=val_f1_micro_list,
			 y_label="F1 Micro",
			 y1_label="Train f1 micro",
			 y2_label="Val f1 micro",
			 file_name="Train_val_f1_micro",
			 epochs=args.epochs,
			 feature=args.feature,
			 model_name=args.model)