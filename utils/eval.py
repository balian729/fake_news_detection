import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, 
	accuracy_score, 
	recall_score, 
	precision_score, 
	roc_auc_score, average_precision_score
)


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch
	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

	assert len(log) == len(size_list)

	accuracy, f1, precision, recall = 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1 += f1_score(y, pred_y, average='binary') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)

	return [accuracy/data_size, f1/data_size, precision/data_size, recall/data_size, auc]

def compute_test(model, loader, device=torch.device('cpu'), verbose=False):
	model.eval()
	with torch.no_grad():
		loss_test = 0.0
		out_log = []
		for data in loader:
			data = data.to(device)
			out = model(data)
			y = data.y
			if verbose:
				print(F.softmax(out, dim=1).cpu().numpy())
			out_log.append([F.softmax(out, dim=1), y])
			loss_test += F.nll_loss(out, y).item()
	loss_test /= len(loader)
	res = eval_deep(out_log, loader)
	res.append(loss_test)
	return res