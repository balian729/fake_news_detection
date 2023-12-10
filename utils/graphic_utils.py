import matplotlib.pyplot as plt


def plot_figure(y1, y2, y_label="Accuracy", y1_label="Train loss", y2_label="Val loss", file_name="Train_val_loss.png", epochs=60):
	x = range(epochs)
	plt.plot(x, y1, label=y1_label)
	plt.plot(x, y2, label=y2_label)
	plt.legend(loc="upper left")
	plt.ylabel(y_label)
	plt.xlabel('Epochs')
	plt.savefig('reports/figures/'+file_name)
	plt.clf()