import matplotlib.pyplot as plt
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, f1_score


def class_results(res):
    """
    Computes and prints class-wise and overall test accuracy, precision, recall, and F1-score.

    :param res: Dictionary containing predicted and true labels.
    """
    preds = res['preds']
    labels = res['labels']
    class_res = {c: [0, 0] for c in labels}
    corrects = 0

    for i, label in enumerate(labels):
        if preds[i] == labels[i]:
            class_res[label][0] += 1
            corrects += 1
        class_res[label][1] += 1

    class_res = dict(sorted(class_res.items(), key=lambda x: x[0]))

    for k, v in class_res.items():
        print(f'Test accuracy of {id2name[k]}: {v[0]}/{v[1]} -> {round(v[0] / v[1] * 100, 2)}%')
    print()
    print(f'Test accuracy: {round(corrects / len(preds) * 100, 2)}%')
    print(f'Precision: {round(precision_score(labels, preds) * 100, 2)}%')
    print(f'Recall: {round(recall_score(labels, preds) * 100, 2)}%')
    print(f'F1: {round(f1_score(labels, preds) * 100, 2)}%')


def plot_results(history, axis, fold):
    """
    Plots training and validation loss and accuracy.

    :param history: Dictionary containing loss and accuracy history.
    :param axis: Matplotlib axis for plotting.
    :param fold: Fold index for cross-validation.
    """
    loss_train, loss_valid = history['train']['loss'], history['valid']['loss']
    acc_train, acc_valid = history['train']['accuracy'], history['valid']['accuracy']
    epochs = range(len(loss_train))

    axis[fold, 0].plot(epochs, loss_train, label='Train')
    axis[fold, 0].plot(epochs, loss_valid, label='Valid')
    axis[fold, 0].set_xlabel("Epochs")
    axis[fold, 0].set_ylabel("CrossEntropy Loss")
    axis[fold, 0].set_ylim(0, max(max(loss_valid), max(loss_train)) + 0.1)
    axis[fold, 0].set_title('CrossEntropy Loss')
    axis[fold, 0].grid(linewidth=1.5, alpha=0.25, linestyle="dashed")
    axis[fold, 0].legend()

    axis[fold, 1].plot(epochs, acc_train, label='Train')
    axis[fold, 1].plot(epochs, acc_valid, label='Valid')
    axis[fold, 1].set_xlabel("Epochs")
    axis[fold, 1].set_ylabel("Accuracy")
    axis[fold, 1].set_ylim(0, 1)
    axis[fold, 1].set_title('Accuracy')
    axis[fold, 1].grid(linewidth=1.5, alpha=0.25, linestyle="dashed")
    axis[fold, 1].legend()


def denormalize(images, means, stds):
    """
    Denormalizes a batch of images using given means and standard deviations.
    """
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_images(img, label, classes):
    """
    Displays a batch of images with corresponding labels.
    """
    plt.figure(figsize=[20, 14])
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        x = img[i].permute((1, 2, 0))
        plt.imshow(x, cmap='gray')
        plt.title(classes[label[i]])
        plt.axis("off")
    plt.show()


class EarlyStopper:
    """
    Implements early stopping mechanism based on validation loss.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def oversampling(index, y, seed=0, sampling_strategy=1):
    """
    Performs random oversampling to balance class distribution.

    :param index: Array of indices corresponding to dataset samples.
    :param y: Array of class labels.
    :param seed: Random seed for reproducibility.
    :param sampling_strategy: Oversampling ratio.
    :return: Resampled indices and corresponding labels.
    """
    index_2d, y_2d = index.reshape(-1, 1), y.reshape(-1, 1)
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
    index_resampled, y_resampled = ros.fit_resample(index_2d, y_2d)
    return index_resampled.flatten(), y_resampled.flatten()
