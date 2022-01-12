import os
from os.path import exists
import matplotlib.pyplot as plt


def save_figures(history, dir):
    # Create folder to store figures (if not exist)
    if not exists(dir): os.makedirs(dir)

    # Features to visualize
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    labels = ['training' , 'validation']

    # Accuracy
    plt.plot(acc); plt.plot(val_acc)
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy'); plt.xlabel('Epoch')
    plt.legend(labels, loc='lower right')
    plt.savefig(dir + 'accuracy.jpg')
    plt.close()

    # Loss
    plt.plot(loss); plt.plot(val_loss)
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(labels, loc='upper right')
    plt.savefig(dir + 'loss.jpg')
    plt.close()