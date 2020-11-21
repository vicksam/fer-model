import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def display_majority_predictions(images,
                                 labels,
                                 class_mapping,
                                 label_ps = None):
    '''Plots majority predictions for 24 images.
    If it does not receive predicted labels, it simply prints images with true labels.

    Args:
        images(ndarray): a batch of (48, 48, 1) images
        labels(ndarray): a batch of basis vector labels
        class_mapping(dict(int, string)): a mapping of class labels to class names
        label_ps(ndarray) : a batch of predicted basis vector labels

    Returns: None
    '''
    # Convert labels to integers
    labels = labels.argmax(axis = 1)

    plt.figure(figsize = (12, 8))
    for i in range(24):
        ax = plt.subplot(4, 6, i + 1)
        plt.imshow(images[i].squeeze(), cmap = 'gray')

        # Print label
        if label_ps is not None:
            color = 'green' if np.argmax(label_ps[i]) == labels[i] else 'red'
            label = np.argmax(label_ps[i])
            plt.title(class_mapping[label], color = color)
        else:
            label = labels[i]
            plt.title(class_mapping[label])

        plt.axis("off")

def display_cross_entropy_predictions(images,
                                      labels,
                                      class_mapping,
                                      label_ps = None):
    '''Plots cross entropy predictions for 12 images.
    If it does not receive predicted labels, it simply prints images with true
    labels distributions.

    Args:
        images(ndarray): a batch of (48, 48, 1) images
        labels(ndarray): a batch of (10, ) label distributions
        class_mapping(dict(int, string)): a mapping of class labels to class names
        label_ps(ndarray) : a batch of (10,) predicted label distributions

    Returns: None
    '''
    class_names = class_mapping.values()

    plt.figure(figsize = (10.75, 13))
    for i in range(0, 24, 2):
        index = int(i / 2)
        image = images[index]
        label = labels[index]

        ## Draw a single plot consisting of two subplots (image + distribution)
        ax1 = plt.subplot(6, 4, i + 1)
        ax2 = plt.subplot(6, 4, i + 2)
        ax1.imshow(image.squeeze(), cmap = 'gray')
        ax1.axis('off')
        y_ticks = np.arange(len(class_names))

        # Draw per two bars if predictions are given
        if label_ps is not None:
            label = pd.DataFrame({ 'true': label, 'predicted': label_ps[i] },
                                 index = class_names)
            label.plot.barh(ax = ax2)
            ax2.legend(prop = {'size':8}, loc = 'lower right')
        else:
            ax2.barh(y_ticks, label)

        ax2.set_aspect(0.12)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(class_names)
        ax2.set_xlim(0, 1)
        ax2.invert_yaxis()  # Labels read top-to-bottom

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    '''Plots training accuracy and loss.

    Args:
        history(Object): history object resulting from training

    Returns: None
    '''
    training_accuracy = history['accuracy']
    validation_accuracy = history['val_accuracy']

    training_loss = history['loss']
    validation_loss = history['val_loss']

    epochs_range = range(len(training_accuracy))

    plt.figure(figsize = (8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label = 'Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label = 'Training Loss')
    plt.plot(epochs_range, validation_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_confusion_matrix(confusion_matrix, class_names):
    '''Plots confusion matrix.

    Args:
        confusion_matrix(list of lists)
        class_names(list): Class names which will be plotted along axes

    Returns: None
    '''
    sns.set(color_codes = True)
    plt.figure(1, figsize = (10, 7))

    plt.title('Confusion Matrix')

    sns.set(font_scale = 1.)
    ax = sns.heatmap(data = confusion_matrix,
                     annot = True,
                     cmap = "YlGnBu",
                     cbar_kws = {'label': 'Scale'},
                     fmt = '4d')

    ax.set_xticklabels(class_names, rotation = -30)
    ax.set_yticklabels(class_names, rotation = -30)

    ax.set(ylabel = "True Label", xlabel = "Predicted Label")

    plt.show()
