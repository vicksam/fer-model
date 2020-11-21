from . import outliers_processing as op
from .model_class.DataPipelineParams import DataPipelineParams
from .model_class.DataPipelineParams import Augmentation, Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np


FER_CLASS_MAPPING = {
    0 : 'anger',
    1 : 'disgust',
    2 : 'fear',
    3 : 'happiness',
    4 : 'sadness',
    5 : 'surprise',
    6 : 'neutral'
}

FER_PLUS_CLASS_MAPPING = {
    0 : 'neutral',
    1 : 'happiness',
    2 : 'surprise',
    3 : 'sadness',
    4 : 'anger',
    5 : 'disgust',
    6 : 'fear',
    7 : 'contempt'
}

COLUMN_NAMES = ['dataset', 'image', 'fer_code', 'neutral', 'happiness', \
'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', \
'no-face']

IMG_SHAPE = 48


def get_data_pipeline(dataset_df,
                      params,
                      shuffle = False):
    '''Get data pipeline that is ready for training. Apply color normalization
    and optionally, data augmentation.

    Args:
        dataset_df(dataframe): specific dataset (train, valid or test) loaded
                               from a unified dataset created using
                               dataset.get_dataset_dict()
        params(DataPipelineParams): wrapper object with pipeline parameters
        shuffle(boolean): indicates whether data points should be shuffled
                          (indicates if data is training data)

    Returns: an iterator yielding tuples of ndarrays (x, y). If non-zero value
             for sample_weights_threshold is given, it will return
             (x, y, sample_weights)
    '''
    # Get image generator that applies proper augmentation and color normalization
    image_gen = _get_image_generator(params.augmentation, shuffle)

    # Extract (x, y) which are (images, labels)
    images, labels = _get_images_labels(dataset_df,
                                        params.dataset,
                                        params.cross_entropy,
                                        params.original_preprocessing)

    # Finally, get dataset iterator
    ds_iterator = image_gen.flow(x = images,
                                 y = labels,
                                 batch_size = params.batch_size,
                                 seed = params.seed,
                                 shuffle = shuffle)

    print("Number of elements: {}".format(len(images)))
    return ds_iterator

def get_fer_class_mapping():
    '''Returns a dictionary mapping FER labels to class names'''
    return FER_CLASS_MAPPING

def get_fer_plus_class_mapping():
    '''Returns a dictionary mapping FER-Plus labels to class names'''
    return FER_PLUS_CLASS_MAPPING

def get_image_data(dataset_df, params):
    '''Returns just image data from a given dataset.
        Args:
            dataset_df(dataframe): specific dataset (train, valid or test) loaded
                                   from a unified dataset created using
                                   dataset.get_dataset_dict()
            params(DataPipelineParams): wrapper object with pipeline parameters
    '''
    # Extract (x, y) which are (images, labels)
    images, _ = _get_images_labels(dataset_df,
                                   params.dataset,
                                   params.cross_entropy,
                                   params.original_preprocessing)
    return images

def get_labels(dataset_df, params):
    '''Returns just labels from a given dataset.
        Args:
            dataset_df(dataframe): specific dataset (train, valid or test) loaded
                                   from a unified dataset created using
                                   dataset.get_dataset_dict()
            params(DataPipelineParams): wrapper object with pipeline parameters
    '''
        # Extract (x, y) which are (images, labels)
    _, labels = _get_images_labels(dataset_df,
                                   params.dataset,
                                   params.cross_entropy,
                                   params.original_preprocessing)
    return labels

def _get_image_generator(augmentation, is_training_set):
    '''Returns image generator that applies augmentation based on augmentation
    argument. It always applies color range normalization, even if
    augmentation = Augmentation.NONE

    Args:
        augmentation(Augmentation)
        is_training_set(boolean)
    '''
    if augmentation == Augmentation.HIGH and is_training_set:
        image_gen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 30,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True,
                                       brightness_range=(0.2, 1.2),
                                       fill_mode = 'nearest')
    elif augmentation == Augmentation.MEDIUM and is_training_set:
        image_gen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 30,
                                       zoom_range = 0.2,
                                       horizontal_flip = True,
                                       brightness_range=(0.2, 1.2),
                                       fill_mode = 'nearest')
    else:
        image_gen = ImageDataGenerator(rescale = 1./255)

    return image_gen

def _get_images_labels(dataset_df,
                       dataset,
                       cross_entropy,
                       original_preprocessing):
    '''Get image (x) and label (y) data out of dataset dataframe.

    Args:
        dataset_df(dataframe): dataframe of the specific dataset (from unified)
        dataset(enum): get labels of FER or FER-Plus
        cross_entropy(boolean): whether labels should be class probabilities
                                (it has effect only on FER-Plus)
        original_preprocessing(boolean): whether to apply original preprocessing

    Returns:
        images(ndarray)
        labels(ndarray)
    '''
    # For FER-Plus start with removing outliers
    if dataset == Dataset.FERPLUS and original_preprocessing:
        dataset_df = op.get_dataset_without_original_outliers(dataset_df,
                                                              cross_entropy,
                                                              COLUMN_NAMES)
    elif dataset == Dataset.FERPLUS:
        dataset_df = op.get_dataset_without_custom_outliers(dataset_df,
                                                            COLUMN_NAMES)

    # Reshape image data into ndarray fromat
    image_data = np.empty((len(dataset_df), IMG_SHAPE, IMG_SHAPE, 1))
    for i, img in enumerate(dataset_df['image']):
        image_data[i] = _str_to_image_data(img).reshape(IMG_SHAPE, IMG_SHAPE, 1)

    # For FER, return the integer label
    if dataset == Dataset.FER:
        int_labels = dataset_df.iloc[:, 2].values
        label_data = _basis_vectors(int_labels, 7)
    # For FER-Plus...
    else:
        label_data = dataset_df.iloc[:, 3:].values
        # Either return the distribution of probabilities as label
        # (array of elements in range [0, 1])
        if cross_entropy:
            label_data = [_p_distribution(row) for row in label_data]
        # or majority label (basis vector)
        else:
            int_labels = label_data.argmax(1)
            label_data = _basis_vectors(int_labels, 8)

    return (image_data, label_data)

def _str_to_image_data(image_blob):
    '''Convert image encoded as a string into image array'''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype = np.uint8)\
                 .reshape(IMG_SHAPE, IMG_SHAPE)
    return image_data

def _p_distribution(x):
    '''Divide each vector element by their sum. Returns a vector'''
    if isinstance(x, np.ndarray):
        return (x / x.sum()).tolist()
    else:
        return [float(xi) / sum(x) for xi in x]

def _basis_vectors(int_labels, n_classes):
    '''Transforms an array of integer labels into an array of basis vectors'''
    label_shape = (int_labels.shape[0], n_classes)
    label_data = np.zeros(label_shape)
    for row, arg in zip(label_data, int_labels):
        row[arg] = 1
    return label_data
