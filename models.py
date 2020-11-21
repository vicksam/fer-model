from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

from keras.regularizers import l2


def get_smart_model(input,
                    leaky_relu_slope,
                    dropout_rate,
                    regularization_rate,
                    input_shape = (48, 48, 1),
                    n_classes = 8,
                    logits = False):

    regularization = l2(regularization_rate)

    x = layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same',
                               input_shape = input_shape)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)
    x = layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)

    x = layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)
    x = layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)

    x = layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)
    x = layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)

    x = layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)
    x = layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.LeakyReLU(leaky_relu_slope)(x)

    x = layers.Conv2D(n_classes, (1, 1), padding = 'same')(x)
    output = layers.GlobalAveragePooling2D()(x)

    if not logits:
        output = layers.Softmax()(output)

    model = Model(input, output)
    return model


def get_base_model(leaky_relu_slope,
                   dropout_rate,
                   regularization_rate,
                   input_shape = (48, 48, 1),
                   n_classes = 8,
                   logits = False):

    regularization = l2(regularization_rate)

    model = keras.Sequential([
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same',
                               input_shape = input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(384,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(n_classes, (1, 1), padding = 'same'),
        layers.GlobalAveragePooling2D()
    ])

    if not logits:
        model.append(layers.Softmax())

    return model


def get_performance_model(leaky_relu_slope,
                          dropout_rate,
                          regularization_rate,
                          input_shape = (48, 48, 1),
                          n_classes = 8,
                          logits = False):

    regularization = l2(regularization_rate)

    model = keras.Sequential([
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same',
                               input_shape = input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(384,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(n_classes, (1, 1), padding = 'same'),
        layers.GlobalAveragePooling2D()
    ])

    if not logits:
        model.append(layers.Softmax())

    return model
