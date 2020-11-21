import numpy as np
import tensorflow as tf

def get_tflite_model_predictions(tflite_model_path, image_data):
    '''Loads the tflite model and runs it on the input images to get predictions.

    Args:
        tflite_model_path(string): path to .tflite model file
        image_data(ndarray): array of unnormalized images (n, 48, 48, 1)

    Returns: array with the most probable integer label for each image.
    '''
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path = tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Collect model's predictions for the input data
    y_pred = np.zeros(len(image_data), dtype = np.int)

    # Model takes in images one by one
    for i in range(len(image_data)):
        # Prepare input data
        image = image_data[i].astype(np.float32) # int -> float32
        image = image / 255. # color range [0, 255] -> [0, 1]
        input_data = np.expand_dims(image, axis = 0) # (48, 48, 1) -> (1, 48, 48, 1)
        # Pass data to the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Predict
        interpreter.invoke()
        # Collect the output
        output = interpreter.get_tensor(output_details[0]['index'])
        # Transform it to an integer label
        prediction = np.argmax(output, axis = 1)
        # Add to predicted values
        y_pred[i] = prediction

    return y_pred
