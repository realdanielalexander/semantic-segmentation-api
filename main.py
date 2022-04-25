from flask import Flask, request, Response, jsonify, send_from_directory, abort, make_response
from flask_cors import CORS, cross_origin
import time
import os
import base64
import csv
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf

# customize your API through the following parameters
classes_path = './classes/class_dict.csv'
weights_path = './weights/deeplabv3_100epochs_bandung_focalloss_iou.h5'
height = 384                      # size images are resized to for model
width = 576                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 9                # number of classes in model

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

labels = dict()

subset_classes = [
                  'Building',
                  'Tree',
                  'Sky',
                  'Car',
                  'SignSymbol',
                  'Road',
                  'Pedestrian',
                  'Sidewalk',
                  'MotorcycleScooter'
]
with open(classes_path, 'r') as csvin:
  reader = csv.DictReader(csvin)
  for row in reader:
    if row['name'] == 'Train':
      continue
    # labels[row['name']] = (int(row['r']), int(row['g']), int(row['b']))
    if row['name'] in subset_classes:
      labels[row['name']] = (int(row['r']), int(row['g']), int(row['b']))
print(labels)

def int_to_rgb_label(input):
  """
  Input: 0-8
  Output: RGB Labels
  """
  # label_seg = np.zeros((input.shape[0], input.shape[1], 3), dtype=np.uint8)
  label_seg = np.full((input.shape[0], input.shape[1], 3), 0, dtype=np.uint8)
  for i, (label, rgb) in enumerate(labels.items()):
    label_seg[np.all(input == i, axis=-1)] = rgb
  
  return label_seg

def convolution_block(
    block_input,
    num_filters=64,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input, num_filters=64):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, num_filters=num_filters, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, num_filters=num_filters, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, num_filters=num_filters, kernel_size=1)
    return output

def DeeplabV3Plus(image_height, image_width, num_filters, num_classes):
    model_input = tf.keras.Input(shape=(image_height, image_width, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x, num_filters=num_filters)

    input_a = tf.keras.layers.UpSampling2D(
        size=(image_height // 4 // x.shape[1], image_width // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=num_filters)
    x = convolution_block(x, num_filters=num_filters)
    x = tf.keras.layers.UpSampling2D(
        size=(image_height // x.shape[1], image_width // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model_output = tf.keras.layers.Activation('softmax', name='softmax')(model_output)
    return tf.keras.Model(inputs=model_input, outputs=model_output)

model_64 = DeeplabV3Plus(image_height=height, image_width=width, num_filters=64, num_classes=num_classes)
model_128 = DeeplabV3Plus(image_height=height, image_width=width, num_filters=128, num_classes=num_classes)
model_256 = DeeplabV3Plus(image_height=height, image_width=width, num_filters=256, num_classes=num_classes)
# load in weights and classes

# model = tf.keras.models.load_model(weights_path, compile=False)
model_64.load_weights('./weights/weights_64.h5')
model_128.load_weights('./weights/weights_128.h5')
model_256.load_weights('./weights/weights_256.h5')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# API that returns image with detections on it
@app.route('/image-64', methods= ['POST'])
def get_image_64():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    
    original_image = cv2.imread(image_name, 3)
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, (384, 576))

    # img = transform_images(img, size)

    t1 = time.time()

    prediction = model_64.predict(img)
    prediction = np.argmax(prediction, axis=3)
      
    print(prediction)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    result = np.expand_dims(prediction[0], axis=-1)
    result = int_to_rgb_label(result)    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path + 'detection.jpg', result)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', result)
    response = base64.b64encode(img_encoded.tostring())
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200)
        # return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# API that returns image with detections on it
@app.route('/image-128', methods= ['POST'])
def get_image_128():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    
    original_image = cv2.imread(image_name, 3)
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, (384, 576))

    # img = transform_images(img, size)

    t1 = time.time()

    prediction = model_128.predict(img)
    prediction = np.argmax(prediction, axis=3)
      
    print(prediction)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    result = np.expand_dims(prediction[0], axis=-1)
    result = int_to_rgb_label(result)   
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    cv2.imwrite(output_path + 'detection.jpg', result)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', result)
    response = base64.b64encode(img_encoded.tostring())
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200)
        # return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# API that returns image with detections on it
@app.route('/image-256', methods= ['POST'])
def get_image_256():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    
    original_image = cv2.imread(image_name, 3)
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, (384, 576))

    # img = transform_images(img, size)

    t1 = time.time()

    prediction = model_256.predict(img)
    prediction = np.argmax(prediction, axis=3)
      
    print(prediction)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    result = np.expand_dims(prediction[0], axis=-1)
    result = int_to_rgb_label(result)    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path + 'detection.jpg', result)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', result)
    response = base64.b64encode(img_encoded.tostring())
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200)
        # return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# API that returns image with detections on it
@app.route('/original', methods= ['POST'])
def get_original():
    print('request')
    print(request.files)
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    
    original_image = cv2.imread(image_name, 3)
    original_image = cv2.resize(original_image, (576, 384))
    cv2.imwrite(output_path + 'original.jpg', original_image)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, original_encoded = cv2.imencode('.png', original_image)
    response = base64.b64encode(original_encoded.tostring())
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200)
        # return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)