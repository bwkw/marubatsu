import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')


model = create_model()

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\n')
print('Test accuracy: {}\n'.format(test_acc))

test_input = np.zeros(28 * 28).reshape((1, 28, 28))
predictions = model.predict(test_input)

print('Predictions for zero input')
print(predictions[0])

model.save_weights('model')
print()
print('Model was saved.')
