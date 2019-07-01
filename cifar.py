from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import model_creation
from keras.optimizers import SGD
from metrics import get_metrics


LEARNING_RATE = 0.001
OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=False)
LOSS = 'categorical_crossentropy'


def cifar_experiment():
    # prepare database
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_val, y_val = x_test[:len(x_test)//2], y_test[:len(y_test)//2]
    x_test, y_test = x_test[len(x_test)//2:], y_test[len(y_test)//2:]
    y_train, y_val, y_test = to_categorical(y_train, 10), \
                             to_categorical(y_val, 10), to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # get generator with augmentations
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)

    model = model_creation.get_convolutional_model((32, 32, 3), 10, 5,
                                                   8, 3, 1, 2, 8,
                                                   None, 0, 'relu', 'relu',
                                                   'softmax', OPTIMIZER, LOSS, get_metrics(),
                                                   padding='valid')
    model.summary()

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=30,
                        validation_data=(x_val, y_val), verbose=1)

if __name__ == '__main__':
    cifar_experiment()