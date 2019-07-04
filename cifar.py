from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import model_creation
from keras.optimizers import SGD
from metrics import get_metrics
from reinitialize_reversion import *
from keras.initializers import glorot_normal
import datetime
from ranking import *


LEARNING_RATE = 0.0001
OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0, nesterov=False)
LOSS = 'categorical_crossentropy'
BATCH_SIZE = 32
THRESHOLD = 0.1
EPOCHS = 100
ITERATION_EPOCHS = 20


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
                                                   16, 3, 1, 2, 32,
                                                   None, 0, 'relu', 'relu',
                                                   'softmax', OPTIMIZER, LOSS, get_metrics(),
                                                   padding='valid')
    model.summary()


    ranker = magnitude_ranking
    weight_reinitializer = MagnitudeReinitializer(KerasKernelInitializerWrapper(glorot_normal), ZeroBiasInitializer(), ranker=ranker)
    model_reinitializer = MagnitudeModelReinitializer(model, weight_reinitializer)
    # weight_reinitializer = DiffReinitializer(ranker, KerasKernelInitializerWrapper(glorot_normal), ZeroBiasInitializer())
    # model_reinitializer = DiffModelReinitializer(model, weight_reinitializer)

    percentage = 0.8
    # find good init
    i = 0
    while percentage > THRESHOLD:
        i += 1
        print("Entering iteration {}".format(i))
        print("Check model's performance before this round of fitting:")
        print(model.evaluate(x=x_val, y=y_val, batch_size=BATCH_SIZE, verbose=2))
        # model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), epochs=1, steps_per_epoch=len(x_train) // (BATCH_SIZE * 100),
        #                     validation_data=(x_val, y_val), validation_steps=len(x_val) // (BATCH_SIZE * 100), verbose=1)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=ITERATION_EPOCHS,
                            validation_data=(x_val, y_val), verbose=2)
        percentage = 0.8 ** i
        model_reinitializer.reinitialize(percentage=percentage)
        print("Reinitialized {} out of {} weights (Percentage = {})".format(percentage * model.count_params(),
                                                                            model.count_params(), percentage))

    # percentages = ([0.5] * 2) + ([0.3] * 2) + ([0.2] * 2)
    # for _ in range(5):
    #     i += 1
    #     print("Entering iteration {}".format(i))
    #     print("Check model's performance before this round of fitting:")
    #     print(model.evaluate(x=x_val, y=y_val, batch_size=BATCH_SIZE, verbose=2))
    #     # model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), epochs=1, steps_per_epoch=len(x_train) // (BATCH_SIZE * 100),
    #     #                     validation_data=(x_val, y_val), validation_steps=len(x_val) // (BATCH_SIZE * 100), verbose=1)
    #     model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=5,
    #                         validation_data=(x_val, y_val), verbose=2)
    #     percentage = percentages[i - 1]
    #     model_reinitializer.reinitialize(percentage=percentage)
    #     print("Reinitialized {} out of {} weights (Percentage = {})".format(percentage * model.count_params(),
    #                                                                         model.count_params(), percentage))

    print("Starting final fitting")
    # now fit
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=EPOCHS,
                        validation_data=(x_val, y_val), verbose=2)

if __name__ == '__main__':
    print("---------- Using {}, {} epochs per iteration ----------".format(ranker, ITERATION_EPOCHS))
    print("Experiment start: {}".format(datetime.datetime.now()))
    cifar_experiment()
    print("Experiment end: {}".format(datetime.datetime.now()))