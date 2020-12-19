import logging

from kerastuner.tuners import RandomSearch
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential

logger = logging.getLogger(__name__)


class A1:
    """
    class to create a model
    :param input_dim: the input dimension of the data
    :return: the sequential model
    """

    def __init__(self, input_dim, X_train, Y_train, X_val, Y_val):
        self.input_shape = input_dim
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.model = self.create_model()
        self.model.summary()

    def create_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=112, kernel_size=(5, 5), input_shape=self.input_shape, padding="same",
                              kernel_regularizer=regularizers.l2(l=0.0001), activation='relu'))
        self.model.add(Conv2D(filters=96, kernel_size=(4, 4), padding="same",
                              kernel_regularizer=regularizers.l2(l=0.0001), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=96, kernel_size=(5, 5), padding="same",
                              kernel_regularizer=regularizers.l2(l=0.0001), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())

        self.model.add(Dense(1056, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1, activation='sigmoid'))

        adam = keras.optimizers.Adam(lr=0.0003)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])

        return self.model

    def train(self):
        """
        Fit function
        """
        es = EarlyStopping(monitor='val_loss', patience=3)

        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=30,
            verbose=True,
            batch_size=32,
            callbacks=[es],
            validation_data=(self.X_val, self.Y_val))

        return self.model, history

    def tune_model(self, hp):
        model = keras.Sequential([
            keras.layers.Conv2D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values=[3, 4, 5]),
                activation='relu',
                input_shape=self.input_shape
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(hp.Choice('drop_out', values=[0.1, 0.2, 0.3])),

            keras.layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_2_kernel', values=[3, 4, 5]),
                activation='relu'
            ),
            keras.layers.Dropout(hp.Choice('drop_out', values=[0.1, 0.2, 0.3])),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(
                filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_3_kernel', values=[3, 4, 5]),
                activation='relu'
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(hp.Choice('drop_out', values=[0.1, 0.2, 0.3])),

            keras.layers.Flatten(),

            keras.layers.Dense(
                units=hp.Int('dense_1_units', min_value=32, max_value=2048, step=64),
                activation='relu'
            ),

            keras.layers.Dropout(hp.Choice('drop_out_dense', values=[0.2, 0.3, 0.5])),

            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.03, 0.0003])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def fit_tune(self):
        """
        Auto Fit function
        """
        tuner_search = RandomSearch(self.tune_model,
                                    objective='val_accuracy',
                                    max_trials=5, directory='tests/kerastuner_out', project_name="task_A1")

        tuner_search.search(self.X_train, self.Y_train, epochs=5, validation_data=(self.X_val, self.Y_val))

        model = tuner_search.get_best_models(num_models=1)[0]

        model.fit(self.X_train, self.Y_train, shuffle=True, epochs=25, validation_split=0.3, initial_epoch=5)

        return model

    def evaluate_model(self, task):
        score = self.model.evaluate(self.X_val, self.Y_val, verbose=0)
        print("--------------------------------------------------")
        print('| Task: %s | loss -  %.3f, | Accuracy -  %.2f |' % (task, score[0], score[1]))
        print("--------------------------------------------------")

        return score[0], score[1]

