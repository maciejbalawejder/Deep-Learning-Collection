import tensorflow as tf

class Model:
    def __init__(self, X, y, lr = 1e-4, units = 512, drate = .3):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.units = units
        self.lr = lr
        self.drate = drate
        self.GRU = self.gru()

    def gru(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.GRU(self.units,return_sequences=True,input_shape = self.X.shape[1:]))
        model.add(tf.keras.layers.Dropout(self.drate))

        model.add(tf.keras.layers.GRU(self.units,return_sequences=True,input_shape = self.X.shape[1:]))
        model.add(tf.keras.layers.Dropout(self.drate))

        model.add(tf.keras.layers.GRU(self.units,return_sequences=False,input_shape = self.X.shape[1:]))
        model.add(tf.keras.layers.Dropout(self.drate))

        model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model


    def train(self, epochs = 50, batch = 1):
        self.history = self.GRU.fit(self.X, self.y, epochs = epochs, batch_size = batch, validation_split=0.2, verbose = 0)


    def plot(self, save = False):
        plt.figure(dpi = 120, figsize = (10,5))
        plt.subplot(121)
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(self.history.history['accuracy'], color = 'b', label = 'Training')
        plt.plot(self.history.history['val_accuracy'], color = 'y', label = 'Validation')
        plt.legend(loc = "lower right")

        if save :
          plt.savefig('Accuracy.png')

        plt.subplot(122)
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(self.history.history['loss'], color = 'b', label = 'Training')
        plt.plot(self.history.history['val_loss'], color = 'y', label = 'Validation')
        plt.tight_layout()
        plt.legend(loc = "lower left")

        if save :
          plt.savefig('Loss.png')

          files.download('Loss.png')
          files.download('Accuracy.png')


    def accuracy(self,test_X, test_Y):
        score = self.GRU.evaluate(test_X.astype(np.float32), test_y.astype(np.float32), verbose=0)
        print(f'Test loss: {score[0]} Test accuracy: {round(score[1],2)}')

