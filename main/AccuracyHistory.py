import keras


class AccuracyHistory(keras.callbacks.Callback):
    EPOCH = 0

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.save()

        self.acc.append(logs.get('acc'))

    def save(self):
        model_yaml = self.model.to_yaml()

        if self.EPOCH % 10 == 0:

            with open("model" + str(self.EPOCH) + ".yaml", "w") as yaml_file:
                yaml_file.write(model_yaml)
            self.model.save_weights("model00" + str(self.EPOCH) + ".h5")
        self.EPOCH = self.EPOCH + 1
        print("Saved model to disk")
