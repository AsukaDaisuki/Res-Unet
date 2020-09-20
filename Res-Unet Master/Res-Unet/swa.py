# -*- coding: utf-8 -*-

import keras as K


class SWA(K.callbacks.Callback):

    def __init__(self, filepath, SWA_START):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.SWA_START = SWA_START

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('SWA selected for last {} epochs.'
              .format(self.nb_epoch - self.SWA_START))

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.backend.get_value(self.model.optimizer.lr))
        print('learning rate of current epoch : {}'.format(lr))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.SWA_START:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.SWA_START:
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] *(epoch - self.SWA_START) + self.model.get_weights()[i]) / ((epoch - self.SWA_START) + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('[FINISH].')
        
        
class LearningRateDisplay(K.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.backend.get_value(self.model.optimizer.lr))
        print('learning rate of current epoch is : {}'.format(lr))