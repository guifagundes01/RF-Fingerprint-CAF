import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import h5py

class IQSampleLoader:
    def __init__(self, dataset_name, labelset_name):
        self.dataset_name = dataset_name
        self.labelset_name = labelset_name

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col/2)], dtype=np.complex64)
        data_complex = data[:, :round(num_col/2)] + 1j * data[:, round(num_col/2):]
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.

        INPUT:
            file_path: The dataset path.
            dev_range: Specifies the loaded device range.
            pkt_range: Specifies the loaded packets range.

        RETURN:
            data: The loaded complex IQ samples.
            label: The true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        label_start = int(label[0][0]) + 1
        label_end = int(label[-1][0]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)

        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            # print(sample_index_dev)
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]

        f.close()
        return data, label

class ChannelIndSpectrogram():
    def __init__(self):
        pass

    def _normalization(self, data):
        '''Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i] / rms

        return s_norm

    def _gen_single_channel_ind_spectrogram(self, sig):
    
        N = len(sig)
        # Finding the right alpha
        taus = np.arange(-300, 200)
        alphas = np.arange(-0.3, 0.3, 0.005)
        CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
        for j in range(len(alphas)):
            for i in range(len(taus)):
                CAF[j, i] = np.sum(sig *
                            np.conj(np.roll(sig, taus[i])) *
                            np.exp(-2j * np.pi * alphas[j] * np.arange(N)))
                
        CAF2=CAF.copy()
        CAF2[60] = 0

        return np.abs(CAF2)

    def channel_ind_spectrogram(self, data):
        '''Converts the data to channel-independent spectrograms.'''
        data = self._normalization(data)
        num_sample = data.shape[0]
        num_row = 256  # after cropping: int(256*0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec

from keras import layers, models
from keras import optimizers
from keras import callbacks
# Define the CNN model

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers with increased L2 regularization
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
   
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())
        

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

from sklearn.metrics import confusion_matrix

def plot_results(history):
    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 5))

    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # Setting y-axis limits from 0 to 1
    plt.legend()

    # Plotting accuracy with y-axis starting from 0
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Setting y-axis limits from 0 to 1
    plt.legend()

    plt.show()

def plot_confusion_matrix(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(20))
    plt.yticks(np.arange(20))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


# Usage example
if __name__ == "__main__":
    file_path = './image_generation/data/dataset_training_aug.h5'    
    dev_range = np.arange(0, 20, dtype=int)
    pkt_range = np.arange(0, 1000, dtype=int)
    print("generate IQ data")
    LoadDatasetObj = IQSampleLoader(dataset_name='data', labelset_name='label')
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)

    print("generate training data")
    ch = ChannelIndSpectrogram()
    data_channel_ind_spec = ch.channel_ind_spectrogram(data)

    X_train, X_test, y_train, y_test = train_test_split(data_channel_ind_spec, label, test_size=0.2, random_state=42,stratify=label)

    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    input_shape = (data_channel_ind_spec.shape[1], data_channel_ind_spec.shape[2], 1)
    
    # Define the model
    model = create_model(input_shape, 20)

    # Set optimizer and initial learning rate
    optimizer = optimizers.Adam(learning_rate=0.0003)

    # Compile the model with the specified optimizer
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, min_lr=1e-6)
    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8,
                        validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler])

    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print("Validation Accuracy:", val_accuracy)
    plot_results(history)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)
    plot_results(history)

# Confusion Matrix
    plot_confusion_matrix(X_val, y_val, model)
    x=models.load_model()