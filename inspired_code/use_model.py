# %%
import numpy as np
import cnn_model
import matplotlib.pyplot as plt
from keras import models
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
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

# %%
if __name__ == "__main__":
    rel_path = "C:/Users/gfagu/OneDrive/Supelec/2A/Pole_Projet/rff/rf-fingerprinct-caf/image_generation/"
    data_channel_ind_spec = np.load(rel_path+"output_dat/data_imgs_caf_abs_.dat", allow_pickle=True)
    label = np.load(rel_path+"output_dat/label2.dat", allow_pickle=True)
    print(data_channel_ind_spec)
    print(label)
    X_train, X_val, y_train, y_val = train_test_split(data_channel_ind_spec, label, test_size=0.4, random_state=42,stratify=label)

    input_shape = (data_channel_ind_spec.shape[1], data_channel_ind_spec.shape[2], 1)
    
    # Define the model
    model = cnn_model.create_model(input_shape, 20)

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

    # %%
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8,
                        validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler])

    # %%
    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print("Validation Accuracy:", val_accuracy)
    plot_results(history)

# Confusion Matrix
    plot_confusion_matrix(X_val, y_val, model)
    x=models.load_model()
# %%
