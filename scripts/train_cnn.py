"""
Etwas abgeänderte Version des CNN-Training-Skripts, das uns gegeben wurde.
"""

import os
import sys
import uuid
import json

sys.path.append(os.path.join(os.path.dirname(__file__)))

import qkeras as qk
from qkeras import *  # type: ignore
import layers as fql

import numpy as np
import pathlib
import seaborn as sns

import matplotlib.pyplot as plt
import tensorflow as tf


if len(sys.argv) != 3:
    print("Usage: python cnn_training.py <config_file> <run_name>")
    sys.exit(1)

config_file = sys.argv[1]

with open(config_file, "r") as f:
    config = json.load(f)

print(config)

run_name = sys.argv[2]

VALIDATION_SPLIT = 0.20
BATCH_SIZE = config["training_batch_size"]
LEARNING_RATE = config["training_lr"]
EPOCHS = config["training_epochs"]


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_2d(waveform):
    waveform = waveform[..., tf.newaxis]
    return waveform


def get_2d_quantized(waveform, bitwidth, integer_bits):
    waveform = waveform[..., tf.newaxis]
    waveform = qk.quantized_bits(bitwidth, integer_bits - 1, 0)(waveform)
    return waveform


def make_2d_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_2d(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def make_2d_ds_quantized(ds, bitwidth, integer_bits):
    return ds.map(
        map_func=lambda audio, label: (
            get_2d_quantized(audio, bitwidth, integer_bits),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


DATASET_PATH = "data/mini_speech_commands"
data_dir = pathlib.Path(DATASET_PATH)

if not data_dir.exists():
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=".",
        cache_subdir="data",
    )

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != "README.md") & (commands != ".DS_Store")]

print("Commands:", commands)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=256,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    seed=42,
    output_sequence_length=16000,
    subset="both",
)

label_names = np.array(train_ds.class_names)
print("\nlabel names:", label_names)

###########################################################################################
### Here are the configurations for the bitwidth and the fractional wordlength.         ###
### The current settings of bitwidth = 16 and fractional_word_length = 10 will work.    ###
### Only change the bitwidth if you know what you are doing, it is aligned with         ###
### the Wakeword Classifier Component.                                                  ###
### When you change the fractional_word_length or bitwidth, also do this in line 17     ###
### of the convert_and_compare_dnn.py script.                                           ###
###########################################################################################

bitwidth = 16  # 16
fractional_word_length = 10
# die tatsächliche Anzahl der Integer Bits ist eins weniger, da das erste Bit das Vorzeichen Bit ist
integer_bits = bitwidth - fractional_word_length

model_name = "audio_cnn"

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

train_2d_ds = make_2d_ds(train_ds)
train_2d_ds_q = make_2d_ds_quantized(train_ds, bitwidth, integer_bits)

val_2d_ds = make_2d_ds(val_ds)
val_2d_ds_q = make_2d_ds_quantized(val_ds, bitwidth, integer_bits)

test_2d_ds = make_2d_ds(test_ds)
test_2d_ds_q = make_2d_ds_quantized(test_ds, bitwidth, integer_bits)

input_shape = (config["input_shape"][0], config["input_shape"][1], config["input_shape"][2])
print("Input shape:", input_shape)

num_labels = len(label_names)
print("Number of labels:", num_labels)

###########################################################################################
### Here the actual model of the Convolution Neural Network (CNN) is defined. With the  ###
### parameters filt_num and kernel_length the size of all layers can be adjusted at     ###
### once. Additional layers have to be added manually, as well ad removing them.        ###
### More Details of the architecture are given in the power point slides                ###
###########################################################################################

filt_num = 8
kernel_len = 5

kernel_quantizer = qk.quantized_bits(bitwidth, integer_bits - 1, 0)
bias_quantizer = qk.quantized_bits(bitwidth, integer_bits - 1, 0)
activation_fn = f"quantized_relu({bitwidth}, {integer_bits - 1})"
quantizers = {"kernel_quantizer": kernel_quantizer, "bias_quantizer": bias_quantizer}

# jetzt müssen wir unser CNN aus den Angaben in der Datei bauen:
arch = []

for layer in config["layers"]:
    type_ = layer["type"]

    if type_ == "Conv2D":
        params = layer["params"]
        filters = params["filters"]
        kernel_size = (params["kernel_size"][0], params["kernel_size"][1])
        strides = (params["strides"][0], params["strides"][1])

        if len(arch) == 0:
            arch.append(
                fql.FullyQConv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    **quantizers,
                    input_shape=input_shape,
                )
            )
        else:
            arch.append(
                fql.FullyQConv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    **quantizers,
                )
            )
    elif type_ == "ReLU":
        arch.append(qk.QActivation(activation_fn))
    elif type_ == "BatchNormalization":
        arch.append(fql.FullyQBatchNormalization())
    elif type_ == "Dropout":
        rate = layer["params"]["rate"]
        arch.append(tf.keras.layers.Dropout(rate))
    elif type_ == "Dense":
        arch.append(fql.FullyQDense(units=layer["params"]["units"], **quantizers))
    elif type_ == "Flatten":
        arch.append(tf.keras.layers.Flatten())
    elif type_ == "GlobalAveragePooling2D":
        arch.append(tf.keras.layers.GlobalAveragePooling2D())

model_qkeras = tf.keras.Sequential(
    arch,
    name=model_name,
)

###########################################################################################
### Here the training process in configured and executed. The most crucial parameters   ###
### are the number of Epochs, which defines the length of the training, and the         ###
### learning rate. Shorter trainings will be faster, but the CNN performance will       ###
### suffer. Is the training to long, overfitting may occur. To observe this, there are  ###
### two dataset: the training dataset (train_2d_ds_q) and the validation dataset        ###
### (val_2d_ds). The CNN is trained on the training dataset, but it has never seen      ###
### the validation data. If the validation accuracy starts to drop,                     ###
### the CNN is overfitting.                                                             ###
### You can just leave these options as they are.                                       ###
###########################################################################################


folder_path = os.path.join("data", "cnns", f"{run_name}")
os.mkdir(folder_path)

model_qkeras.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model_qkeras.summary()
num_params = np.sum([np.prod(v.get_shape().as_list()) for v in model_qkeras.trainable_variables])

# Add ModelCheckpoint callback to save best model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(folder_path, "best_model.h5"),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

history = model_qkeras.fit(
    train_2d_ds_q, validation_data=val_2d_ds, epochs=EPOCHS, callbacks=[checkpoint_callback]
)

# Save final model
model_qkeras.save(os.path.join(folder_path, "model.h5"))

# Load the best model based on validation accuracy
model_qkeras = tf.keras.models.load_model(
    os.path.join(folder_path, "best_model.h5"),
    custom_objects={
        "FullyQConv2D": fql.FullyQConv2D,
        "QActivation": qk.QActivation,
        "FullyQDense": fql.FullyQDense,
        "FullyQBatchNormalization": fql.FullyQBatchNormalization,
        "FullyQDepthwiseConv2D": fql.FullyQDepthwiseConv2D,
        "FullyQSeparableConv2D": fql.FullyQSeparableConv2D,
        "QGlobalAveragePooling2D": qk.QGlobalAveragePooling2D,
        "quantized_relu": qk.quantized_relu,
    },
)

# plot training history
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="lower right")
plt.savefig(os.path.join(folder_path, "training_accuracy.png"))
plt.close()


best_acc = round(max(history.history["val_accuracy"]) * 100, 2)
best_loss = round(min(history.history["val_loss"]), 3)

with open(os.path.join(folder_path, f"_acc_{best_acc}%_loss_{best_loss}.txt"), "w") as f:
    f.write(
        f"""=== ACCURACY REPORT ===
    Best validation accuracy: {best_acc}%
    Best validation loss:     {best_loss}
    Training accuracy:   {history.history['accuracy'][-1]*100:.4f}%
    Training loss:       {history.history['loss'][-1]:.4f}\n

=== PARAMETER REPORT ===
    Number of parameters: {num_params}
    Learning rate:        {LEARNING_RATE}
    Epochs:               {EPOCHS}
    Batch size:           {BATCH_SIZE}
    Num conv layers:      N/A
    First layer mul:      N/A
    Bitwidth:             {bitwidth}
    Frac word length:     {fractional_word_length}
    Integer bits:         {integer_bits}
    Model name:           {model_name}
    Run name:             {run_name}"""
    )


with open(os.path.join(folder_path, "info.json"), "w") as f:
    json.dump(
        {
            "run_name": run_name,
            "model_name": model_name,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "best_val_accuracy": best_acc,
            "best_val_loss": best_loss,
            "num_params": int(num_params),
            "arch": config["name"],
        },
        f,
    )


# Plot training & validation loss values
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.savefig(os.path.join(folder_path, "training_loss.png"))
plt.close()

y_pred = tf.argmax(model_qkeras.predict(test_2d_ds_q), axis=1)
y_true = tf.concat(list(test_2d_ds_q.map(lambda s, lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt="g")
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.savefig(os.path.join(folder_path, "confusion_matrix.png"))

# choose test audio sample out of 64 batch
test_index = 5

example_audio, example_labels = None, None

for example_audio, example_labels in train_2d_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)

if example_audio is not None and example_labels is not None:
    sample_in = example_audio.numpy()[test_index, :, :, :]
    sample_out = example_labels.numpy()[test_index]
    # export example from dataset as numpy array
    file_name = "./" + model_name + "_sample.npy"
    np.save(file_name, sample_in)
    sample_in = np.expand_dims(sample_in, 0)
    sample_out = np.expand_dims(sample_out, 0)
