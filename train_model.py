import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models

from ml_utils import MODEL_IMAGE_SIZE


DATASET_ROOT = Path("COVID-19_Radiography_Dataset")
CLASS_DIRS = {
    "COVID": DATASET_ROOT / "COVID" / "images",
    "NORMAL": DATASET_ROOT / "Normal" / "images",
    "PNEUMONIA": DATASET_ROOT / "Viral Pneumonia" / "images",
}
MODEL_OUTPUT = Path("models") / "CNN_Covid19_Xray_Version.h5"
ENCODER_OUTPUT = Path("models") / "Label_encoder.pkl"
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42


def build_dataframe():
    filepaths = []
    labels = []
    for label, folder in CLASS_DIRS.items():
        for image_path in sorted(folder.glob("*.png")):
            filepaths.append(str(image_path))
            labels.append(label)
    if not filepaths:
        raise FileNotFoundError("Dataset images not found. Check the dataset folders.")
    return filepaths, labels


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, MODEL_IMAGE_SIZE)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment_image(image, label):
    image = tf.image.random_contrast(image, 0.9, 1.15)
    image = tf.image.random_brightness(image, 0.08)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(paths), seed=SEED) if training else ds
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes):
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*MODEL_IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*MODEL_IMAGE_SIZE, 3),
            include_top=False,
            weights=None,
        )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*MODEL_IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main():
    os.makedirs("models", exist_ok=True)
    filepaths, labels = build_dataframe()
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    x_train, x_temp, y_train, y_temp = train_test_split(
        filepaths,
        encoded_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=encoded_labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val)
    test_ds = make_dataset(x_test, y_test)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_map = {index: weight for index, weight in enumerate(class_weights)}

    model, base_model = build_model(len(encoder.classes_))
    fit_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=fit_callbacks,
        class_weight=class_weight_map,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=fit_callbacks,
        class_weight=class_weight_map,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    model.save(MODEL_OUTPUT)
    with open(ENCODER_OUTPUT, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


if __name__ == "__main__":
    main()
