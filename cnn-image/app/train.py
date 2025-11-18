import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASSES = ["airplane", "automobile", "bird"]
CLASS_INDICES = [0, 1, 2]  
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 32
EPOCHS = 2
BATCH_SIZE = 64
MODEL_PATH = "models/cnn_model.h5"


def load_and_filter_data():
    logger.info("Descargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    logger.info(f"Filtrando clases: {CLASSES}")
    
    train_mask = np.isin(y_train.flatten(), CLASS_INDICES)
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    test_mask = np.isin(y_test.flatten(), CLASS_INDICES)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    y_train_remapped = np.zeros_like(y_train_filtered)
    y_test_remapped = np.zeros_like(y_test_filtered)
    
    for new_idx, old_idx in enumerate(CLASS_INDICES):
        y_train_remapped[y_train_filtered == old_idx] = new_idx
        y_test_remapped[y_test_filtered == old_idx] = new_idx
    
    x_train_norm = x_train_filtered.astype('float32') / 255.0
    x_test_norm = x_test_filtered.astype('float32') / 255.0

    y_train_cat = to_categorical(y_train_remapped, NUM_CLASSES)
    y_test_cat = to_categorical(y_test_remapped, NUM_CLASSES)
    
    logger.info(f"Datos preparados:")
    logger.info(f"- Entrenamiento: {x_train_norm.shape[0]} imágenes")
    logger.info(f"- Prueba: {x_test_norm.shape[0]} imágenes")
    
    return (x_train_norm, y_train_cat), (x_test_norm, y_test_cat)


def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3), name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model():
    logger.info("Iniciando entrenamiento del modelo CNN...")
    
    (x_train, y_train), (x_test, y_test) = load_and_filter_data()

    logger.info("🏗️ Construyendo arquitectura del modelo...")
    model = create_cnn_model()
    model.summary()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    logger.info(f"Entrenando por {EPOCHS} épocas...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Evaluando modelo en conjunto de prueba...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    logger.info("=" * 50)
    logger.info(f"Entrenamiento completado!")
    logger.info(f"Accuracy en prueba: {test_acc:.2%}")
    logger.info(f"Loss en prueba: {test_loss:.4f}")
    logger.info(f"Modelo guardado en: {MODEL_PATH}")
    logger.info("=" * 50)
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_model()
        print("\nModelo CNN entrenado exitosamente!")
        print(f"Ubicación: {os.path.abspath(MODEL_PATH)}")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        exit(1)