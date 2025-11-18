import os
import logging
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "cnn-service", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/cnn_model.h5")
IMG_SIZE = 32
CLASSES = ["airplane", "automobile", "bird"]


def apply_filters(image: Image.Image):
    from PIL import ImageFilter

    blur_img = image.filter(ImageFilter.GaussianBlur(radius=2))
    edge_img = image.filter(ImageFilter.FIND_EDGES)
    sharp_img = image.filter(ImageFilter.SHARPEN)

    return blur_img, edge_img, sharp_img


def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            logger.info(f"Modelo CNN cargado desde: {MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            return None
    else:
        logger.warning(f"Modelo no encontrado en: {MODEL_PATH}")
        logger.warning("Solución: Ejecuta 'python train.py' para entrenar el modelo")
        return None


model = load_cnn_model()


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    applied_filters: List[str]
    limitations: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str


app = FastAPI(
    title="CNN Image Classification Service",
    version="1.0.0",
    description="Servicio de clasificación de imágenes usando CNN",
)


@app.get("/", response_model=dict)
def root():
    return {
        "service": "CNN Image Classifier",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "classes": CLASSES,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    if model is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="Modelo cargado y listo para predicciones",
        )
    else:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            message="Servicio activo pero modelo no entrenado. Ejecuta train.py",
        )


@app.post("/classify", response_model=PredictionResponse)
async def classify_image(file: UploadFile = File(...)):
    if model is None:
        logger.error("Intento de clasificación sin modelo entrenado")
        return PredictionResponse(
            predicted_class="N/A",
            confidence=0.0,
            applied_filters=[],
            limitations="",
            error="Modelo no disponible. Por favor, entrena el modelo primero ejecutando train.py",
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        original_size = image.size

        logger.info(
            f"Imagen recibida: {file.filename}, tamaño original: {original_size}"
        )

        blur_img, edge_img, sharp_img = apply_filters(image)
        applied_filters = ["Gaussian Blur", "Edge Detection", "Sharpen"]

        image_resized = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(image_resized)
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASSES[predicted_class_idx]

        logger.info(f"Clasificación: {predicted_class} (confianza: {confidence:.2%})")

        limitations = (
            f"LIMITACIONES DEL MODELO:\n"
            f"- Solo reconoce 3 clases: {', '.join(CLASSES)}\n"
            f"- Entrenado con imágenes de 32x32 píxeles\n"
            f"- Precisión limitada (~70-80%)\n"
            f"- Imágenes fuera de estas categorías pueden clasificarse incorrectamente"
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            applied_filters=applied_filters,
            limitations=limitations,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error al procesar imagen: {str(e)}", exc_info=True)
        return PredictionResponse(
            predicted_class="N/A",
            confidence=0.0,
            applied_filters=[],
            limitations="",
            error=f"Error al procesar la imagen: {str(e)}",
        )


@app.get("/model-info")
def model_info():
    if model is None:
        return {"loaded": False, "message": "Modelo no entrenado"}

    try:
        return {
            "loaded": True,
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "architecture": "Sequential CNN con 3 capas convolucionales",
        }
    except Exception as e:
        return {"loaded": True, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    logger.info("Iniciando CNN Service...")
    logger.info(f"Ruta del modelo: {MODEL_PATH}")
    logger.info(f"Clases disponibles: {CLASSES}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
