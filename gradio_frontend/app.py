import os
import requests
import gradio as gr
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import io
import logging
import json
import random
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_SERVICE_URL", "http://llm_connector:8001")
ML_URL = os.getenv("ML_SERVICE_URL", "http://sklearn_model:8002")
CNN_URL = os.getenv("CNN_SERVICE_URL", "http://cnn:8003")


def check_service_health(service_url: str, service_name: str) -> bool:
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"{service_name} conectado correctamente.")
            return True
        else:
            logger.warning(f"{service_name} respondió con código: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"No se puede conectar a {service_name}: {e}")
        return False


def wait_for_services(max_attempts=30, delay=2):
    services = [
        (LLM_URL, "LLM Service"),
        (ML_URL, "ML Service"),
        (CNN_URL, "CNN Service")
    ]
    
    for attempt in range(max_attempts):
        all_healthy = True
        for service_url, service_name in services:
            if not check_service_health(service_url, service_name):
                all_healthy = False
                logger.info(f"Intento {attempt + 1}/{max_attempts}: {service_name} no disponible. Esperando...")
        
        if all_healthy:
            logger.info("¡Todos los servicios están disponibles!")
            return True
        
        time.sleep(delay)
    
    logger.error("No todos los servicios están disponibles después de los intentos máximos.")
    return False

logger.info("Verificando disponibilidad de servicios...")
services_ready = wait_for_services()

def chat_with_llm(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Por favor, escribe una pregunta."
    
    try:
        logger.info(f"Enviando consulta al LLM: {prompt[:50]}...")
        response = requests.post(
            f"{LLM_URL}/query",
            json={"prompt": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No se pudo obtener respuesta del servicio.")
        else:
            logger.error(f"Error LLM: {response.status_code}")
            return f"Error del servicio LLM: {response.status_code}"

    except requests.Timeout:
        return "Timeout: El servicio LLM tardó demasiado en responder."
    except requests.ConnectionError:
        return "Error de conexión: No se puede conectar al servicio LLM."
    except Exception as e:
        logger.exception("Error en chat_with_llm")
        return f"Error inesperado: {str(e)}"

def predict_ml(*features_tuple) -> Any:
    features = list(features_tuple)
    
    try:
        features = [float(f) if f is not None else 0.0 for f in features]
    except (ValueError, TypeError):
        return "Error: Todas las características deben ser números.", {}
    
    if len(features) != 13:
        return f"Se requieren exactamente 13 características. Recibidas: {len(features)}", {}

    try:
        logger.info(f"Enviando predicción ML con features: {features[:3]}...")
        response = requests.post(
            f"{ML_URL}/predict",
            json={"features": features},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("error"):
                return f"Error: {data['error']}", {}
            
            pred = data.get("prediction")
            probs = data.get("probabilities", [])
            
            if pred is None or not probs:
                return "Respuesta incompleta del servicio ML.", {}
            
            # Crear diccionario de probabilidades
            class_names = ["Barolo", "Grignolino", "Barbera"]
            probs_dict = {class_names[i]: float(probs[i]) for i in range(min(len(probs), 3))}
            
            pred_name = class_names[pred] if 0 <= pred < len(class_names) else f"Clase {pred}"

            result_msg = f"Predicción: {pred_name}\n"
            result_msg += f"Confianza: {max(probs)*100:.2f}%"

            return result_msg, probs_dict

        logger.error(f"Error ML Service: {response.status_code}")
        return f"Error del servicio ML: {response.status_code}", {}

    except requests.Timeout:
        return "⏱️ Timeout: El servicio ML tardó demasiado.", {}
    except requests.ConnectionError:
        return "Error de conexión con el servicio ML.", {}
    except Exception as e:
        logger.exception("Error en predict_ml")
        return f"Error inesperado: {str(e)}", {}

def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def classify_image(image) -> Any:
    if image is None:
        return "Por favor, sube una imagen.", "N/A", 0.0, "N/A", "N/A"

    try:
        img = Image.fromarray(image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("upload.jpg", img_bytes.getvalue(), "image/jpeg")}
        
        logger.info("Enviando imagen para clasificación...")
        response = requests.post(
            f"{CNN_URL}/classify",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("error"):
                error_msg = data["error"]
                return f"Error: {error_msg}", "N/A", 0.0, "N/A", "N/A"

            predicted_class = data.get("predicted_class", "Desconocida")
            confidence = float(data.get("confidence", 0.0))
            filters = safe_text(data.get("applied_filters", ""))
            limitations = safe_text(data.get("limitations", ""))


            result_msg = f"Clasificación exitosa\n"
            result_msg += f"Clase: {predicted_class}\n"
            result_msg += f"Confianza: {confidence*100:.2f}%"

            return (
                str(result_msg),
                str(predicted_class),
                float(confidence),
                str(filters),
                str(limitations)
            )


        logger.error(f"Error CNN Service: {response.status_code}")
        return (
            f"Error del servicio CNN: {response.status_code}",
            "N/A",
            0.0,
            "N/A",
            "N/A"
        )

    except requests.Timeout:
        return "Timeout: El servicio CNN tardó demasiado.", "N/A", 0.0, "N/A", "N/A"
    except requests.ConnectionError:
        return "Error de conexión con el servicio CNN.", "N/A", 0.0, "N/A", "N/A"
    except Exception as e:
        logger.exception("Error en classify_image")
        return f"Error inesperado: {str(e)}", "N/A", 0.0, "N/A", "N/A"


# =============================
#       GRADIO UI
# =============================

def create_app():
    """
    Crea y configura la aplicación Gradio.
    
    Returns:
        gr.Blocks: Aplicación Gradio configurada
    """
    with gr.Blocks(
        title="Pipeline Inteligente MLOps",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .loader {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loader.show {
            display: block;
        }
        """
    ) as demo:

        gr.Markdown(
            """
            # LLM + ML Clásico + CNN
            
            Plataforma integrada de Machine Learning con tres servicios especializados:
            - **Chat LLM**: Conversación con modelo de lenguaje
            - **ML Clásico**: Clasificación de vinos con scikit-learn
            - **CNN**: Clasificación de imágenes con filtros convolucionales
            """
        )

        with gr.Tab("Chat con LLM"):
            gr.Markdown(
                """
                ### Conversa con el Modelo de Lenguaje
                Escribe cualquier pregunta y obtén respuestas inteligentes.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    llm_input = gr.Textbox(
                        label="Tu pregunta",
                        placeholder="Escribe tu pregunta aquí...",
                        lines=3
                    )
                    llm_button = gr.Button("Enviar", variant="primary")
                
            llm_loader = gr.HTML('<div class="loader" id="llm_loader">Procesando...</div>')
            
            llm_output = gr.Textbox(
                label="Respuesta del LLM",
                lines=10,
                interactive=False
            )
            
            def chat_with_llm_and_loader(prompt):
                yield "", gr.update(visible=True), ""
                result = chat_with_llm(prompt)
                yield result, gr.update(visible=False), ""
            
            llm_button.click(
                fn=chat_with_llm_and_loader,
                inputs=llm_input,
                outputs=[llm_output, llm_loader, llm_input]  
            )

        with gr.Tab("Validación ML Clásico"):
            gr.Markdown(
                """
                ### Clasificación de Vinos
                Ingresa las 13 características del vino para predecir su clase.
                
                **Características:** Alcohol, Ácido málico, Ceniza, Alcalinidad de ceniza, Magnesio, 
                Fenoles totales, Flavonoides, Fenoles no flavonoides, Proantocianinas, 
                Intensidad de color, Tono, OD280/OD315, Prolina
                """
            )
            
            with gr.Row():
                FEATURE_NAMES = [
                    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
                ]
                features_cols = []
                for i in range(13):
                    with gr.Column():
                        feat = gr.Number(
                            label=FEATURE_NAMES[i],
                            value=0.0,
                            precision=2
                        )
                        features_cols.append(feat)
            
            ml_button = gr.Button("Predecir", variant="primary")
            ml_random_button = gr.Button("Generar aleatorio", variant="secondary")
            
            ml_loader = gr.HTML('<div class="loader" id="ml_loader">Procesando predicción...</div>')
            
            with gr.Row():
                with gr.Column():
                    ml_output = gr.Textbox(
                        label="Resultado de Predicción",
                        lines=3,
                        interactive=False
                    )
                with gr.Column():
                    ml_probs = gr.Label(
                        label="Probabilidades por Clase",
                        num_top_classes=3
                    )
            
            def predict_ml_and_loader(*features):
                yield "", {}, gr.update(visible=True)
                result, probs = predict_ml(*features)
                yield result, probs, gr.update(visible=False)
            
            ml_button.click(
                fn=predict_ml_and_loader,
                inputs=features_cols,
                outputs=[ml_output, ml_probs, ml_loader]
            )
            
            def generate_random_features():
                random_values = []
                ranges = [
                    (10, 15),    # Alcohol
                    (0.5, 6),    # Ácido málico
                    (1.5, 3.5),  # Ceniza
                    (15, 25),    # Alcalinidad de ceniza
                    (70, 150),   # Magnesio
                    (0.5, 4),    # Fenoles totales
                    (0.5, 5),    # Flavonoides
                    (0.5, 3),    # Fenoles no flavonoides
                    (0.5, 4),    # Proantocianinas
                    (2, 15),     # Intensidad de color
                    (0.5, 1.5),  # Tono
                    (2, 4),      # OD280/OD315
                    (900, 1600)  # Prolina
                ]
                
                for i in range(13):
                    min_val, max_val = ranges[i % len(ranges)]
                    random_val = round(random.uniform(min_val, max_val), 2)
                    random_values.append(random_val)
                
                return random_values
            
            ml_random_button.click(
                fn=generate_random_features,
                inputs=[],
                outputs=features_cols
            )

        with gr.Tab("Clasificación de Imágenes"):
            gr.Markdown(
                """
                ### Clasificador de Imágenes con CNN
                Sube una imagen para clasificarla usando redes neuronales convolucionales.
                
                **Nota:** El modelo tiene capacidades limitadas. Lee las limitaciones después de la clasificación.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Subir Imagen",
                        type="numpy",
                        height=300
                    )
                    cnn_button = gr.Button("Clasificar", variant="primary")
                
                with gr.Column():
                    cnn_output = gr.Textbox(
                        label="Resultado",
                        lines=4,
                        interactive=False
                    )
                    cnn_class = gr.Textbox(
                        label="Clase Predicha",
                        interactive=False
                    )
                    cnn_conf = gr.Number(
                        label="Nivel de Confianza",
                        precision=4
                    )
            
            with gr.Row():
                cnn_filters = gr.Textbox(
                    label="Filtros Convolucionales Aplicados",
                    value="Gaussian Blur, Edge Detection y Sharpen",
                    interactive=False
                )
                cnn_limitations = gr.Textbox(
                    label="Limitaciones del Modelo",
                    value="El modelo puede tener dificultades con imágenes de baja resolución, objetos parcialmente ocultos, o clases no vistas durante el entrenamiento. La precisión puede variar según la calidad de la imagen de entrada y solo detecta estas clases airplane, automobile, bird",
                    interactive=False,
                    lines=3
                )

            cnn_loader = gr.HTML('<div class="loader" id="cnn_loader">⏳ Clasificando imagen...</div>')
            
            def classify_image_and_loader(image):
                yield "", "N/A", 0.0, "", "", gr.update(visible=True)

                r, c, conf, f, lim = classify_image(image)
                yield r, c, conf, f, lim, gr.update(visible=False)
            
            cnn_button.click(
                fn=classify_image_and_loader,
                inputs=image_input,
                outputs=[cnn_output, cnn_class, cnn_conf, cnn_filters, cnn_limitations, cnn_loader]
            )

        gr.Markdown(
            """
            ---
            **MLOps Pipeline Project** -- Juan Mosquera, Anderson Bornachera 
            """
        )

    return demo


if __name__ == "__main__":
    logger.info("Iniciando aplicación Gradio...")

    import gradio_client.utils
    original_func = gradio_client.utils._json_schema_to_python_type
    
    def safe_json_schema_to_python_type(schema, defs=None):
        try:
            if not isinstance(schema, dict):
                return "Any"
            return original_func(schema, defs)
        except TypeError:
            return "Any"
    
    gradio_client.utils._json_schema_to_python_type = safe_json_schema_to_python_type
    
    app = create_app()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )