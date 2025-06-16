# 🌿 WebIR 2025 – Detector de Malezas y Evaluador de Imágenes

Este proyecto trata de una aplicación web desarrollada con Flask que permite detectar la presencia de enfermedades o malezas en plantas a través de imágenes, ya sea cargadas por el usuario o buscadas automáticamente en Google. También ofrece una funcionalidad para evaluar la precisión del modelo con datasets personalizados.

---

## 🚀 Características

- 🔍 **Búsqueda de imágenes en Google** mediante SerpApi.
- 📷 **Subida de imágenes manuales** para detección con modelo YOLOv8.
- 🌱 **Detección automática** de plagas y malezas en plantas.
- 📊 **Evaluación del modelo** a partir de archivos CSV con URLs de imágenes.
- 📁 **Interfaz amigable y simple** con dos secciones: Detector y Evaluador.

---

## 🧠 Tecnología utilizada

- Backend: **Python + Flask**
- Frontend: **HTML5, CSS3, JS**
- Modelo: **YOLOv8 (Ultralytics)**
- Otros: **SerpApi, OpenCV, Pandas, Scikit-learn, Pillow**

---

## 🛠️ Instalación

### 1. Clonar el repositorio

### 2. Crear entorno virtual (opcional pero recomendado)

```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```
pip install -r requisitos.txt
```

### 4. Ejecucion

```
python app.py
```
Luego abrir tu navegador y dirigirte a:
👉 http://localhost:5000


## 📂 Estructura del proyecto
```
WEBIR2025/
├── api/
│   ├── app.py                  # Lógica principal de Flask
│   └── templates/
│       └── index.html          # Interfaz web
├── app/
│   └── models/
│       └── best.pt             # Modelo YOLO entrenado
├── presentacion/
│   ├── example_images/         # Imágenes de ejemplo para detección
│   ├── example_and_search.txt  # Términos sugeridos para búsquedas
│   └── example.csv             # CSV de ejemplo para evaluación
├── requisitos.txt              # Dependencias
├── README.md                   # Este archivo
├── PPTs.pdf                    # Presentación del proyecto
└── .gitignore
```


### Nota
En caso de que no funcione el CSV, puede deberse a algun link caido. Verificar antes de probar.


## 👥 Autores
Desarrollado para el curso WebIR - Facultad de Ingeniería - 2025 \
Autores: Guzman Pieroni, Matias Forcelledo, Juan Ignacio Cabrera
