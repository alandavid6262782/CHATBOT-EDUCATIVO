import re, unicodedata, ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

# 1. Dataset supervisado (4 intenciones)
DATASET = [
    # Explicación de Conceptos
    ("que significa la gravedad en fisica", "Explicación de Conceptos"),
    ("explicame que es un atomo", "Explicación de Conceptos"),
    ("cual es la definicion de velocidad", "Explicación de Conceptos"),
    ("que es una ecuacion lineal", "Explicación de Conceptos"),
    ("a que se refiere la energia cinetica", "Explicación de Conceptos"),
    ("que es un numero primo", "Explicación de Conceptos"),
    # Resolución de Fórmulas
    ("no se como usar la formula general", "Resolución de Fórmulas"),
    ("como se despeja una ecuacion cuadratica", "Resolución de Fórmulas"),
    ("ayudame a aplicar el teorema de pitagoras", "Resolución de Fórmulas"),
    ("pasos para usar la formula de velocidad", "Resolución de Fórmulas"),
    ("como calculo la hipotenusa", "Resolución de Fórmulas"),
    ("guiame para despejar x en una formula", "Resolución de Fórmulas"),
    # Petición de Ejemplos
    ("dame un ejercicio de algebra resuelto", "Petición de Ejemplos"),
    ("ponme un ejemplo practico de fisica", "Petición de Ejemplos"),
    ("quiero practicar con un problema resuelto", "Petición de Ejemplos"),
    ("muestrame un ejercicio paso a paso", "Petición de Ejemplos"),
    ("dame un problema matematico con su solucion", "Petición de Ejemplos"),
    ("quiero un ejemplo resuelto de pitagoras", "Petición de Ejemplos"),
    # Consulta Fuera de Dominio
    ("cuando es el examen de historia", "Consulta Fuera de Dominio"),
    ("que calificacion saque en espanol", "Consulta Fuera de Dominio"),
    ("quien descubrio america", "Consulta Fuera de Dominio"),
    ("ayudame con mi tarea de ingles", "Consulta Fuera de Dominio"),
    ("cuando empiezan las vacaciones", "Consulta Fuera de Dominio"),
    ("donde consulto mi boleta", "Consulta Fuera de Dominio"),
]

# 2. Preprocesamiento NLP y Red Neuronal (TF-IDF + MLP)
limpiar = lambda t: re.sub(r"[^a-z0-9\s]", "", "".join(
    c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn"
)).strip()

X = [limpiar(f) for f, _ in DATASET]
y = [e for _, e in DATASET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=800, random_state=42)
).fit(X_tr, y_tr)
acc = modelo.score(X_te, y_te)
modelo.fit(X, y)  # Entrenamiento final con todo el dataset

# 3. Base de respuestas e inferencia probabilística
RESPUESTAS = {
    "Explicación de Conceptos": {
        "gravedad": "Definición: Fuerza de atracción entre cuerpos con masa.\nEjemplo real: Al soltar un objeto cae al suelo por la gravedad terrestre (9.81 m/s²).",
        "atomo": "Definición: Unidad mínima de la materia que conserva las propiedades de un elemento.\nEjemplo real: El agua (H₂O) tiene 2 átomos de hidrógeno y 1 de oxígeno.",
        "default": "Definición: Identifica qué propiedad física o matemática mide el concepto y en qué unidades se expresa, relacionándolo con un fenómeno cotidiano."
    },
    "Resolución de Fórmulas": {
        "general": "Fórmula general: x = (-b ± √(b² - 4ac)) / 2a\n1. Iguala a cero (ax² + bx + c = 0).\n2. Identifica a, b, c.\n3. Resuelve la raíz y divide entre 2a.",
        "pitagoras": "Teorema de Pitágoras: c² = a² + b²\n1. Eleva al cuadrado ambos catetos (a y b).\n2. Súmalos y obtén la raíz cuadrada para hallar la hipotenusa (c).",
        "default": "Pasos: 1) Identifica datos e incógnita. 2) Sustituye con unidades. 3) Despeja usando operaciones inversas."
    },
    "Petición de Ejemplos": {
        "algebra": "Ejemplo de álgebra: 4x - 6 = 18\n• Paso 1: Suma 6 → 4x = 24\n• Paso 2: Divide entre 4 → x = 6.",
        "fisica": "Ejemplo de física (F = m·a): Fuerza para acelerar 5 kg a 3 m/s².\n• Solución: F = (5 kg)(3 m/s²) = 15 N.",
        "default": "Ejercicio resuelto: 2(x + 3) = 14 → 2x + 6 = 14 → 2x = 8 → x = 4."
    },
    "Consulta Fuera de Dominio": {
        "default": "Solo brindo tutoría en matemáticas y ciencias básicas (Física y Química). No tengo acceso a calificaciones, fechas ni humanidades."
    }
}

def responder(consulta):
    txt = limpiar(consulta)
    probs = modelo.predict_proba([txt])[0]
    idx = probs.argmax()
    clase, conf = modelo.classes_[idx], probs[idx]
    if conf < 0.45:
        return "Baja confianza", conf, "No estoy seguro de tu duda. Indica si buscas un concepto, una fórmula o un ejemplo de ciencias exactas."
    plantillas = RESPUESTAS[clase]
    resp = next((v for k, v in plantillas.items() if k != "default" and k in txt), plantillas["default"])
    return clase, conf, resp

# 4. Interfaz del chat en Google Colab
salida = widgets.Output(layout={"border": "1px solid #cbd5e1", "padding": "10px", "height": "280px", "overflow_y": "auto"})
caja = widgets.Text(placeholder="Escribe tu duda aquí...", layout={"width": "420px"})
btn = widgets.Button(description="Enviar", button_style="primary")

def enviar(_):
    q = caja.value.strip()
    if not q: return
    caja.value = ""
    clase, conf, resp = responder(q)
    with salida:
        print(f"Tú: {q}\nChatbot_educativo [{clase} | Confianza: {conf*100:.1f}%]:\n{resp}\n" + "-"*50)

btn.on_click(enviar)
with salida:
    print(f"Modelo Neuronal MLP listo (Precisión en prueba: {acc*100:.0f}%).\n" + "-"*50)
display(salida, widgets.HBox([caja, btn]))
