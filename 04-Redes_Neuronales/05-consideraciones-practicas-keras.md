# Tema 5. Aprendizaje profundo y redes neuronales

## Consideraciones prácticas en tareas de Deep Learning con Keras y TensorFlow

### Flujo de trabajo en un problema de Deep Learning

El desarrollo de un modelo de Deep Learning sigue una serie de pasos esenciales que se pueden resumir a continuación

#### Definición del problema

Antes de diseñar una arquitectura de red neuronal, es esencial definir con precisión el problema a resolver. Algunas preguntas clave incluyen:

- ¿Es un problema de clasificación, regresión o detección de patrones?
- ¿Se trabaja con datos estructurados, imágenes o series temporales?
- ¿Cuál es el objetivo del modelo? (ejemplo: predecir ventas, clasificar imágenes, detectar anomalías).
- ¿Cuáles son las restricciones de datos, hardware y tiempo?

| Tipo de problema  | Ejemplo                                        | Salida esperada              |
| ----------------- | ---------------------------------------------- | ---------------------------- |
| Clasificación     | Diagnóstico de enfermedades en imágenes        | Etiqueta discreta (0,1,2...) |
| Regresión         | Predicción del precio de una vivienda          | Valor continuo               |
| Series temporales | Predicción de temperatura en los próximos días | Secuencia de valores         |
| Segmentación      | Detección de objetos en imágenes médicas       | Mapa de etiquetas por píxel  |

#### Preprocesamiento de datos

El preprocesamiento es un paso clave para asegurar que los datos sean compatibles con la red neuronal y evitar sesgos o problemas de convergencia.

##### Datos tabulares

- Normalización o estandarización de valores entre `[0,1]` o con media 0 y varianza 1.
- Manejo de valores faltantes mediante imputación con la media, mediana o valores especiales.
- Codificación de variables categóricas con one-hot encoding o embeddings.

##### Imágenes

- Redimensionado para ajustar todas las imágenes a una misma dimensión, por ejemplo, `224x224`.
- Normalización de píxeles en el rango `[0,1]`.
- Data augmentation con rotaciones, flips y cambios de brillo para mejorar la generalización.

##### Series temporales

- Aplicación de ventana deslizante para transformar datos en secuencias de entrada y salida.
- Diferenciación y suavizado para eliminar tendencias.
- Padding y alineación para asegurar que todas las secuencias tengan la misma longitud.

> [!tip]
>
> El uso de `tf.data.Dataset` mejora la eficiencia en la carga de datos.

#### Elección de arquitectura según el problema

Seleccionar la arquitectura adecuada depende del tipo de datos y de la tarea a resolver.

| Tipo de datos         | Arquitectura recomendada | Ejemplo de aplicación                     |
| --------------------- | ------------------------ | ----------------------------------------- |
| Datos tabulares       | MLP (red neuronal densa) | Predicción de ventas, precios             |
| Imágenes              | CNN (red convolucional)  | Clasificación de objetos, visión          |
| Series temporales     | RNN, LSTM, GRU           | Predicción financiera, IoT                |
| Detección de patrones | Autoencoders, GANs       | Análisis de fraudes, síntesis de imágenes |

Elegir una arquitectura más compleja solo si es necesario, ya que los modelos más simples tienden a ser más interpretables y eficientes.

#### Entrenamiento y optimización

Una vez que los datos están listos y la arquitectura definida, el siguiente paso es entrenar el modelo.

##### Configuraciones a tener en cuenta

- Tamaño del batch entre `32` y `128`, ya que valores grandes aceleran el entrenamiento pero pueden afectar la generalización.
- Uso de optimización con `Adam`, aunque `SGD` con ajuste de tasa de aprendizaje puede ser más adecuado en algunos casos.
- Regularización mediante `Dropout`, `L2` y `Batch Normalization` para evitar sobreajuste.
- Ajuste de hiperparámetros mediante `GridSearch` o `Optuna`.
- Aplicar reducción de tasa de aprendizaje con `ReduceLROnPlateau` si la pérdida deja de mejorar.

#### Evaluación y métricas

El rendimiento de un modelo debe evaluarse con métricas adecuadas al tipo de problema.

| Tipo de problema  | Métrica principal      | Alternativa                 |
| ----------------- | ---------------------- | --------------------------- |
| Clasificación     | Accuracy, F1-score     | AUC, Precision-Recall Curve |
| Regresión         | MSE, MAE               | R², RMSE                    |
| Series temporales | RMSE, MAE              | Correlación de Pearson      |
| Generación        | Perplexity, BLEU score | Rouge-L, Cosine Similarity  |

Si los datos están desbalanceados, evitar accuracy y usar métricas como F1-score o AUC-ROC.

#### Despliegue y consideraciones en producción

Una vez entrenado, el modelo debe ser eficiente y escalable en producción.

##### Conversión y optimización

- Reducción del tamaño del modelo mediante `pruning` y `quantization` con `TensorFlow Lite`.
- Exportación del modelo en formato `SavedModel` o `ONNX`.
- Inferencia eficiente con `batching` y `caching`.

##### Implementación en producción

- Despliegue mediante API REST con `FastAPI` o `Flask`.
- Uso de `TensorFlow Serving` o `Triton Inference Server` para inferencia en tiempo real.
- Escalabilidad en la nube con AWS, GCP o Azure.

Evaluar el impacto del modelo antes del despliegue con técnicas de explicabilidad como `SHAP` o `LIME`.

### Preprocesamiento de datos

Dependiendo del tipo de datos, los procedimientos de preprocesamiento pueden variar significativamente.

#### Datos estructurados

##### Normalización y estandarización

Las redes neuronales son sensibles a la escala de los datos, por lo que es recomendable aplicar técnicas de normalización o estandarización.

Un ejemplo sencillo de código

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()  # O usar StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Ambas técnicas tienen el objetivo de mejorar la estabilidad y convergencia del entrenamiento, evitando gradientes inestables y problemas de escala en los datos. Sin embargo, en Deep Learning influyen factores adicionales, como la activación utilizada, la arquitectura y la sensibilidad a la escala de los datos que hay que tener en cuenta para escoger un método u otro.

###### **Cuándo usar normalización en Deep Learning**

- **Redes con activación sigmoide o tanh**
  - Las funciones **sigmoide** y **tanh** son sensibles a la escala de los datos.
  - La normalización ayuda a evitar saturación en los extremos de la función (donde la derivada es casi cero).
- **Modelos de redes profundas con múltiples capas ocultas**
  - Facilita la propagación del gradiente evitando valores grandes.
  - Es recomendable en arquitecturas **muy profundas** que no utilicen batch normalization.
- **Datos que tienen un rango bien definido**
  - Sensorización o mediciones físicas (temperatura, presión, humedad).
  - Datos de píxeles en imágenes, donde los valores están en `[0,255]` y se normalizan a `[0,1]`.
- **Cuando las unidades de los datos tienen significados distintos**
  - Si hay múltiples características con escalas diferentes (por ejemplo, precio en dólares y peso en kilogramos).
  - Evita que una variable domine el entrenamiento debido a su magnitud.

###### **Cuándo usar estandarización en Deep Learning**

- **Redes con activación ReLU o variantes (LeakyReLU, ELU, GELU)**
  - ReLU no está limitada en su salida, por lo que la normalización no es tan efectiva como la estandarización.
  - Modelos con ReLU pueden manejar datos con mayor dispersión sin afectar la convergencia.
- **Cuando los datos tienen una distribución desconocida o con valores extremos**
  - La estandarización reduce la influencia de los valores atípicos porque ajusta los datos en función de la desviación estándar.
  - Es preferible si hay variables con alta varianza.
- **Cuando los modelos usan Batch Normalization**
  - La estandarización favorece la estabilización de los valores de activación en cada capa.
  - Al aplicar Batch Normalization, los datos se reescalan dinámicamente en cada mini-lote, por lo que una estandarización previa mejora la estabilidad.

##### Manejo de valores faltantes

Los datos estructurados suelen contener valores faltantes que pueden afectar el rendimiento del modelo.

- Eliminación de registros si la cantidad de valores nulos es pequeña.
- Imputación con la media o mediana si los valores siguen una distribución normal o sesgada.
- Uso de valores especiales como `-999` si el modelo debe aprender la ausencia de datos.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")  # O usar "mean"
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

##### Codificación de variables categóricas

Las redes neuronales no pueden procesar directamente datos categóricos, por lo que es necesario codificarlos numéricamente.

- One-hot encoding: Útil para categorías sin orden (ciudad, color, país).
- Label encoding: Se usa en categorías con orden lógico (bajo, medio, alto).

Ejemplo de código

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# Datos categóricos de ejemplo
X_train = np.array(["rojo", "azul", "verde", "azul", "rojo", "verde"]).reshape(-1, 1)

# OneHotEncoder
encoder_ohe = OneHotEncoder(sparse=False)
X_onehot = encoder_ohe.fit_transform(X_train)

print("One-Hot Encoding:")
print(X_onehot)

# LabelEncoder
encoder_label = LabelEncoder()
X_label = encoder_label.fit_transform(X_train.ravel())

print("\nLabel Encoding:")
print(X_label)

```

#### Datos de imágenes

Las imágenes deben ser convertidas en **tensores** antes de ser utilizadas en redes neuronales convolucionales (CNN).

##### Redimensionado y normalización

Las imágenes pueden tener tamaños variados, por lo que es necesario redimensionarlas a una dimensión estándar. Además, los valores de los píxeles deben ser normalizados.

```python
import tensorflow as tf

def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])  # Redimensionar a 224x224
    image = image / 255.0  # Normalización en el rango [0,1]
    return image, label
```

##### Aumento de datos

El aumento de datos puede mejorar la generalización del modelo al aplicar transformaciones aleatorias a las imágenes de entrenamiento.

- Rotaciones y traslaciones para simular variaciones en la toma de imágenes.
- Flips horizontales y verticales para aumentar la diversidad de los datos.
- Ajuste de brillo y contraste para mejorar la robustez del modelo ante cambios de iluminación.

Ejempo de código

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1)
])
```

#### Datos en series temporales

El preprocesamiento de datos secuenciales, como mediciones de sensores o registros financieros, requiere la conversión de datos en ventanas de tiempo.

##### Creación de ventanas deslizantes

Los modelos de aprendizaje profundo requieren que las series temporales sean transformadas en secuencias de entrada-salida.

Ejemplo de función personalizada que genera un dataset seq2seq a partir de ventana deslizante

```python
import numpy as np

def genera_secuencia(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Datos de ejemplo (serie temporal)
serie = [100, 102, 105, 110, 120]

X_train_seq, y_train_seq = genera_secuencia(serie, window_size=2)
```

También existe la posibilidad de usar ventanas deslizantes variables, aunque esta técnica es mucho más habitual en NLP. Sin embargo, también aparecen en **series temporales, aprendizaje por refuerzo y sistemas de predicción secuencial**. Son útiles cuando la dependencia entre eventos puede variar en longitud y se desea permitir que el modelo aprenda de secuencias progresivamente más largas.

```python
import numpy as np

def generar_secuencias_variable(serie):
    X, y = [], []
    for i in range(1, len(serie)):
        X.append(serie[:i])  # Secuencia creciente
        y.append(serie[i])    # Predicción del siguiente valor
    return X, y

# Datos de ejemplo (serie temporal)
serie = [100, 102, 105, 110, 120]

# Generar secuencias variables
X_variable, y_variable = generar_secuencias_variable(serie)
```

##### Diferenciación y suavizado

En el **preprocesamiento de series temporales**, la **diferenciación** y el **suavizado** son técnicas clave para mejorar la capacidad predictiva del modelo y reducir el ruido en los datos.

La **diferenciación** consiste en calcular la diferencia entre valores consecutivos de la serie temporal para eliminar tendencias y hacer que los datos sean estacionarios. Será interesante usar diferenciación en los siguientes casos:

- Cuando los datos muestran una **tendencia creciente o decreciente**.
- Para hacer la serie **estacionaria**, lo que es útil en modelos como ARIMA o redes LSTM.
- Si los valores absolutos tienen mucha variabilidad, pero los cambios relativos son más predecibles.

Este es un ejemplo práctico de código

```python
import numpy as np
import matplotlib.pyplot as plt

# Serie de datos con tendencia
serie = np.array([100, 105, 110, 120, 130, 145])

# Aplicar diferenciación (resta del valor anterior)
serie_diferenciada = np.diff(serie)

print("Serie original:", serie)
print("Serie diferenciada:", serie_diferenciada)

# Visualización
plt.figure(figsize=(8,4))
plt.plot(serie, label="Serie original", marker='o')
plt.plot(range(1, len(serie)), serie_diferenciada, label="Serie diferenciada", marker='s')
plt.legend()
plt.show()
```

Puede verse como la serie diferenciada muestra los incrementos entre valores en lugar de los valores absolutos. Pueden usarse variantes de diferenciación **de primer o segundo orden**

Por su parte, el **suavizado** ayuda a reducir el ruido en una serie temporal manteniendo su tendencia principal. Es útil cuando los datos contienen **fluctuaciones irregulares** que podrían afectar el modelo. Es decir, los casos en los que usar suavizado serían:

- Cuando los datos tienen **variabilidad alta o ruido**, lo que dificulta la detección de patrones.
- Si el modelo tiene dificultades para aprender debido a **datos demasiado irregulares**.
- Para preprocesar series en **modelos de predicción financiera, meteorológica o industrial**.

Existen varias técnicas de suavizado, entre las que destacamos las siguientes

1. **Media móvil simple**: Calcula el promedio de los últimos *n* valores para cada punto.
2. **Media móvil exponencial (EMA)**: Da más peso a valores recientes.
3. **Filtro de Savitzky-Golay**: Mantiene la estructura de la señal mientras la suaviza.

Sin embargo, es muy habitual usar el primero de ellos. A continuación tienes un ejemplo de código

```python
def media_movil(serie, ventana=3):
    return np.convolve(serie, np.ones(ventana)/ventana, mode='valid')

# Aplicar suavizado con ventana de 3
serie_suavizada = media_movil(serie, ventana=3)

print("Serie original:", serie)
print("Serie suavizada:", serie_suavizada)

# Visualización
plt.figure(figsize=(8,4))
plt.plot(serie, label="Serie original", marker='o')
plt.plot(range(2, len(serie)), serie_suavizada, label="Serie suavizada (Media Móvil)", marker='s')
plt.legend()
plt.show()
```

Puede observarse en la salida como la serie suavizada sigue la tendencia general, pero elimina fluctuaciones bruscas.

###### **Comparación rápida: Diferenciación vs. Suavizado**

| **Técnica**        | **Objetivo**                                      | **Cuándo usarla**                                            |
| ------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| **Diferenciación** | Eliminar tendencia y hacer la serie estacionaria. | Si los datos tienen tendencia fuerte y el modelo requiere estacionariedad. |
| **Suavizado**      | Reducir ruido y preservar la tendencia.           | Si los datos tienen fluctuaciones aleatorias que dificultan el aprendizaje. |

Es importante tener en cuenta que ambas técnicas pueden combinarse. Por ejemplo, en **predicción financiera**, es común **suavizar primero** para reducir ruido y luego **diferenciar** para eliminar tendencia antes de aplicar modelos como LSTM.

A continuación tienes un ejemplo de código donde se combinan ambas técnicas. Primero se suaviza la serie para después aplicar una diferenciación de primer orden y una posterior normalización de los datos

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Datos de ejemplo (Serie con tendencia y ruido)
np.random.seed(42)
tiempo = np.arange(30)
serie = 50 + tiempo * 2 + np.random.normal(scale=5, size=len(tiempo))  # Tendencia + Ruido

# 2. Suavizado: Media Móvil Simple
def media_movil(serie, ventana=3):
    return np.convolve(serie, np.ones(ventana)/ventana, mode='valid')

serie_suavizada = media_movil(serie, ventana=3)

# 3. Diferenciación: Eliminar tendencia
serie_diferenciada = np.diff(serie_suavizada)

# 4. Normalización de los datos (Min-Max Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
serie_normalizada = scaler.fit_transform(serie_diferenciada.reshape(-1, 1)).flatten()

# Visualización de las transformaciones
plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(serie, label="Serie original", marker='o')
plt.legend()

plt.subplot(3,1,2)
plt.plot(serie_suavizada, label="Serie suavizada (Media Móvil)", marker='s')
plt.legend()

plt.subplot(3,1,3)
plt.plot(serie_normalizada, label="Serie diferenciada y normalizada", marker='x')
plt.legend()

plt.show()

```

#### Generación de datasets en tf.data

El uso de `tf.data` optimiza el manejo de grandes volúmenes de datos y permite aplicar transformaciones en paralelo.

##### Creación de dataset desde tensores

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```



### **Modelización y arquitecturas en Deep Learning**

La selección de la arquitectura en Deep Learning depende del tipo de problema y los datos disponibles. A continuación, se describen los principales enfoques y consideraciones prácticas para cada tipo de red.

#### **Perceptrón y MLP**

Las redes neuronales multicapa (MLP) son la opción clásica para trabajar con **datos tabulares**, problemas de **regresión** y **clasificación**. Se basan en capas completamente conectadas (*fully connected layers*), lo que permite modelar relaciones entre variables de entrada.

##### **Casos de uso**

- **Regresión numérica:** Predicción de valores continuos (ejemplo: precios de viviendas).
- **Clasificación:** Identificación de categorías a partir de datos estructurados.
- **Procesamiento de datos tabulares:** Problemas en los que los atributos no tienen relaciones espaciales o temporales claras.

##### **Selección de capas y funciones de activación**

- **Número de capas ocultas:** Entre `1 y 3` suele ser suficiente para datos tabulares.
- **Número de neuronas:** Se suele elegir entre el tamaño de la entrada y el tamaño de la salida.
- **Activaciones recomendadas:**
  - `ReLU`: Función estándar en capas ocultas para evitar el problema del gradiente.
  - `Softmax`: Para clasificación multiclase en la capa de salida.
  - `Sigmoid`: Para clasificación binaria.

Ejemplo de un modelo que se puede aplicar a tareas de **clasificación binaria**.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Suponemos que se han generado un X_train y un y_train

# Definir el modelo de perceptrón
modelo = Sequential([
    Dense(1, activation='sigmoid', input_dim=X_train.shape[1])  # Uso de input_dim en lugar de input_shape
])

# Compilación del modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen del modelo
modelo.summary()

```

La siguiente tabla resume las opciones de trabajo habituales con un perceptron

| **Tarea**                    | **Salida**              | **Activación** | **Función de pérdida**     |
| ---------------------------- | ----------------------- | -------------- | -------------------------- |
| **Clasificación binaria**    | 1 neurona (`Dense(1)`)  | `sigmoid`      | `binary_crossentropy`      |
| **Clasificación multiclase** | N neuronas (`Dense(N)`) | `softmax`      | `categorical_crossentropy` |
| **Regresión**                | 1 neurona (`Dense(1)`)  | `linear`       | `mean_squared_error (MSE)` |

##### Cuándo usar perceptrones y cuándo usar redes MLP

El **perceptrón simple** es útil para problemas **linealmente separables**, pero en la mayoría de los casos del mundo real, los datos no son perfectamente lineales. Para abordar problemas más complejos, es necesario **introducir capas ocultas**, convirtiendo el modelo en una **red MLP**.

A continuación, se presentan algunos escenarios en los que un **perceptrón simple no es adecuado**, y se recomienda pasar a una **MLP**.

- **Clasificación de datos no linealmente separables**. Por ejemplo:
  - Identificar la región donde caen puntos en un plano cartesiano. Una red MLP puede aprender **fronteras de decisión más complejas** usando combinaciones no lineales de características.
  - Clasificación de imágenes simples donde las diferencias entre clases no pueden ser expresadas con una única frontera lineal.
  - Problemas como **XOR**, donde los datos positivos y negativos no pueden separarse con una sola línea.

- **Procesamiento de datos estructurados con interacciones complejas**. Por ejemplo:
  - Un modelo que predice si un cliente comprará un producto en función de múltiples factores (edad, ingresos, frecuencia de compra). → la red MLP aprende **interacciones no evidentes entre variables** y puede capturar patrones más complejos.
  - Predicción de fraudes bancarios donde múltiples variables interactúan de forma no trivial.
  - Diagnóstico médico con datos tabulares en los que varias características pueden combinarse para indicar una condición.

- **Reducción de dimensionalidad y extracción de características**. Por ejemplo:
  - Un conjunto de datos con **múltiples atributos redundantes** → En **análisis de clientes**, variables como “salario” y “gasto mensual” pueden estar correlacionadas. Una **MLP** puede aprender representaciones más compactas sin necesidad de ingeniería de características manual.

- **Tareas que requieren abstracciones jerárquicas**. Por ejemplo:
  - Clasificación de imágenes donde es necesario detectar bordes, texturas y formas antes de clasificar un objeto → Una red MLP puede ser útil en el reconocimiento de escritura manual o clasificación de imágenes simples en datasets pequeños donde no se requiere una CNN.

Pasar de un modelo de perceptrón a una red MLP tiene riesgos y ventajas. Las ventajas podrían ser las siguientes:

- **Captura de relaciones no lineales**: Una MLP con activaciones no lineales (como `ReLU` o `sigmoid`) puede modelar datos más complejos.
- **Mejor generalización**: Al aumentar la capacidad del modelo, se pueden aprender patrones más robustos.
- **Posibilidad de ajuste fino**: Permite optimizar la arquitectura agregando más neuronas o capas según sea necesario.
- **Extracción de características**: Las capas ocultas pueden aprender representaciones útiles del problema sin necesidad de un diseño manual de atributos.

Pero también existen riesgos:

- **Mayor riesgo de sobreajuste**: Una red más grande puede memorizar en lugar de generalizar, especialmente en conjuntos de datos pequeños.
- **Mayor costo computacional**: Más neuronas y capas requieren mayor poder de cómputo y tiempos de entrenamiento más largos.
- **Más hiperparámetros a ajustar**: Número de capas, neuronas, tasa de aprendizaje, regularización… aumenta la complejidad del ajuste del modelo.
- **Dificultad en la interpretabilidad**: Un perceptrón es fácil de interpretar, pero una MLP se vuelve una “caja negra”.

###### Conclusiones ¿Cuándo NO ES recomendable pasar a una MLP?

Cuando ocurra alguna de las siguientes situaciones:

- **Los datos son linealmente separables** (ejemplo: un dataset simple con dos clases claramente diferenciadas).
- **El dataset es muy pequeño**, ya que una MLP puede sobreajustarse con facilidad.
- **El modelo debe ser interpretable**, como en aplicaciones donde es crucial entender por qué se toma una decisión (por ejemplo, en finanzas o salud).
- **El problema no justifica la complejidad extra**, como en tareas simples de regresión o clasificación con pocas variables relevantes

#### **Redes convolucionales (CNN)**

Las **CNN** son especialmente efectivas en tareas de **procesamiento de imágenes**, aunque también se aplican en otros dominios del DL. Su estructura captura **patrones espaciales** mediante operaciones de convolución.

##### **Preprocesamiento de imágenes**

A continuación vamos a detallar algunas consideraciones prácticas en el caso de preprocesamiento de imágenes a la hora de usar redes convolucionales ya que es un paso fundamental para optimizar el rendimiento de una **Red Neuronal Convolucional (CNN)**.

Keras proporciona herramientas específicas para manejar imágenes antes de alimentarlas a una red neuronal convolucional

###### **Carga de imágenes con `ImageDataGenerator`**

`ImageDataGenerator` permite cargar imágenes desde directorios y aplicar transformaciones en tiempo real. Veamos un ejemplo de cómo funciona.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,         # Normalización de valores de píxeles a [0,1]
    rotation_range=20,      # Rotación aleatoria de 20 grados
    width_shift_range=0.2,  # Desplazamiento horizontal del 20%
    height_shift_range=0.2, # Desplazamiento vertical del 20%
    shear_range=0.2,        # Transformación de corte
    zoom_range=0.2,         # Zoom aleatorio del 20%
    horizontal_flip=True,   # Inversión horizontal
    fill_mode='nearest'     # Modo de relleno de píxeles fuera del marco
)

train_generator = datagen.flow_from_directory(
    'dataset/train',        # Carpeta donde están las imágenes
    target_size=(150, 150), # Redimensionar todas las imágenes a 150x150
    batch_size=32,
    class_mode='categorical' # Modo de clasificación (binario/multiclase)
)

# Obtener un batch de datos
X_batch, y_batch = next(train_generator)

# Ver formas de los datos generados
print(f"Forma de X_batch: {X_batch.shape}")  # (32, 150, 150, 3)
print(f"Forma de y_batch: {y_batch.shape}")  # (32, N_clases)
```

`ImageDataGenerator` devuelve **batches de imágenes y etiquetas en formato NumPy**, listos para ser utilizados en el entrenamiento de un modelo CNN en Keras. Es interesante usar  `ImageDataGenerator` cuando:

- Se tiene un **dataset de imágenes almacenadas en carpetas**.
- Se quiere realizar **data augmentation** en tiempo real.
- Se tiene un **dataset pequeño** y se necesita aumentar su diversidad.

###### **Preprocesamiento manual de imágenes**

Si las imágenes ya están cargadas en memoria como matrices NumPy, se pueden procesar directamente.

Por ejemplo, para cargar imágenes desde archivos y transformarlas en tensores podemos seguir el siguiente código de ejemplo

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar imagen
img_path = 'imagen.jpg'
img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar

# Convertir imagen a tensor NumPy
img_array = image.img_to_array(img)

# Expandir dimensión para simular batch (CNN espera batch de imágenes)
img_array = np.expand_dims(img_array, axis=0)

# Normalización (opcional)
img_array /= 255.0

print(f"Forma de la imagen procesada: {img_array.shape}")
```

Podemos usar preprocesamiento manual cuando:

- Se trabaja con **imágenes individuales** en tareas de inferencia.
- Se tiene **control total sobre la manipulación de los datos**.

###### **Conversión de imágenes en `tf.data.Dataset`**

Para mejorar el rendimiento del entrenamiento en TensorFlow, se puede utilizar `tf.data.Dataset` para manejar grandes volúmenes de imágenes de manera eficiente.

```python
import tensorflow as tf

# Función para cargar y preprocesar imágenes
def cargar_y_preprocesar_imagen(ruta):
    img = tf.io.read_file(ruta)
    img = tf.image.decode_jpeg(img, channels=3)  # Decodificar JPEG
    img = tf.image.resize(img, [150, 150])  # Redimensionar
    img = img / 255.0  # Normalización
    return img

# Lista de rutas de imágenes
rutas_imagenes = tf.data.Dataset.list_files("dataset/train/*.jpg")

# Crear dataset aplicando la función de preprocesamiento
dataset = rutas_imagenes.map(cargar_y_preprocesar_imagen).batch(32).prefetch(tf.data.AUTOTUNE)
```

###### **Técnicas clave de preprocesamiento y su impacto en CNN**

Se resumen en la siguiente tabla:

| **Técnica**             | **Descripción**                                             | **Cuándo usarla**                             |
| ----------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| **Normalización**       | Convierte valores de píxeles a `[0,1]` o `[-1,1]`           | Siempre (mejora la estabilidad)               |
| **Estandarización**     | Sustrae la media y divide por la desviación estándar        | Cuando la distribución es desigual            |
| **Redimensionamiento**  | Ajusta el tamaño de la imagen                               | Si el modelo requiere entradas de tamaño fijo |
| **Data augmentation**   | Genera variaciones artificiales de las imágenes             | Para evitar sobreajuste en datasets pequeños  |
| **Conversión a tensor** | Convierte imágenes en arrays NumPy o tensores de TensorFlow | Para entrenamiento eficiente                  |

##### **Consideraciones prácticas en la modelización de distintas arquitecturas CNN**

Al modelizar redes convolucionales (CNN), la arquitectura debe adaptarse al **tamaño del dataset, la complejidad del problema y la capacidad computacional**. A continuación, se presentan consideraciones prácticas para **arquitecturas desde simples hasta avanzadas**.

###### **Arquitectura CNN simple (para datasets pequeños)**

Casos de uso:

- Cuando el dataset es **pequeño (~<10.000 imágenes)** y se busca un modelo liviano.  
- Para problemas de **clasificación binaria o multiclase** con imágenes simples.  

Estrategia recomendada:
- Pocas capas convolucionales (2-4).
- Filtros pequeños (`3x3`).
- Capas de pooling (`MaxPooling2D`) para reducir dimensionalidad.
- **Evitar sobreajuste** con `Dropout` (30-50%).

Ejemplo de modelo CNN simple:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

modelo = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Reduce sobreajuste
    Dense(10, activation='softmax')  # 10 clases
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
###### **Resumen de mejores prácticas según arquitectura**
| **Arquitectura**       | **Dataset ideal**       | **Características clave**                             | **Cuándo usarla**                               |
| ---------------------- | ----------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| **CNN simple**         | Pequeño (<10k imágenes) | 2-4 capas, MaxPooling, Dropout                        | Clasificación básica                            |
| **VGG16/VGG19**        | Mediano (~50k imágenes) | Profundidad moderada, Transfer Learning               | Clasificación de imágenes con Transfer Learning |
| **ResNet50/101**       | Grande (>100k imágenes) | Atajos residuales, evita desvanecimiento de gradiente | Imágenes con muchos detalles                    |
| **YOLO, Faster R-CNN** | Variable                | Bounding boxes, detección de objetos                  | Detección de múltiples objetos en imágenes      |

###### Caso práctico completo de procesamiento de imágenes y clasificación binaria

El presente caso cubre **todo el flujo de trabajo** desde la carga de imágenes hasta el entrenamiento de una **CNN simple** en Keras, siguiendo las mejores prácticas. Supongamos que estamos ante un problema de clasificación de imágenes en dos categorías: **perros** 🐶 y **gatos** 🐱. Las imágenes están organizadas en dos carpetas dentro de un directorio

```
dataset/
├── perros/  # 1000 imágenes de perros
├── gatos/   # 1000 imágenes de gatos
```

Cada imagen tiene dimensiones distintas y está en color (`RGB`).

Pasos en el código:

1. **Carga de imágenes y preprocesamiento** con `ImageDataGenerator`.
2. **Definición de una CNN simple** para la clasificación binaria.
3. **Entrenamiento del modelo** usando las imágenes preprocesadas.
4. **Evaluación y predicción en nuevas imágenes**.

```python
# Importar librerías necesarias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1️⃣ CONFIGURACIÓN Y PREPROCESAMIENTO DE IMÁGENES

# Directorio donde están las imágenes organizadas en carpetas (perros/gatos)
ruta_dataset = "dataset/"

# Definir generador de datos con aumentación para entrenar la red
datagen_entrenamiento = ImageDataGenerator(
    rescale=1./255,  # Normalización: escala valores entre 0 y 1
    rotation_range=20,  # Rotar imágenes aleatoriamente hasta 20 grados
    width_shift_range=0.1,  # Pequeños desplazamientos horizontales
    height_shift_range=0.1,  # Pequeños desplazamientos verticales
    shear_range=0.2,  # Transformación en corte
    zoom_range=0.2,  # Zoom aleatorio hasta 20%
    horizontal_flip=True,  # Voltear imágenes horizontalmente
    validation_split=0.2  # Separar un 20% de datos para validación
)

# Cargar imágenes para entrenamiento
generador_train = datagen_entrenamiento.flow_from_directory(
    ruta_dataset,  # Carpeta raíz con subcarpetas de clases
    target_size=(150, 150),  # Redimensionar todas las imágenes a 150x150 píxeles
    batch_size=32,  # Tamaño de lotes
    class_mode="binary",  # Clasificación binaria (perros o gatos)
    subset="training"  # Usar el 80% de los datos para entrenamiento
)

# Cargar imágenes para validación
generador_validacion = datagen_entrenamiento.flow_from_directory(
    ruta_dataset,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"  # Usar el 20% restante para validación
)

# 2️⃣ DEFINICIÓN DEL MODELO CNN

# Modelo CNN simple
modelo = Sequential([
    # Primera capa convolucional con 32 filtros de 3x3
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2,2)),  # Reduce dimensiones de 150x150 -> 75x75
    
    # Segunda capa convolucional con 64 filtros
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),  # Reduce dimensiones a 37x37
    
    # Tercera capa convolucional con 128 filtros
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),  # Reduce dimensiones a 18x18
    
    # Aplanado de la imagen en un vector
    Flatten(),
    
    # Capa densa con 128 neuronas
    Dense(128, activation="relu"),
    Dropout(0.5),  # Regularización para evitar sobreajuste
    
    # Capa de salida con activación sigmoide para clasificación binaria
    Dense(1, activation="sigmoid")
])

# 3️⃣ COMPILACIÓN DEL MODELO

modelo.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam con tasa de aprendizaje baja para estabilidad
    loss="binary_crossentropy",  # Pérdida adecuada para clasificación binaria
    metrics=["accuracy"]  # Métrica de precisión
)

# 4️⃣ ENTRENAMIENTO DEL MODELO

# Entrenar la CNN usando el generador de imágenes
historial = modelo.fit(
    generador_train,  # Conjunto de entrenamiento
    validation_data=generador_validacion,  # Conjunto de validación
    epochs=10,  # Número de iteraciones sobre el dataset
    verbose=1  # Mostrar progreso
)

# 5️⃣ EVALUACIÓN Y PRUEBA EN NUEVAS IMÁGENES

import numpy as np
from tensorflow.keras.preprocessing import image

def predecir_imagen(ruta_imagen):
    """ Función para predecir si una imagen es de un perro o un gato """
    
    img = image.load_img(ruta_imagen, target_size=(150, 150))  # Cargar imagen
    img_array = image.img_to_array(img)  # Convertir a array NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para modelo
    img_array /= 255.0  # Normalizar
    
    prediccion = modelo.predict(img_array)  # Realizar predicción
    clase = "Perro" if prediccion[0][0] > 0.5 else "Gato"  # Interpretar resultado
    
    print(f"Predicción: {clase} (confianza: {prediccion[0][0]:.4f})")

# Ejemplo de predicción
predecir_imagen("ejemplo.jpg")
```

**Explicación de cada paso**

**1️⃣ Preprocesamiento de imágenes**

- Se usa `ImageDataGenerator` para **cargar, redimensionar y normalizar** imágenes.
- Se aplican **técnicas de aumentación de datos** (rotaciones, zoom, desplazamientos) para mejorar la generalización.
- Se separa **80% para entrenamiento y 20% para validación** con `validation_split=0.2`.

**2️⃣ Definición del modelo CNN**

- Se usa una arquitectura con **3 capas convolucionales** y `MaxPooling2D` para reducir dimensiones.
- La activación **ReLU** ayuda a mantener la no linealidad.
- Se usa `Dropout(0.5)` para **evitar sobreajuste** en la capa densa.

**3️⃣ Compilación del modelo**

- Se usa **Adam** con `learning_rate=0.0001` para estabilidad.
- Se define la función de pérdida **`binary_crossentropy`** para clasificación binaria.
- Se monitorea la **precisión (`accuracy`)**.

**4️⃣ Entrenamiento del modelo**

- Se usa `modelo.fit()` con `epochs=10` (ajustable según rendimiento).
- Se entrena en imágenes ya **preprocesadas con `ImageDataGenerator`**.

**5️⃣ Evaluación y predicción**

- Se implementa `predecir_imagen()` para cargar y predecir imágenes nuevas.
- Se normaliza la imagen y se pasa por la red CNN para obtener **perro o gato**.

#### **Redes recurrentes (RNN, LSTM, GRU)**

Las **redes recurrentes** están diseñadas para trabajar con **secuencias de datos** y capturar dependencias temporales. Se aplican en tareas como **series temporales, procesamiento de lenguaje natural y predicción de señales**.

##### **Aplicaciones en series temporales**

- **Predicción financiera:** Modelado de tendencias en bolsas de valores.
- **Análisis de sensores:** Predicción de fallos en sistemas industriales.
- **Reconocimiento de voz:** Procesamiento de señales de audio.

##### **Diferencias entre RNN, LSTM y GRU**

| **Arquitectura** | **Ventajas**                                            | **Desventajas**                                             |
| ---------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| **RNN**          | Captura dependencias temporales.                        | Problema de gradientes desvanecientes, difícil de entrenar. |
| **LSTM**         | Maneja dependencias largas mediante puertas de control. | Más costosa computacionalmente.                             |
| **GRU**          | Similar a LSTM pero con menos parámetros.               | Puede no capturar dependencias tan largas como LSTM.        |

### **Optimización y entrenamiento en Deep Learning**

El entrenamiento de redes neuronales profundas requiere el ajuste cuidadoso de varios elementos clave, como el **algoritmo de optimización, los hiperparámetros y las técnicas de regularización**. Veamos algunas consideraciones prácticas para mejorar el rendimiento de los modelos de deep learning.

#### **Optimización en redes neuronales**

Los modelos de deep learning entrenan mediante **descenso de gradiente**, donde el objetivo es minimizar una función de pérdida ajustando los pesos del modelo. Existen diferentes algoritmos de optimización que afectan la rapidez y estabilidad del entrenamiento.

| **Optimizador**                       | **Descripción**                                              | **Cuándo usarlo**                                            |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SGD (Stochastic Gradient Descent)** | Optimización clásica basada en gradiente descendente. Puede incluir momentum. | Modelos grandes, pero con una tasa de aprendizaje bien ajustada. Puede ser inestable si no se usa momentum. |
| **Adam**                              | Combina **momentum** y **RMSprop**, adaptando la tasa de aprendizaje en cada paso. | Recomendado por defecto en la mayoría de los modelos. Estable y rápido en convergencia. |
| **RMSprop**                           | Divide la tasa de aprendizaje entre la media cuadrática de gradientes recientes. | Bueno para **series temporales y RNN**, donde los gradientes pueden volverse inestables. |

En **TensorFlow/Keras**, el optimizador se define en la compilación del modelo:

```python
from tensorflow.keras.optimizers import Adam

modelo.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
```

Consideraciones prácticas

- **Si el modelo no converge**: prueba **Adam** con una tasa de aprendizaje más baja.
- **Si el modelo oscila mucho**: usa **SGD con momentum** (`momentum=0.9`).
- **Si hay problemas de gradiente en RNN**: prueba **RMSprop**.

##### **Ajuste de hiperparámetros**

El rendimiento del modelo depende de la configuración de los **hiperparámetros**, como el tamaño del batch, la tasa de aprendizaje y la arquitectura.

###### **Batch size**

Define cuántas muestras se procesan antes de actualizar los pesos. Se recomienda:

| **Tamaño del dataset**     | **Batch size recomendado** |
| -------------------------- | -------------------------- |
| Pequeño (<10,000 muestras) | 8 - 32                     |
| Mediano (10,000 - 100,000) | 32 - 128                   |
| Grande (>100,000)          | 128 - 512                  |

Ejemplo en Keras:

```python
modelo.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
```

Consideraciones prácticas:

- **Batch pequeño (<32)**: mejora la generalización, pero el entrenamiento es más lento.
- **Batch grande (>128)**: acelera el entrenamiento, pero puede reducir la capacidad de generalización.

###### **Tasa de aprendizaje (`learning_rate`)**

Controla cuánto cambian los pesos en cada iteración.

| **Problema**         | **Posible solución**     |
| -------------------- | ------------------------ |
| Modelo no aprende    | Aumentar `learning_rate` |
| Pérdida oscila mucho | Reducir `learning_rate`  |

Ajuste con `Adam`:

```python
Adam(learning_rate=0.0001)
```

##### **Dropout**

Reduce el sobreajuste eliminando neuronas aleatorias en cada iteración:

```python
from tensorflow.keras.layers import Dropout

modelo.add(Dense(128, activation="relu"))
modelo.add(Dropout(0.5))  # Desactiva el 50% de las neuronas en cada paso
```

Cuándo usarlo:

- En **capas densas** (`0.2 - 0.5` es un buen rango).
- No se usa en **redes convolucionales antes de Flatten** (puede eliminar demasiada información).

##### **Técnicas de regularización**

La regularización evita que el modelo se sobreajuste a los datos de entrenamiento.

###### **L2 Regularization (Weight Decay)**

Añade una penalización sobre los pesos grandes:

```python
from tensorflow.keras.regularizers import l2

modelo.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
```

Cuándo usarlo:

- Si el modelo se sobreajusta.
- En **MLP y CNN** para restringir pesos excesivos.

###### **Batch Normalization**

Normaliza la activación en cada capa para estabilizar el entrenamiento:

```python
from tensorflow.keras.layers import BatchNormalization

modelo.add(Dense(128, activation="relu"))
modelo.add(BatchNormalization())  # Normaliza después de la activación
```

Cuándo usarlo:

- Si el modelo es **muy profundo**.
- Si la pérdida fluctúa demasiado durante el entrenamiento.

##### **Resumen de estrategias clave** de optimización

| **Problema detectado** | **Solución práctica**                                       |
| ---------------------- | ----------------------------------------------------------- |
| Modelo no aprende      | Aumentar `learning_rate`, probar `SGD + momentum`           |
| Sobreajuste            | Añadir `Dropout`, `L2 regularization`, `BatchNormalization` |
| Entrenamiento lento    | Usar `batch_size` más grande, optimizador Adam              |
| Pérdida no mejora      | Usar `ReduceLROnPlateau`, probar `BatchNormalization`       |
| Pérdida oscila mucho   | Reducir `learning_rate`, usar `SGD` en lugar de Adam        |

##### **Ejemplo de implementación completa de optimizaciones**

A continuación tienes un ejemplo de un modelo **entrenado con todas las optimizaciones**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Definir modelo CNN
modelo = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compilar modelo con Adam
modelo.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Definir callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Entrenar modelo
modelo.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)
```

#### **Consideraciones prácticas en la fase de análisis del modelo**  

El entrenamiento de un modelo en deep learning no solo depende de los hiperparámetros, sino también de una correcta inspección del modelo y de estrategias prácticas para gestionar su entrenamiento de manera eficiente. 

##### **Uso de `model.summary()` para analizar la arquitectura del modelo**  

Antes de entrenar un modelo, es fundamental verificar su arquitectura, el número de parámetros y la conectividad entre capas. La función `model.summary()` en Keras proporciona esta información de manera clara y estructurada.  

Ejemplo de uso para un modelo CNN:

```python
modelo.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 32)      896       
max_pooling2d (MaxPooling2D)  (None, 74, 74, 32)       0         
conv2d_1 (Conv2D)            (None, 72, 72, 64)       18496     
max_pooling2d_1 (MaxPooling2D) (None, 36, 36, 64)     0         
flatten (Flatten)            (None, 82944)            0         
dense (Dense)                (None, 128)              10616960  
dropout (Dropout)            (None, 128)              0         
dense_1 (Dense)              (None, 1)                129       
=================================================================
Total params: 10,635,481
Trainable params: 10,635,481
Non-trainable params: 0
_________________________________________________________________
```

---

Al usar `model.summary()` es importante considerar las siguientes cuestiones: 

- **Verificación de la estructura del modelo:**  
   - Comprueba que cada capa tiene las dimensiones esperadas.
   - Verifica que la capa de salida tiene la cantidad correcta de neuronas y activación (`softmax` para clasificación multiclase, `sigmoid` para clasificación binaria, etc.).

- **Número de parámetros entrenables:**  
   - Modelos muy grandes pueden necesitar más datos o regularización.
   - Si el número de **parámetros no entrenables** es alto, revisa si hay capas congeladas (por ejemplo, en Transfer Learning).

- **Consistencia de la propagación de dimensiones:**  
   - La salida de cada capa debe coincidir con la entrada de la siguiente.
   - En redes convolucionales, el tamaño de las imágenes debe ir reduciéndose progresivamente antes de llegar a `Flatten()`.

- **Uso en modelos preentrenados:**  
   - Cuando se usa Transfer Learning, es útil verificar cuántos parámetros son **entrenables** (`Trainable params`) y cuántos se mantienen fijos (`Non-trainable params`).

#### **Consideraciones prácticas en la fase de entrenamiento**  

Una vez que el modelo ha sido correctamente definido y compilado, es importante asegurarse de que el entrenamiento se realice de manera eficiente. A continuación, se detallan las estrategias prácticas a considerar durante el entrenamiento del modelo.

##### **Monitorización del progreso del entrenamiento**  
En cada época hay que revisar:
- **Pérdida de entrenamiento (`loss`)**: Debe disminuir progresivamente.  
- **Pérdida de validación (`val_loss`)**: No debe disminuir demasiado rápido ni aumentar antes que la pérdida de entrenamiento (signo de sobreajuste).  
- **Precisión (`accuracy`)**: Debe mejorar gradualmente, pero sin saltos abruptos.  
- **Diferencia entre `loss` y `val_loss`**:  
  - Si `val_loss` comienza a subir mientras `loss` sigue bajando → posible sobreajuste.  
  - Si `loss` y `val_loss` no bajan → posible tasa de aprendizaje incorrecta.  

Un ejemplo de salida esperada durante el entrenamiento podría ser el siguiente
```
Epoch 5/20
1000/1000 [==============================] - 5s 5ms/step - loss: 0.45 - accuracy: 0.85 - val_loss: 0.50 - val_accuracy: 0.82
```
- Si **`val_loss` es menor que `loss`**, es probable que el modelo esté generalizando bien.  
- Si **`val_loss` sube mientras `loss` sigue bajando**, considera **Early Stopping** o **Regularización**.  

##### **Estrategias para mejorar el entrenamiento**

###### Para mejorar convergencia y estabilidad  

- **Si la pérdida no mejora o es inestable**:  
   - Reducir la **tasa de aprendizaje** (`learning_rate`).
   - Usar **Batch Normalization** para estabilizar la activación en cada capa.
   - Asegurar que el dataset esté correctamente preprocesado (normalización, codificación correcta de etiquetas).

- **Si el modelo tarda demasiado en entrenar**:  
   - Aumentar el tamaño del **batch** (`batch_size`).
   - Usar **Transfer Learning** si es un problema de imágenes.
   - Aplicar **cuantización** o **pruning** si el modelo es muy grande.

- **Si el modelo muestra signos de sobreajuste**:  
   - Aplicar **Dropout** en capas densas.
   - Usar **Data Augmentation** en imágenes.
   - Aplicar **L2 Regularization** en los pesos de las capas densas.

##### **Uso de `callbacks` para mejorar el proceso de entrenamiento**
Los **callbacks** permiten mejorar el control sobre el entrenamiento. Los más usados son:

- **Early Stopping** → Detiene el entrenamiento si la métrica de validación deja de mejorar:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,  # Número de épocas sin mejora antes de detener
    restore_best_weights=True
)
```

**Reduce Learning Rate on Plateau** → Reduce la tasa de aprendizaje si la validación deja de mejorar:

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # Reduce LR a la mitad
    patience=3,
    min_lr=1e-6
)
```

**Guardado de pesos del mejor modelo (`ModelCheckpoint`)**  → Guarda el modelo con la mejor métrica de validación:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "mejor_modelo.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
```

Podemos integrar los callbacks anteriores en un ejemplo de entrenamiento:
```python
modelo.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint]
)
```

##### **Resumen de mejores prácticas en la fase de entrenamiento** 

Puedes considerar la siguiente tabla resumen

| **Problema**                   | **Posible solución**                                         |
| ------------------------------ | ------------------------------------------------------------ |
| **Pérdida no mejora**          | Reducir `learning_rate`, probar `BatchNormalization`         |
| **Pérdida de validación sube** | Aplicar regularización (`L2`, Dropout), usar `EarlyStopping` |
| **Entrenamiento lento**        | Aumentar `batch_size`, usar una GPU o TPU                    |
| **Modelo sobreajustado**       | Aumentar datos (Data Augmentation), añadir Dropout           |
| **Gradientes inestables**      | Usar `BatchNormalization`, probar `RMSprop` en vez de Adam   |
| **Dataset desbalanceado**      | Ajustar pesos de clases (`class_weight`) o aumentar datos    |

### **Evaluación y métricas en Deep Learning**  

La evaluación de modelos en deep learning depende del tipo de tarea: clasificación, regresión o series temporales. La selección adecuada de métricas permite entender el rendimiento real del modelo y compararlo con otras alternativas.  

#### **Métricas para clasificación**  

En problemas de **clasificación**, el objetivo es evaluar qué tan bien el modelo asigna etiquetas a cada muestra. Existen diferentes métricas según el tipo de clasificación:  

##### **Precisión global (`accuracy`)**  
La **exactitud** mide la proporción de predicciones correctas sobre el total de muestras. Se usa en **problemas balanceados**, pero puede ser engañosa en datasets desbalanceados.  

En **Keras**, se usa así:  
```python
from tensorflow.keras.metrics import Accuracy

modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

##### **Matriz de confusión y métricas derivadas**  
La **matriz de confusión** desglosa los aciertos y errores según clases. A partir de ella, se obtienen métricas más detalladas:  

- **Precisión (`Precision`)**: ¿Cuántos de los positivos predichos eran realmente positivos?  
- **Recall (`Sensibilidad`)**: ¿Cuántos de los positivos reales fueron detectados?  
- **F1-score**: Promedio armónico entre precisión y recall, útil en datasets desbalanceados.  

En **Python**, se calcula con `sklearn`:  
```python
from sklearn.metrics import classification_report

y_pred = modelo.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Convertir a clases si es one-hot
print(classification_report(y_test, y_pred_classes))
```

##### **Cuándo usar cada métrica en clasificación**:  

| **Escenario**                                  | **Métrica recomendada**  |
| ---------------------------------------------- | ------------------------ |
| Dataset balanceado                             | Accuracy                 |
| Dataset desbalanceado                          | F1-score                 |
| Clasificación binaria (pocos falsos positivos) | Precision                |
| Clasificación binaria (pocos falsos negativos) | Recall                   |
| Multiclase                                     | Matriz de confusión + F1 |

#### **Métricas para regresión**  

En **regresión**, el objetivo es minimizar el error entre los valores predichos y los reales. Se utilizan métricas basadas en diferencias entre valores numéricos.  

##### **Error cuadrático medio (`MSE`)**  
Mide el error promedio **elevado al cuadrado**, penalizando más los errores grandes. 

Implementación en Keras:**

```python
modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

##### **Error absoluto medio (`MAE`)**  
Promedio de las diferencias absolutas entre valores reales y predichos. Es menos sensible a valores extremos que el MSE.

##### **Coeficiente de determinación (`R²`)**  
Mide qué porcentaje de la variabilidad de los datos es explicado por el modelo. Un valor cercano a **1** indica un ajuste perfecto.  

**Implementación en `sklearn`:**  

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")
```

**Cuándo usar cada métrica en regresión**:  

| **Escenario**                        | **Métrica recomendada** |
| ------------------------------------ | ----------------------- |
| Penalizar errores grandes            | MSE                     |
| Penalizar errores pequeños por igual | MAE                     |
| Comparar con un modelo base          | R²                      |

#### **Evaluación en problemas de series temporales**  

Las series temporales tienen características específicas, por lo que su evaluación requiere métricas que tengan en cuenta la dependencia temporal.

##### **Errores estándar (MSE, MAE)**  
Se usan igual que en regresión, pero considerando el **orden temporal** de los datos.

##### **Error absoluto porcentual medio (`MAPE`)**  
Evalúa el error en **términos relativos**, útil cuando las escalas de valores varían.

Ejemplo de código en `sklearn`:

```python
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape}%")
```

##### **Errores en forecasting (séries temporales futuras)**  
Cuando el modelo predice valores futuros, es importante analizar el **comportamiento de los errores a lo largo del tiempo**. Estrategias comunes incluyen:

- **Rolling window evaluation:** Evaluar el error en distintos períodos.  
- **Comparación con modelos base:** Evaluar si el modelo supera a un modelo naive (`y(t) = y(t-1)`).  

**Cuándo usar cada métrica en series temporales**:  

| **Escenario**                               | **Métrica recomendada**   |
| ------------------------------------------- | ------------------------- |
| Datos con valores extremos                  | MAE                       |
| Comparación de errores en distintas escalas | MAPE                      |
| Predicción a largo plazo                    | Rolling window evaluation |

#### **Diferencias entre usar métricas dentro o fuera del entrenamiento**

La elección de si calcular la métrica **dentro del entrenamiento de la red** (mediante `model.compile()`) o **fuera del entrenamiento** (mediante `sklearn` u otros métodos) tiene implicaciones importantes en cómo se analiza el rendimiento del modelo.

##### **Uso de métricas dentro del entrenamiento (`model.compile()`)**

Cuando se define una métrica dentro de `model.compile()`, Keras la calcula en **cada batch y en cada época** durante el entrenamiento. Esto permite monitorear la evolución del modelo en tiempo real.

Las ventajas de este método son:

- Se obtiene un registro continuo de la métrica durante el entrenamiento.
- Permite visualizar curvas de aprendizaje (`loss`, `accuracy`, etc.).
- Se puede usar `EarlyStopping` y `ReduceLROnPlateau` basados en la métrica de validación.
- Computación optimizada en GPU/TPU.

Pero también acarrea algunos problemas

- Algunas métricas como `F1-score` o `R²` no están implementadas de forma nativa en `model.compile()`.
- En datasets desbalanceados, `accuracy` puede ser engañosa (se necesita un análisis más detallado).
- Solo se refleja la métrica en el **dataset de entrenamiento y validación**, sin considerar generalización en test.

##### **Uso de métricas fuera del entrenamiento (post-entrenamiento con `sklearn`)**

Calcular métricas después del entrenamiento permite analizar **en detalle** el rendimiento del modelo, especialmente cuando se requiere una evaluación más específica en **datos de test o en problemas desbalanceados**. Al igual que el caso anterior este escenario presenta ventajas y desventajas:

Ventajas:

- Se pueden calcular métricas más avanzadas como `F1-score`, `AUC`, `R²`, etc.
- Permite evaluar en un **dataset de test independiente**, reflejando mejor la capacidad de generalización.
- Se pueden construir **matrices de confusión** y analizar errores específicos.
- Útil para ajustar umbrales de decisión en modelos con salida `sigmoid`.

Limitaciones:

- No influye en el entrenamiento, por lo que no permite usar `EarlyStopping` o `ReduceLROnPlateau` directamente con estas métricas.
- Requiere calcular predicciones en test manualmente (`y_pred = model.predict(X_test)`).
- Puede ser más costoso computacionalmente si el dataset es grande..

###### **Comparación práctica según escenario**

| **Escenario**                      | **Métrica dentro (`compile`)**                   | **Métrica fuera (`sklearn`)**        |
| ---------------------------------- | ------------------------------------------------ | ------------------------------------ |
| Monitorización del entrenamiento   | ✔ Se actualiza en cada época                     | ✘ No, se calcula después             |
| Uso de `EarlyStopping`             | ✔ Sí, permite detener cuando no mejora           | ✘ No aplica directamente             |
| Cálculo de `F1-score` o `R²`       | ✘ No disponible en `compile`                     | ✔ Se puede calcular manualmente      |
| Análisis en dataset de test        | ✘ No se evalúa directamente                      | ✔ Se puede evaluar en test           |
| Corrección de umbrales (`sigmoid`) | ✘ No permite ajuste manual                       | ✔ Se puede ajustar según necesidades |
| Detección de errores específicos   | ✘ Difícil identificar falsos positivos/negativos | ✔ Posible con matriz de confusión    |

Sin duda la mejor estrategia es combinar ambos enfoques:

- Monitorizar `accuracy` y `loss` dentro del entrenamiento para ajustar el modelo en tiempo real.
- Calcular métricas avanzadas en test después del entrenamiento para evaluar la generalización.
