# Tema 5. Aprendizaje profundo y redes neuronales

## Fundamentos del aprendizaje profundo

### Objetivos del módulo

> - Comprender la diferencia entre **Machine Learning tradicional y Deep Learning**.
> - Explicar la arquitectura básica de una **red neuronal artificial (MLP)**.
> - Analizar el proceso de **propagación hacia adelante y retropropagación del error**.
> - Implementar una **red neuronal multicapa en TensorFlow y PyTorch**.
> - Entrenar y evaluar la red neuronal en un **dataset básico (MNIST o Fashion-MNIST)**.
> - Visualizar pesos, activaciones y comportamiento del modelo.

---

### Introducción

#### La evolución desde el Machine Learning hacia Deep Learning

Desde sus inicios, la inteligencia artificial ha buscado desarrollar sistemas capaces de aprender a partir de los datos. Dentro de este campo, el **Machine Learning (ML)** ha permitido la construcción de modelos capaces de hacer predicciones sin necesidad de definir reglas explícitas. Sin embargo, conforme el volumen de datos y la complejidad de los problemas han aumentado, las limitaciones de los algoritmos tradicionales han impulsado el desarrollo del **Deep Learning (DL)**, un enfoque basado en redes neuronales profundas y que ha transformado la manera en que se diseñan los sistemas inteligentes.

##### **De los modelos clásicos a las redes profundas**

El aprendizaje automático comenzó con métodos de naturaleza estadística que permitían modelar relaciones entre variables y hacer inferencias tomando datos desconocidos como entradas. La regresión lineal y logística, los árboles de decisión o las máquinas de soporte vectorial (SVM) fueron algunos de los enfoques más utilizados en las primeras décadas de desarrollo del machine learning. En paralelo, las redes neuronales artificiales comenzaron a tomar relevancia a partir de la década de los 60 del siglo XX, con modelos como el **Perceptrón**, un sistema capaz de clasificar datos linealmente separables mediante combinaciones ponderadas de entradas y una función de activación.

A pesar de su efectividad en tareas específicas, estos métodos tradicionales presentaban desafíos significativos cuando se enfrentaban, por ejemplo, a datos no estructurados, como imágenes o texto. Uno de los principales problemas ha sido siempre la necesidad de realizar una **ingeniería de características manual**, un proceso en el que los expertos **diseñaban representaciones de los datos adecuadas** para alimentar a los algoritmos. Este enfoque resultaba costoso y limitaba la escalabilidad de los modelos.

El Deep Learning surge como una solución a esta problemática al eliminar la dependencia de la ingeniería de características manual. A través del uso de **redes neuronales profundas**, es posible extraer representaciones jerárquicas de los datos de manera automática, permitiendo que el modelo aprenda patrones complejos directamente desde la información de entrada. Este avance ha sido clave para abordar problemas que antes eran intratables con los enfoques tradicionales.

> [!warning]
>
> **¿Qué significa ingeniería manual de características?**
>
> En todo proceso de aprendizaje automático conviene separar **dos etapas** que muchas veces se confunden
>
> 1. **Extracción (o ingeniería) de características**: Cómo convertir los datos brutos (texto, imágenes, audio, etc.) en **representaciones numéricas** que un algoritmo pueda procesar.
> 2. **Aprendizaje (modelado)**: Cómo, a partir de esas representaciones, se **ajusta un modelo** para hacer predicciones o clasificaciones.
>
> En el *Machine Learning* “clásico”, se acostumbra a **hacer ingeniería de características a mano**. Por ejemplo, en visión artificial, se aplican algoritmos como HOG (Histogram of Oriented Gradients) o SIFT para extraer descriptores de bordes, contornos, etc. En NLP, se utilizan métodos como bolsa de palabras (BoW) o TF-IDF. Una vez que se tienen esos vectores de características, se entrena un algoritmo tradicional (SVM, Random Forest, regresión logística, etc.). Dicho de otro modo, **en el ML clásico, el modelo “aprende” a partir de esas características** que **tú has decidido** cómo extraer.
>
> El *Machine Learning* clásico “aprende” de los datos, pero **no descubre por sí solo** cómo representar la información en bruto: depende de la selección (ingeniería) de características **hecha por personas** (o por pasos heurísticos).
>
> **¿Qué pasa con el Deep Learning?**
>
> En el aprendizaje profundo, especialmente con redes neuronales convolucionales (CNN) en visión artificial o modelos de lenguaje (Transformers, LSTM) en NLP, la red se entrena de extremo a extremo: Primero recibe como **entrada el dato bruto** (imagen, texto, audio), seguidamente aprende **automáticamente** qué transformaciones y filtros aplicar para extraer patrones relevantes. Finalmente ajusta las salidas para la tarea de predicción o clasificación.
>
> Esto implica que **el modelo se encarga de la “ingeniería de características”** (los filtros que detectan contornos en una CNN, por ejemplo) y **del “aprendizaje final”** en una sola pasada. No hace falta un experto que defina manualmente cuáles propiedades del dato deben medirse.
>
> **Por tanto...**
>
> Con **Machine Learning clásico**, la parte de “traducir” el problema al lenguaje del algoritmo (ingeniería de características) corre a cargo de la persona experta, y, **posteriormente** el algoritmo “aprende” con esos datos transformados.
>
> Con **Deep Learning**, gran parte de ese trabajo de “traducción” se automatiza. La red neuronal **descubre** cómo procesar la imagen, texto o audio, y **también** aprende a clasificarlo o predecir sobre él.
>
> En consecuencia, **ambos procesos “aprenden”**, pero solo el **Deep Learning** se encarga además de la **extracción automática de rasgos** directamente desde la entrada cruda. Esto es lo que marca la gran diferencia y el gran avance de los últimos años.

##### Factores que impulsaron la adopción del Deep Learning

El auge del Deep Learning no ha sido fortuito, sino que ha estado respaldado por una serie de avances en diversas áreas. En primer lugar, el crecimiento exponencial en la disponibilidad de datos ha permitido entrenar modelos de gran escala sin que estos caigan en sobreajuste. La digitalización masiva y la proliferación de sensores, redes sociales y sistemas de registro han generado volúmenes de datos sin precedentes, proporcionando el insumo necesario para que las redes neuronales profundas logren generalizar patrones complejos.

Otro factor determinante ha sido el incremento en la capacidad computacional. La evolución de las **unidades de procesamiento gráfico (GPUs)** y, más recientemente, de las **unidades de procesamiento tensorial (TPUs)** ha permitido realizar operaciones matriciales a gran escala de manera eficiente. Sin este avance, el entrenamiento de redes neuronales profundas, que requiere una cantidad considerable de cálculos, habría sido impracticable en entornos reales.

Además de los avances en hardware, las mejoras en los algoritmos de entrenamiento han sido esenciales para la consolidación del Deep Learning. Métodos como la optimización mediante **descenso de gradiente estocástico con momentum**, el uso de **normalización por lotes (batch normalization)** y la introducción de técnicas de regularización como **dropout** han hecho que las redes neuronales sean más estables y eficientes en su proceso de aprendizaje.

Finalmente, la accesibilidad a **frameworks de Deep Learning** como **TensorFlow, PyTorch y Keras** ha facilitado la experimentación y la adopción de estas tecnologías. Gracias a estas herramientas, investigadores e ingenieros pueden construir modelos avanzados sin necesidad de programar desde cero las operaciones matemáticas subyacentes.

##### **El impacto del Deep Learning en la inteligencia artificial**

La llegada del Deep Learning ha transformado múltiples campos dentro de la inteligencia artificial. En visión por computadora, ha permitido desarrollar sistemas de reconocimiento facial, diagnóstico médico basado en imágenes y detección de objetos con niveles de precisión superiores a los obtenidos por humanos. En el ámbito del procesamiento de lenguaje natural, modelos como **BERT, GPT y T5** han revolucionado la traducción automática, la generación de texto y la comprensión del lenguaje.

El aprendizaje por refuerzo profundo, por su parte, ha llevado a la creación de sistemas capaces de superar a jugadores humanos en juegos complejos, como en el caso de **AlphaGo**, desarrollado por DeepMind. En robótica y automatización, los modelos basados en Deep Learning han demostrado ser capaces de controlar sistemas físicos con un alto grado de autonomía.

A pesar de su éxito, el Deep Learning no ha desplazado completamente al Machine Learning tradicional. En muchas aplicaciones donde los datos son limitados o la interpretabilidad del modelo es un factor clave, los métodos clásicos siguen siendo una alternativa válida y eficiente. Sin embargo, en tareas donde la complejidad y el volumen de datos son elevados, el Deep Learning ha demostrado ser el enfoque dominante, marcando un antes y un después en la evolución de la inteligencia artificial.

##### **Para reflexionar...**

> **¿Por qué el Deep Learning ha superado al Machine Learning tradicional en muchas aplicaciones?**
>  **Clave**: Reflexiona sobre cómo la automatización de la extracción de características, el aumento en la capacidad computacional y el crecimiento de los datos han favorecido la adopción del DL en la industria.

#### **Diferencias clave con algoritmos tradicionales de aprendizaje supervisado**

Como hemos comentado, el **aprendizaje supervisado** se basa en la construcción de modelos que aprenden a partir de datos etiquetados, estableciendo relaciones entre las entradas y las salidas esperadas. Los algoritmos tradicionales dentro de este paradigma, como la **regresión lineal**, las **máquinas de soporte vectorial (SVM)** o los **árboles de decisión**, han sido ampliamente utilizados debido a su interpretabilidad y eficiencia computacional. Sin embargo, conforme los problemas se han vuelto más complejos, las limitaciones de estos enfoques han evidenciado la necesidad de modelos más expresivos, como los que ofrece el **Deep Learning**.

##### **Capacidad de representación y extracción de características**

Una diferencia fundamental entre los métodos tradicionales y el Deep Learning radica en la forma en que representan y procesan los datos. En modelos clásicos, como las SVM o la regresión logística, la efectividad del aprendizaje depende en gran medida de la selección de **características relevantes**, un proceso que requiere conocimiento experto del dominio del problema. Esta necesidad de definir manualmente los atributos más representativos impone una barrera significativa cuando se trabaja con datos altamente dimensionales o no estructurados, como imágenes o texto.

En contraste, los modelos de Deep Learning automatizan este proceso mediante **redes neuronales profundas**, capaces de extraer representaciones jerárquicas de los datos. Cada capa de la red aprende a transformar la información de entrada en niveles progresivamente más abstractos, permitiendo que las características de alto nivel emerjan sin intervención humana. Esta propiedad ha sido clave para el éxito de modelos como las **redes convolucionales (CNN)** en visión por computadora o los **Transformers** en procesamiento de lenguaje natural.

##### **Escalabilidad y aprendizaje en grandes volúmenes de datos**

Otra diferencia esencial está en la **capacidad de escalabilidad**. Los algoritmos tradicionales pueden ser efectivos en conjuntos de datos relativamente pequeños, donde las relaciones entre las variables pueden modelarse de manera eficiente con funciones matemáticas bien definidas. Sin embargo, cuando se trabaja con volúmenes masivos de información, estos enfoques tienden a encontrar límites en su capacidad de generalización.

El Deep Learning, por su parte, ha demostrado una gran capacidad para **aprovechar grandes cantidades de datos**. Modelos como **GPT-4, BERT o ResNet** han sido entrenados con millones o incluso miles de millones de ejemplos, permitiendo que capturen patrones extremadamente complejos que serían imposibles de modelar con métodos convencionales. No obstante, esta ventaja también trae consigo un costo computacional elevado, lo que hace que la implementación práctica de redes neuronales profundas dependa en gran medida del acceso a hardware especializado, como **GPUs y TPUs**.

##### **Estrategias de optimización y entrenamiento**

El proceso de entrenamiento en los modelos tradicionales de aprendizaje supervisado suele basarse en la **minimización de una función de pérdida** mediante métodos de optimización convexa, como por ejemplo el **descenso de gradiente** en la regresión logística o la maximización del margen en las SVM. Dado que estos métodos suelen operar en espacios de optimización relativamente simples, la convergencia a una solución óptima suele ser rápida y estable.

En contraste, el entrenamiento de redes neuronales profundas involucra una optimización mucho más compleja debido a la **gran cantidad de parámetros** y a la **no convexidad de la función de pérdida**. Esto introduce desafíos adicionales, como la aparición de **mínimos locales**, el **desvanecimiento o explosión del gradiente** o la necesidad de técnicas avanzadas para mejorar la convergencia. Métodos como el **descenso de gradiente estocástico (SGD) con momentum**, la **normalización por lotes (batch normalization)** y la **regularización mediante dropout** han sido desarrollados específicamente para abordar estos problemas y hacer que el entrenamiento sea más eficiente.

##### **Interpretabilidad y uso en la industria**

Un aspecto en el que los métodos tradicionales aún tienen ventajas sobre el Deep Learning es la **interpretabilidad**. Modelos como los árboles de decisión o la regresión logística permiten analizar directamente la contribución de cada variable en la predicción final, lo que facilita su uso en sectores como la medicina, las finanzas o el derecho, donde la explicabilidad es un requisito crítico.

Las redes neuronales profundas, en cambio, suelen ser percibidas como **cajas negras**, ya que el conocimiento aprendido se distribuye en miles o millones de parámetros sin una correspondencia explícita con variables individuales. Aunque han surgido técnicas como **SHAP, LIME o Grad-CAM** para hacer que estos modelos sean más interpretables, el desafío sigue siendo una barrera en su adopción en algunos entornos.

A pesar de esta diferencia, en muchas aplicaciones el **rendimiento superior del Deep Learning justifica su uso**, incluso si su interpretabilidad es menor. En problemas como la clasificación de imágenes, la traducción automática o el reconocimiento de voz, los modelos tradicionales han sido ampliamente superados, lo que ha llevado a una adopción masiva de redes neuronales profundas en la industria.

> [!tip]
>
> El Deep Learning representa una evolución del aprendizaje supervisado que ha permitido abordar problemas de alta complejidad con una eficacia sin precedentes. Su capacidad para **aprender representaciones jerárquicas**, **escalar con grandes volúmenes de datos** y **resolver tareas antes intratables** lo ha convertido en la opción preferida en numerosos campos. Sin embargo, su alto costo computacional y su menor interpretabilidad en comparación con los métodos tradicionales siguen siendo factores a considerar en su aplicación práctica.



###### **Para reflexionar...**

> **¿En qué situaciones los modelos tradicionales de aprendizaje supervisado siguen siendo preferibles al Deep Learning?**
>  **Clave**: Analiza casos donde la interpretabilidad, el bajo costo computacional o la escasez de datos pueden hacer que los métodos clásicos sean una mejor opción.

#### **Aplicaciones prácticas en visión por computadora, NLP y otros campos**

El Deep Learning ha trascendido la teoría para convertirse en una herramienta fundamental en múltiples disciplinas. Su capacidad para **aprender representaciones complejas a partir de los datos** ha permitido resolver problemas que antes eran intratables con los métodos tradicionales de Machine Learning. Dentro de sus aplicaciones más destacadas se encuentran la **visión por computadora**, el **procesamiento de lenguaje natural (NLP)** y otros campos donde la interpretación de datos no estructurados es fundamental.

##### **Visión por computadora: entendiendo el mundo visual**

Uno de los campos donde el Deep Learning ha tenido mayor impacto es la **visión por computadora**, una disciplina que busca dotar a las máquinas de la capacidad de interpretar imágenes y videos. Tradicionalmente, el análisis visual dependía de técnicas manuales de procesamiento de imágenes, donde se diseñaban filtros y algoritmos específicos para detectar bordes, formas o texturas. Este enfoque resultaba limitado, ya que requería un ajuste manual considerable y tenía dificultades para generalizar ante variaciones en los datos.

Con la llegada de las **redes neuronales convolucionales (CNN)**, el paradigma cambió completamente. Estas redes son capaces de aprender de forma autónoma representaciones jerárquicas de las imágenes, capturando desde patrones básicos hasta estructuras de alto nivel. Gracias a esto, modelos basados en CNN han logrado superar el desempeño humano en tareas como el reconocimiento de objetos o la clasificación de imágenes.

En la actualidad, el Deep Learning es el motor detrás de aplicaciones como la **detección de rostros en fotografías**, el **diagnóstico médico basado en imágenes** y la **conducción autónoma**, donde los vehículos identifican obstáculos, señales de tráfico y peatones en tiempo real. Modelos como **ResNet, EfficientNet y YOLO (You Only Look Once)** han sido fundamentales en estos avances, proporcionando soluciones eficientes y escalables para el procesamiento visual.

##### **Procesamiento de lenguaje natural: entendiendo el significado del texto**

El lenguaje humano es altamente complejo, con estructuras gramaticales, semántica implícita y múltiples niveles de interpretación. Los métodos tradicionales de procesamiento de lenguaje natural (NLP) dependían de reglas manuales o representaciones simplificadas del texto, lo que limitaba su capacidad de capturar la riqueza del lenguaje.

El Deep Learning ha revolucionado el NLP con el uso de **modelos basados en redes neuronales recurrentes (RNN), LSTM y Transformers**. Estos modelos han permitido avances en tareas como la **traducción automática, el análisis de sentimientos y la generación de texto**. En particular, los **Transformers**, introducidos a partir de la publicación del trabajo de investigación ***Attention is All You Need***, han cambiado por completo el campo al permitir que las redes procesen oraciones completas de manera paralela, mejorando la eficiencia y la precisión.

Ejemplos como **BERT (Bidirectional Encoder Representations from Transformers), GPT (Generative Pre-trained Transformer) y T5 (Text-to-Text Transfer Transformer)** han llevado a aplicaciones prácticas como los asistentes virtuales, los chatbots avanzados y los sistemas de búsqueda semántica. Hoy en día, modelos como **ChatGPT o Google Bard** pueden generar textos altamente coherentes, responder preguntas complejas y mantener conversaciones naturales con los usuarios.

##### **Más allá de imágenes y texto: otras aplicaciones clave**

Si bien la visión por computadora y el NLP han sido dos de los campos más beneficiados por el Deep Learning, su impacto se extiende a muchas otras áreas. En biomedicina, por ejemplo, las redes neuronales profundas han permitido el descubrimiento de nuevos fármacos mediante el análisis de estructuras moleculares y la simulación de interacciones químicas. En el ámbito financiero, se utilizan modelos de Deep Learning para la detección de fraudes, el análisis de riesgos y la predicción de tendencias en los mercados.

Otro campo emergente es el **aprendizaje por refuerzo profundo**, donde agentes inteligentes son entrenados para optimizar estrategias en entornos dinámicos. Aplicaciones en robótica, automatización industrial y videojuegos han demostrado la capacidad de estos modelos para aprender tareas complejas sin intervención humana directa.

Además, en climatología y modelado ambiental, los algoritmos de Deep Learning se han utilizado para predecir patrones meteorológicos, modelar el impacto del cambio climático y optimizar la generación de energía renovable. En astronomía, han sido clave para el análisis de imágenes de telescopios y la detección de exoplanetas a partir de datos de observación.

> [!tip]
>
> El Deep Learning ha redefinido la forma en que las máquinas procesan y comprenden información. Desde la **visión por computadora**, donde ha permitido a los sistemas interpretar el mundo visual, hasta el **procesamiento de lenguaje natural**, que ha hecho posible la interacción fluida entre humanos y máquinas, su impacto ha sido profundo. Además, su aplicación en áreas como la biomedicina, las finanzas y la robótica continúa expandiendo los límites de la inteligencia artificial.
>
> A medida que los modelos siguen evolucionando, es probable que veamos aún más aplicaciones que aprovechen la capacidad de las redes neuronales profundas para **extraer patrones complejos, generalizar a nuevos escenarios y resolver problemas antes inalcanzables**.

###### **Para reflexionar...**

> **¿Qué factores han permitido que el Deep Learning se aplique con éxito en tantos campos diferentes?**
>  **Clave**: Considera el papel de la capacidad de representación automática, la escalabilidad de los modelos y la disponibilidad de datos en la expansión del Deep Learning a múltiples disciplinas.

### Arquitectura de redes neuronales

#### **Concepto de neurona artificial y perceptrón**

Desde un punto de vista biológico, el **cerebro humano** es una red altamente interconectada de neuronas que procesan información de manera distribuida. Cada neurona recibe señales eléctricas a través de sus **dendritas**, procesa esta información y, si la señal es suficientemente fuerte, genera una respuesta a través del **axón**, transmitiéndola a otras neuronas. Este modelo ha servido como inspiración para el desarrollo de las **neuronas artificiales**, unidades fundamentales en las redes neuronales artificiales.  

El concepto de **neurona artificial** surge como una simplificación matemática de las neuronas biológicas. Formalmente, una neurona artificial recibe una serie de entradas, cada una con un peso asociado, y realiza una operación matemática para calcular su salida. Este cálculo puede representarse como: 

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

donde:
- $x_i$ representa las entradas del modelo.
- $w_i$ son los pesos que determinan la importancia de cada entrada.
- $b$ es un término de sesgo o bias, que ajusta el umbral de activación.
- $z$ es la combinación ponderada de las entradas antes de pasar por una función de activación. 

Hasta aquí la neurona realiza una transformación **lineal** de los datos de entrada, aplicando unos pesos $w_i$ y un sesgo $b$

Sin embargo, para decidir si la neurona se activa o no, se aplica una **función de activación** sobre $z$, que introduce **no linealidad** y permite que el modelo aprenda relaciones más complejas en los datos. En su forma más simple, esta función puede ser un **escalón**, activando la neurona solo si la suma ponderada de las entradas supera un umbral.

##### **Modelo del Perceptrón: función de activación escalón y regla de actualización de pesos**

El **Perceptrón**, introducido por **Frank Rosenblatt en 1958**, fue el primer modelo de neurona artificial funcional. Su funcionamiento se basa en una regla de decisión binaria que clasifica las entradas en dos categorías. Matemáticamente, su salida se define como:

$$
y = f(z) =
\begin{cases} 
1, & \text{si } z \geq 0 \\ 
0, & \text{si } z < 0
\end{cases}
$$

Esta función de activación **escalón** es la más simple y permite modelar problemas de clasificación binaria. El entrenamiento del Perceptrón consiste en **ajustar los pesos** de forma iterativa para minimizar los errores de clasificación. La regla de actualización de pesos se basa en la siguiente ecuación:

$$
w_i \leftarrow w_i + \eta (y_{\text{real}} - y_{\text{predicho}}) x_i
$$

donde:
- $\eta$ es la tasa de aprendizaje, que controla la magnitud del ajuste.
- $y_{\text{real}}$ es la etiqueta real de la muestra.
- $y_{\text{predicho}}$ es la salida calculada por el modelo.
- $x_i$ es la entrada asociada al peso $w_i$.  

El Perceptrón ajusta los pesos tras cada observación, de modo que si una muestra se clasifica incorrectamente, los pesos se actualizan en la dirección que reduciría el error en futuras iteraciones.

> **Ejemplo:**
>
> Veamos el ejemplo de un perceptrón que aprende la función **OR** en el que se muestra explícitamente el **cálculo manual de los pesos** en cada iteración. Queremos que el perceptrón aprenda la mencionada función lógica **OR** a partir de los siguientes datos de entrenamiento:
>
> | $x_1$ | $x_2$ | OR($x_1, x_2$) |
> | ----- | ----- | -------------- |
> | 0     | 0     | 0              |
> | 0     | 1     | 1              |
> | 1     | 0     | 1              |
> | 1     | 1     | 1              |
>
> El **modelo del perceptrón** tiene una función de activación:
>
> $$
> y = \text{step}(w_1 x_1 + w_2 x_2 + b)
> $$
> 
> Donde la **función escalón (step function)** se define como:
>
> $$
> \text{step}(z) =
> \begin{cases} 
> 1, & \text{si } z \geq 0 \\
> 0, & \text{si } z < 0
> \end{cases}
> $$
> 
> Según la propuesta inicial del perceptrón de Rosenblatt, el **algoritmo de aprendizaje** sigue la regla de actualización:
>
> $$
> w_j \leftarrow w_j + \eta \cdot (y_{\text{real}} - y_{\text{pred}}) \cdot x_j
> $$
>
> $$
> b \leftarrow b + \eta \cdot (y_{\text{real}} - y_{\text{pred}})
> $$
> 
> Donde:
> - $w_j$ son los pesos de la neurona.
> - $b$ es el sesgo.
> - $\eta$ es la tasa de aprendizaje (usaremos $\eta = 0.1$).
> - $y_{\text{real}}$ es la etiqueta real.
> - $y_{\text{pred}}$ es la predicción del perceptrón.
>
> **Iteración 0: Datos de incio**
>
> - Supongamos que los pesos y el sesgo inician en $w_1 = 0$, $w_2 = 0$, $b = 0$.
> - $\eta = 0.1$
>
> **Iteración 1 (con los datos de entrenamiento uno por uno):**
>
> - Entrada: $(x_1 = 0, x_2 = 0)$
> - Salida esperada: $y = 0$
> - Cálculo de salida:  
> 
> $$
>  z = (0 \cdot 0) + (0 \cdot 0) + 0 = 0
> $$
>  
> $$
>  \text{step}(0) = 1
> $$
>  
> - Error: $y_{\text{real}} - y_{\text{pred}} = 0 - 1 = -1$
> - Actualización:
> 
> $$
> w_1 = 0 + (0.1 \times -1 \times 0) = 0
> $$
>   
> $$
> w_2 = 0 + (0.1 \times -1 \times 0) = 0
> $$
>   
> $$
> b = 0 + (0.1 \times -1) = -0.1
> $$
>
> El estado de los pesos será entonces:
> 
> $$
> w_1 = 0; w_2 = 0; b = -0.1
> $$
> 
> Pasamos ahora la siguiente observación:
>
> - Entrada: $(x_1 = 0, x_2 = 1)$
> - Salida esperada: $y = 1$
> - Cálculo de salida: 
> 
> $$
> z = (0 \cdot 0) + (0 \cdot 1) + (-0.1) = -0.1
> $$
>   
> $$
> \text{step}(-0.1) = 0
> $$
>   
> - Error: $1 - 0 = 1$
> - Actualización:
> 
> $$
> w_1 = 0 + (0.1 \times 1 \times 0) = 0
> $$
>   
> $$
> w_2 = 0 + (0.1 \times 1 \times 1) = 0.1
> $$
>   
> $$
> b = -0.1 + (0.1 \times 1) = 0
> $$
>
> Ahora los pesos serían:
> 
> $$
> w_1 = 0; w_2 = 0.1; b = 0
> $$
> 
> Igualmente podemos hacer para las siguientes observaciones, obteniendo los siguientes pesos en ambos casos
> 
> $$
> w_1 = 0; w_2 = 0.1; b = 0
> $$
> 
> Después de una sola **época de entrenamiento**, los pesos han convergido a:
> - **$w_1 = 0$**
> - **$w_2 = 0.1$**
> - **$b = 0$**
>
> La función aprendida es:
>
> $$
> y = \text{step}(0 \cdot x_1 + 0.1 \cdot x_2 + 0)
> $$
>
> Si bien en este caso la solución no es ideal, si entrenamos durante más épocas, los pesos se ajustarán mejor para representar la función OR.
>
> Este ejemplo muestra cómo el perceptrón ajusta sus pesos iterativamente **solo cuando comete errores**, siguiendo una regla de actualización basada en los ejemplos de entrenamiento.  
>

##### **Limitaciones del Perceptrón y el problema de la separación lineal**

A pesar de su éxito inicial, el Perceptrón tiene limitaciones fundamentales. Su principal restricción es que **solo puede resolver problemas linealmente separables**, es decir, aquellos en los que las clases pueden dividirse con una línea recta en dos dimensiones o con un hiperplano en espacios de mayor dimensión.  

Un ejemplo clásico de su limitación es el problema de la función lógica **XOR**, donde no es posible trazar una única línea de decisión que separe correctamente las clases. En 1969, **Marvin Minsky y Seymour Papert** demostraron matemáticamente esta deficiencia, lo que llevó a un estancamiento en la investigación de redes neuronales durante varios años.  

Para superar esta limitación, fue necesario extender el concepto del Perceptrón hacia **redes neuronales multicapa (MLP)**, que incorporan múltiples capas de neuronas y funciones de activación más sofisticadas, permitiendo modelar relaciones no lineales y resolver problemas más complejos.

##### **Para reflexionar...**

> **¿Por qué el Perceptrón no puede resolver problemas como la función XOR?** 
> **Clave**: Reflexiona sobre cómo la separación lineal limita la capacidad de este modelo y por qué la introducción de múltiples capas permite superar esta restricción.  

#### **Evolución a redes neuronales multicapa (MLP)**

El Perceptrón, a pesar de haber sido un hito en la historia de la inteligencia artificial, demostró ser limitado al enfrentarse a problemas donde las clases no podían separarse con una simple línea recta. Esta restricción, conocida como el problema de la **separación lineal**, llevó a la necesidad de arquitecturas más flexibles que permitieran capturar relaciones más complejas en los datos. La solución vino con la introducción de las **redes neuronales multicapa (MLP, Multi-Layer Perceptron)**, modelos que incorporan múltiples niveles de procesamiento para aprender representaciones progresivamente más abstractas.

El concepto clave detrás de una **MLP** es la introducción de **capas ocultas**, es decir, neuronas intermedias que transforman la información antes de llegar a la capa de salida. A diferencia del perceptrón, que toma una decisión inmediata basada en una combinación lineal de las entradas, una red multicapa es capaz de construir **representaciones jerárquicas que permiten resolver problemas más complejos**. Esta capacidad de abstracción es lo que ha permitido que las redes neuronales sean aplicadas con éxito en tareas de predicción de lo más diverso.

##### **Capas de entrada, ocultas y salida**

El funcionamiento de una **MLP** se puede entender como un flujo de información a través de diferentes niveles de procesamiento. Todo comienza con la **capa de entrada**, donde se reciben los datos en su forma original. Su único propósito es transmitir la información sin alteraciones hacia la siguiente capa.

El verdadero procesamiento ocurre en las **capas ocultas**, que son las responsables de transformar los datos. Cada neurona en estas capas recibe la información de las neuronas de la capa anterior, la procesa aplicando una transformación matemática y la pasa a la siguiente capa. Cuantas más capas tenga una red, más complejas pueden ser las representaciones que aprende. Esta capacidad de construir representaciones progresivas es lo que distingue a las redes neuronales profundas de otros modelos de aprendizaje automático.

Finalmente, la **capa de salida** produce el resultado final. Su número de neuronas depende del tipo de tarea a resolver. En un problema de clasificación binaria, por ejemplo, suele haber una sola neurona que indica la probabilidad de pertenecer a una de las dos clases. Para clasificación multiclase, en cambio, la capa de salida tendrá tantas neuronas como categorías posibles, utilizando una función de activación que normaliza los valores en probabilidades.

<img src=".\assets\multicapa.png" alt="Código Flutter" />

##### **El papel de la función de activación**

Ya vimos con el perceptrón que para que una red neuronal pueda modelar relaciones no lineales, es fundamental que las neuronas no se limiten a realizar simples combinaciones lineales de las entradas. Esta transformación se logra mediante las **funciones de activación**, que introducen no linealidad en la red y permiten que aprenda patrones más complejos. Analicemos un poco mas detenidamente el papel de la función de activación

Cuando los primeros modelos de redes neuronales comenzaron a tomar forma, uno de los desafíos fundamentales era encontrar una manera efectiva de transformar las salidas de las neuronas en valores interpretables. La **función sigmoide** fue una de las primeras respuestas a este problema, ya que permitía convertir cualquier entrada en un número entre 0 y 1, lo que facilitaba la interpretación de los resultados como probabilidades. Su suavidad y comportamiento gradual parecían ideales para permitir que las redes aprendieran de manera controlada, evitando cambios bruscos en la salida ante pequeñas modificaciones en los datos de entrada.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

<img src=".\assets\image-20250217230246496.png" alt="image-20250217230246496" />

Sin embargo, a medida que empezó a investigar el funcionamiento de redes más profundas, surgieron problemas inesperados. Durante la fase de entrenamiento, en la que los pesos de la red se ajustan para minimizar el error, la sigmoide comenzó a mostrar una gran limitación: su derivada se vuelve extremadamente pequeña para valores de entrada muy grandes o muy pequeños. Esto significa que, cuando la red intenta aprender, las capas profundas reciben gradientes casi nulos, lo que hace que sus pesos apenas se actualicen. Conocido como el **problema del desvanecimiento del gradiente**, este fenómeno limitaba la capacidad de las redes para capturar patrones complejos en los datos.

Para abordar este problema, se exploraron alternativas y una de las primeras soluciones fue la **función tangente hiperbólica (Tanh)**. A diferencia de la sigmoide, la Tanh también es una función en forma de S, pero tiene la ventaja de que su salida oscila entre -1 y 1, en lugar de entre 0 y 1.

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<img src=".\assets\image-20250217230325803.png" alt="image-20250217230325803" />

Esto significaba que los valores podían estar centrados en torno a cero, lo que ayudaba a mejorar la estabilidad del aprendizaje al proporcionar una salida simétrica. Con esta modificación, los modelos podían aprender de manera más eficiente, ya que los gradientes tendían a ser más grandes que los de la sigmoide en la mayoría de las regiones de la función. Sin embargo, el problema no desapareció por completo. Cuando los valores de entrada a la función eran muy grandes o muy pequeños, la Tanh también sufría el desvanecimiento del gradiente, limitando el aprendizaje en redes profundas.

Con el auge de modelos más complejos y profundos, se hizo evidente que era necesario un cambio más radical en la forma en que las neuronas procesaban la información. Fue entonces cuando se popularizó la **función de activación ReLU (Rectified Linear Unit)**, que marcó un antes y un después en la evolución del deep learning. A diferencia de las funciones anteriores, ReLU no tiene una forma sigmoidal, sino que simplemente devuelve el valor de entrada si es positivo y 0 si es negativo.

$$
\text{ReLU}(x) = \max(0, x)
$$

<img src=".\assets\image-20250217230530677.png" alt="image-20250217230530677" />



Esto le da una propiedad clave: evita el problema del desvanecimiento del gradiente en la mayoría de los casos, ya que su derivada es 1 para valores positivos. Gracias a esto, los modelos pueden aprender con mayor rapidez y eficacia, permitiendo entrenar redes neuronales con muchas más capas sin los obstáculos que habían limitado el progreso del campo durante años.

Lo acontecido con las funciones de activación refleja en cierto modo la evolución del deep learning en su búsqueda de soluciones cada vez más eficientes. La elección de la función de activación en cada capa de la red es un factor clave en el diseño de una MLP. Mientras que en las capas ocultas suelen usarse funciones como ReLU para facilitar el aprendizaje, en la capa de salida la elección depende del tipo de problema. En clasificación binaria, se utiliza la sigmoide para producir probabilidades; en clasificación multiclase, la **softmax**, que asigna una distribución de probabilidades sobre las categorías.

> [!tip]
>
> Las redes neuronales multicapa representan un avance crucial respecto al Perceptrón al introducir múltiples niveles de procesamiento que permiten modelar relaciones no lineales. Su estructura jerárquica, compuesta por **capas de entrada, ocultas y salida**, facilita la transformación progresiva de los datos hasta generar predicciones precisas. Sin embargo, para que estas redes sean realmente efectivas, es fundamental la correcta elección de las **funciones de activación**, las cuales determinan cómo se propaga la información a través de la red. Estos conceptos son la base del **Deep Learning** moderno y constituyen el fundamento de arquitecturas más complejas que veremos más adelante.

##### **Para reflexionar...**

> **¿Cómo influye el número de capas ocultas en la capacidad de una red neuronal para aprender representaciones complejas?**
>  **Clave**: Considera cómo el aumento en la profundidad permite capturar patrones más abstractos, pero también introduce nuevos desafíos, como la propagación del gradiente y el riesgo de sobreajuste.

#### **Propagación de la información en una MLP**

El aprendizaje en una **red neuronal multicapa (MLP)** se basa en el flujo de información a través de sus capas, un proceso conocido como **propagación hacia adelante** o *forward pass*. Este mecanismo permite que los datos atraviesen la red desde la capa de entrada hasta la capa de salida, transformándose en cada nivel de la arquitectura hasta obtener una predicción final. Para comprender este proceso, es esencial analizar cómo se calculan las activaciones en cada neurona y cómo la función de pérdida guía el aprendizaje del modelo.

##### **Propagación hacia adelante (forward pass): cálculo de activaciones**

El **forward pass** es el primer paso en la ejecución de una red neuronal. Su objetivo es calcular la salida de la red a partir de las entradas y los pesos actuales del modelo. Cada neurona de la red realiza dos operaciones fundamentales:

###### **Cálculo de la suma ponderada de las entradas**  

Cada neurona recibe señales desde la capa anterior. Estas señales se combinan en una suma ponderada mediante los pesos sinápticos del modelo:

$$
z^{(l)}_i = \sum{w^{(l)}_{ij} a^{(l-1)}_j + b^{(l)}_i}
$$

donde:
- $z^{(l)}_i$ es la activación neta de la neurona $i$ en la capa $l$.
- $w^{(l)}_{ij}$ representa el peso que conecta la neurona $j$ de la capa $l-1$ con la neurona $i$ de la capa $l$.
- $a^{(l-1)}_j$ es la activación de la neurona $j$ en la capa anterior.
- $b^{(l)}_i$ es el sesgo asociado a la neurona $i$ en la capa $l$.

###### **Aplicación de la función de activación**  

Una vez obtenida la suma ponderada, se pasa a través de una **función de activación** que introduce no linealidad en la red, permitiendo que el modelo capture relaciones complejas. La salida activada de cada neurona se define como:

$$
a^{(l)}_i = f(z^{(l)}_i)
$$

donde $f$ puede ser una función como **ReLU, sigmoide o tangente hiperbólica**, dependiendo del diseño de la red.

Este proceso se repite capa por capa hasta llegar a la **capa de salida**, donde las activaciones finales representan la predicción del modelo. En el caso de una tarea de clasificación multiclase, por ejemplo, la capa de salida suele aplicar una función *softmax*, que convierte los valores en probabilidades normalizadas.

##### **Importancia de la función de pérdida en el aprendizaje**

El **objetivo de una MLP** es ajustar sus pesos de manera que la salida predicha sea lo más cercana posible a la salida real. Para medir qué tan bien está desempeñándose la red, se utiliza una **función de pérdida**, que cuantifica la diferencia entre la predicción y la etiqueta real.  

Para diferentes tipos de problemas, se emplean distintas funciones de pérdida:

En **regresión**, se usa el **error cuadrático medio (MSE)**:

$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

donde $y_i$ es el valor real y $\hat{y}_i$ la predicción del modelo.

En **clasificación binaria**, se emplea la **entropía cruzada binaria**:

$$
L = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log( \hat{y}_i) + (1 - y_i) \log (1 - \hat{y}_i) \right]
$$

que mide la diferencia entre la distribución de probabilidades predicha y la real.

En **clasificación multiclase**, se utiliza la **entropía cruzada categórica**, una extensión de la versión binaria:

$$
L = - \sum_{i} y_i \log (\hat{y}_i)
$$

Una función de pérdida bien elegida permite que el modelo aprenda de manera eficiente y mejore sus predicciones en cada iteración de entrenamiento.

> [!tip]
>
> La **propagación hacia adelante** es el mecanismo central en la predicción de una red neuronal multicapa. En este proceso, las entradas son transformadas a través de una serie de capas ocultas donde se aplican combinaciones lineales y funciones de activación, hasta obtener una salida final. La calidad de la predicción se evalúa mediante una función de pérdida, que determina qué tan lejos está la salida generada del resultado esperado. En la siguiente sección, exploraremos cómo este error se utiliza para ajustar los pesos del modelo mediante la **retropropagación del error**, el algoritmo clave detrás del aprendizaje de las MLP.



##### **Para reflexionar...**  
> **¿Cómo afecta la elección de la función de pérdida al proceso de aprendizaje de una red neuronal?** 
> **Clave**: Reflexiona sobre cómo diferentes funciones de pérdida impactan en la estabilidad del entrenamiento y en la convergencia del modelo. 



#### **Retropropagación del error: el mecanismo de aprendizaje en MLP**

Las redes neuronales multicapa han demostrado ser modelos altamente expresivos, pero su utilidad depende de un proceso de aprendizaje efectivo que permita ajustar sus pesos de manera óptima. El método que hace posible este ajuste es la **retropropagación del error** (*backpropagation*), un algoritmo fundamental que permite que las redes neuronales aprendan a partir de ejemplos. Su importancia en el desarrollo del Deep Learning es tal que, sin él, entrenar redes profundas sería prácticamente imposible.

##### **El problema del aprendizaje en redes neuronales**

En una MLP, cada conexión entre neuronas tiene un peso asociado que determina la influencia de una entrada en la activación de la siguiente capa. Inicialmente, estos pesos se establecen con valores aleatorios y se van actualizando progresivamente a medida que la red aprende. Pero surge una pregunta clave: **¿cómo determinar la dirección y magnitud del ajuste necesario en cada peso?**  

La respuesta radica en la función de pérdida, que mide la discrepancia entre la salida predicha por la red y el valor real esperado. En términos matemáticos, el aprendizaje de la red consiste en **minimizar la función de pérdida** ajustando los pesos de la manera más eficiente posible.

El algoritmo de **backpropagation**, formalizado en los años 80 por **Rumelhart, Hinton y Williams**, permite que las redes neuronales ajusten sus pesos de manera eficiente mediante la aplicación de la **regla de la cadena del cálculo diferencial**. Su principio fundamental se basa en distribuir el error cometido en la salida a lo largo de todas las capas de la red, de manera que cada conexión pueda ser ajustada en función de su contribución a dicho error.

El proceso de retropropagación ocurre en dos fases interdependientes. En la primera, conocida como **propagación hacia adelante**, los datos atraviesan la red desde la capa de entrada hasta la capa de salida, generando una predicción basada en los pesos actuales del modelo. Al finalizar esta etapa, la función de pérdida compara la salida obtenida con el valor real, cuantificando la discrepancia entre ambas.

A partir de esta medición, comienza la segunda fase, la **propagación hacia atrás**, en la que se calcula el **gradiente de la función de pérdida** con respecto a cada peso utilizando derivadas parciales. La regla de la cadena permite distribuir este gradiente a lo largo de la red, asignando correcciones proporcionales a cada conexión. Finalmente, los pesos son ajustados en la dirección opuesta al gradiente mediante una actualización controlada por la **tasa de aprendizaje**, un hiperparámetro que determina la magnitud de los cambios en cada iteración.

Este mecanismo de ajuste progresivo es lo que permite que una red neuronal aprenda patrones en los datos a través de múltiples ciclos de entrenamiento, refinando sus pesos hasta alcanzar una configuración óptima que minimice el error.

Matemáticamente, el ajuste de pesos se realiza con la regla de descenso de gradiente:

$$
w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta \frac{\partial L}{\partial w_{ij}^{(l)}}
$$

donde:
- $w_{ij}^{(l)}$ representa el peso entre la neurona $j$ de la capa $l-1$ y la neurona $i$ de la capa $l$.
- $\eta$ es la **tasa de aprendizaje**, un hiperparámetro que controla la magnitud del ajuste.
- $L$ es la función de pérdida, cuyo gradiente indica la dirección de ajuste de los pesos.

##### **Cálculo del gradiente con la regla de la cadena**

Dado que cada neurona recibe información de la capa anterior y transmite su activación a la siguiente, el error de cada peso depende de múltiples factores. Para calcular su contribución exacta, se utiliza la **regla de la cadena** del cálculo diferencial, descomponiendo el impacto de cada peso en el error total:

$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_i^{(l)}} \cdot \frac{\partial a_i^{(l)}}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial w_{ij}^{(l)}}
$$

donde:
- $\frac{\partial L}{\partial a_i^{(l)}}$ mide el impacto de la activación de la neurona en la pérdida total.
- $\frac{\partial a_i^{(l)}}{\partial z_i^{(l)}}$ representa la derivada de la función de activación.
- $\frac{\partial z_i^{(l)}}{\partial w_{ij}^{(l)}}$ captura la relación entre la suma ponderada de entradas y los pesos.

Esta descomposición permite calcular el gradiente de manera eficiente y aplicar el ajuste correspondiente en cada iteración.

##### **Importancia de la tasa de aprendizaje y sus efectos**

Uno de los aspectos más críticos en la retropropagación es la elección de la **tasa de aprendizaje ($\eta$)**. Si este valor es demasiado grande, la actualización de los pesos puede ser demasiado brusca, causando inestabilidad en el entrenamiento. Si es demasiado pequeño, la convergencia del modelo será extremadamente lenta.  

Para abordar este problema, se han desarrollado variantes del descenso de gradiente que mejoran la estabilidad y rapidez del aprendizaje, como **Adam, RMSprop y SGD con momentum**, que ajustan dinámicamente la magnitud del paso de actualización en función de la topología de la función de pérdida.

##### **Desafíos en la retropropagación: desvanecimiento y explosión del gradiente**

Aunque el algoritmo de retropropagación ha sido fundamental para el entrenamiento de redes neuronales, su aplicación en modelos profundos presenta desafíos significativos. A medida que el error se propaga hacia atrás a través de las capas, los gradientes pueden volverse demasiado pequeños o demasiado grandes, afectando la capacidad de aprendizaje del modelo.

El **desvanecimiento del gradiente** ocurre cuando los valores de los gradientes disminuyen progresivamente al retroceder por la red, haciendo que las capas más profundas apenas actualicen sus pesos. Este fenómeno es especialmente problemático cuando se utilizan funciones de activación como la sigmoide, cuya derivada tiende a valores muy pequeños en la mayor parte de su dominio. Como resultado, las primeras capas de la red tienen dificultades para aprender representaciones útiles, lo que limita la capacidad de la red para extraer características efectivas de los datos.

En el extremo opuesto, la **explosión del gradiente** se manifiesta cuando los gradientes crecen exponencialmente durante la retropropagación, generando cambios abruptos en los pesos y provocando inestabilidad en el entrenamiento. Este problema puede hacer que el modelo diverja, impidiendo la convergencia hacia una solución óptima.

Para mitigar estos efectos, se han desarrollado diversas estrategias que estabilizan el proceso de entrenamiento. La **normalización por lotes (batch normalization)** ha demostrado ser efectiva al mantener las activaciones dentro de un rango adecuado, reduciendo la propagación de valores extremos. Además, el uso de funciones de activación como **ReLU** evita la saturación de gradientes al eliminar la región de bajas derivadas que caracteriza a funciones como la sigmoide o la tangente hiperbólica. Estas técnicas han sido clave en la evolución de las redes neuronales profundas, permitiendo entrenamientos más estables y eficientes en arquitecturas de gran escala.

> [!tip]
>
> La **retropropagación del error** ha sido el pilar fundamental del aprendizaje en redes neuronales, permitiendo que las MLP ajusten sus pesos de manera eficiente a partir de los datos. Su implementación mediante el **descenso de gradiente y la regla de la cadena** ha hecho posible la construcción de modelos complejos capaces de aprender relaciones no triviales en los datos. Sin embargo, la efectividad del proceso de aprendizaje depende de factores clave como la tasa de aprendizaje y la mitigación de problemas como el desvanecimiento del gradiente, lo que ha llevado al desarrollo de técnicas avanzadas que optimizan el entrenamiento de redes neuronales profundas.

##### **Para reflexionar...**

> **¿Cómo influye el desvanecimiento del gradiente en la capacidad de aprendizaje de redes profundas?** 
> **Clave**: Analiza cómo la propagación del error a través de muchas capas afecta la actualización de pesos y cómo se pueden aplicar estrategias como la normalización por lotes o el uso de ReLU para contrarrestar este efecto. 

#### **Capacidades y limitaciones de las MLP**

Las redes neuronales multicapa (**MLP**) han marcado un avance significativo en la capacidad de los modelos de aprendizaje automático para resolver problemas complejos. Su capacidad para aprender representaciones jerárquicas y modelar relaciones no lineales ha permitido superar muchas de las limitaciones de los métodos tradicionales de Machine Learning. Sin embargo, su uso no está exento de desafíos, especialmente en términos de entrenamiento, interpretabilidad y eficiencia computacional.

##### **Comparación con métodos tradicionales de Machine Learning**

Antes del auge del Deep Learning, la mayoría de los problemas de clasificación y regresión se resolvían mediante algoritmos tradicionales como la regresión logística, los árboles de decisión o las máquinas de soporte vectorial (**SVM**). Estos enfoques tienen la ventaja de ser más interpretables y eficientes en términos computacionales, lo que los hace ideales para conjuntos de datos de tamaño moderado o cuando la interpretabilidad del modelo es crítica.

Sin embargo, estos métodos dependen en gran medida de la **ingeniería de características**, un proceso que requiere diseñar manualmente las representaciones más adecuadas para que el modelo pueda capturar patrones relevantes. En problemas de datos altamente dimensionales, como el reconocimiento de imágenes o el procesamiento de lenguaje natural, esta dependencia de características predefinidas se convierte en una limitación significativa.

Las MLP superan esta restricción al aprender **representaciones de los datos de manera automática**. En lugar de requerir una transformación previa de los datos, la propia red descubre y construye representaciones útiles a través de sus capas ocultas. Esta capacidad ha hecho que las redes neuronales sean superiores en tareas donde los patrones subyacentes son demasiado complejos para ser modelados explícitamente con reglas predefinidas.

##### **Ventajas de las redes multicapa en problemas no lineales**

Uno de los aspectos más destacados de las MLP es su capacidad para capturar relaciones no lineales en los datos. Mientras que un modelo de regresión logística, por ejemplo, solo puede aprender fronteras de decisión lineales, una MLP con suficientes capas y neuronas puede aproximar cualquier función matemática, según el **teorema de aproximación universal**.

Esta propiedad es clave en problemas donde los datos no pueden separarse mediante un simple hiperplano. Un caso paradigmático es la función **XOR**, donde ninguna línea recta puede dividir correctamente las clases. Mientras que un Perceptrón de una sola capa fracasa en este escenario, una MLP con al menos una capa oculta y funciones de activación no lineales puede aprender la separación correcta.

La capacidad de modelar relaciones no lineales ha sido determinante en la aplicación de MLP en áreas como el reconocimiento de patrones, el modelado de series temporales y la clasificación de imágenes. Sin embargo, esta misma capacidad de ajuste introduce nuevos desafíos en términos de entrenamiento y generalización.

##### **Desafíos en el entrenamiento: sobreajuste y convergencia**

Si bien las MLP ofrecen un poder expresivo considerable, su entrenamiento es más complejo que el de los modelos tradicionales. Entre los principales desafíos se encuentran el **sobreajuste** y la **convergencia del modelo**.

El **sobreajuste** ocurre cuando la red memoriza los datos de entrenamiento en lugar de generalizar patrones útiles. Esto sucede cuando la red es demasiado grande en relación con la cantidad de datos disponibles, lo que la lleva a ajustar sus parámetros de manera excesiva a ejemplos específicos. Como consecuencia, el modelo obtiene un desempeño excelente en el conjunto de entrenamiento, pero falla al enfrentarse a datos nuevos. Para mitigar este problema, se emplean estrategias como la **regularización L2 (weight decay)**, el **dropout** y la **normalización por lotes (batch normalization)**.

Por otro lado, la **convergencia del modelo** también es un factor crítico en el entrenamiento de MLP. Dado que la función de pérdida no es convexa, la optimización mediante **descenso de gradiente estocástico (SGD)** puede quedar atrapada en mínimos locales o sufrir problemas de **desvanecimiento del gradiente**, especialmente cuando la red tiene muchas capas. El uso de funciones de activación como **ReLU**, junto con técnicas avanzadas de optimización como **Adam o RMSprop**, ha permitido abordar estos problemas y mejorar la estabilidad del entrenamiento.

> [!tip]
>
> Las redes neuronales multicapa han demostrado ser modelos extremadamente versátiles, capaces de resolver problemas no lineales con un rendimiento superior al de los métodos tradicionales de Machine Learning. Sin embargo, su aplicación requiere un equilibrio cuidadoso entre capacidad de representación y generalización, evitando problemas como el sobreajuste y las dificultades en la convergencia. A medida que se introducen arquitecturas más profundas y técnicas de entrenamiento más avanzadas, las MLP han evolucionado hasta convertirse en la base del **Deep Learning moderno**, sentando las bases para arquitecturas más sofisticadas como las redes convolucionales y los Transformers.

##### **Para reflexionar...**

> **¿En qué situaciones podría ser preferible utilizar un modelo de Machine Learning tradicional en lugar de una MLP?**
>  **Clave**: Considera factores como el tamaño del conjunto de datos, la interpretabilidad del modelo y la eficiencia computacional en distintas aplicaciones.

### **Implementación práctica de redes neuronales simples**

Hasta ahora hemos explorado la teoría detrás de las **redes neuronales multicapa (MLP)** y su mecanismo de aprendizaje mediante **retropropagación del error**. Ahora es momento de trasladar estos conceptos a la práctica, construyendo y entrenando una **MLP desde cero** en uno de los frameworks más utilizados en la actualidad: **TensorFlow**.

En los siguientes ejemplos trabajaremos con el dataset **MNIST**, un conjunto de imágenes de dígitos escritos a mano ampliamente utilizado como punto de partida en aprendizaje automático. Nuestro objetivo será diseñar una red capaz de reconocer estos dígitos y ajustar su desempeño a través de hiperparámetros básicos.

#### **Construcción de una MLP en TensorFlow**

**TensorFlow** es una de las bibliotecas más utilizadas en Deep Learning. Desarrollada por Google, proporciona una infraestructura optimizada para el entrenamiento y despliegue de modelos de redes neuronales. Su integración con la API **Keras** simplifica la construcción de arquitecturas al ofrecer una interfaz de alto nivel que permite definir modelos de manera intuitiva. Su enfoque basado en gráficos computacionales permite optimizar la ejecución en dispositivos como GPUs y TPUs, lo que ha contribuido a su popularidad en aplicaciones industriales.

La implementación de una **MLP** en Keras sigue unos principios fundamentales: definir una arquitectura con capas totalmente conectadas, aplicar funciones de activación para introducir no linealidad y entrenar el modelo utilizando retropropagación del error. En las siguientes secciones, veremos cómo construir una **red neuronal multicapa** en este framework, analizando su desempeño en el dataset **MNIST**.

La construcción de una red neuronal en **TensorFlow** se realiza utilizando la API de **Keras**, lo que permite definir el modelo de forma sencilla mediante la clase `Sequential`.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carga del dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalización de los datos
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construcción de la MLP
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Convierte la imagen en un vector de 784 elementos
    layers.Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y activación ReLU
    layers.Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada dígito)
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluación del modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrecisión en el conjunto de prueba: {test_acc:.4f}")
```

Este código comienza cargando el dataset **MNIST**, que contiene imágenes de dígitos escritos a mano en escala de grises de **28x28 píxeles**. Como las redes neuronales trabajan mejor con valores normalizados, dividimos los datos entre 255 para llevar los valores de los píxeles al rango **[0,1]**.

El modelo se define con tres capas:

- **`Flatten`** transforma la imagen en un vector de 784 valores (28x28).
- **`Dense(128, activation='relu')`** añade una capa oculta de 128 neuronas con activación **ReLU**, que introduce no linealidad.
- **`Dense(10, activation='softmax')`** genera la salida con **10 neuronas**, una por cada dígito del 0 al 9, utilizando la activación **softmax** para convertir los valores en probabilidades.

El modelo se **compila** usando el optimizador **Adam**, que ajusta los pesos de manera eficiente, y la función de pérdida **sparse_categorical_crossentropy**, adecuada para clasificación multiclase. Finalmente, se **entrena** con cinco épocas y se evalúa en el conjunto de prueba.

#### Visualización de pesos y activaciones

Uno de los aspectos clave en el entrenamiento de redes neuronales es la interpretación de lo que el modelo está aprendiendo. Aunque las redes profundas pueden ser vistas como una "caja negra", existen métodos que permiten inspeccionar sus parámetros internos. Cada conexión en una MLP tiene un **peso** asociado, el cual se ajusta durante el entrenamiento para minimizar la función de pérdida. Inspeccionar estos pesos puede revelar información interesante sobre cómo la red procesa la información.

Para analizar los pesos de la primera capa oculta de nuestro modelo en TensorFlow, podemos acceder a ellos directamente y representarlos gráficamente.

```python
import matplotlib.pyplot as plt
import numpy as np

# Extraer pesos de la primera capa oculta
pesos_capa1 = model.layers[1].get_weights()[0]  # Los pesos están en la posición 0
pesos_capa1 = pesos_capa1.T  # Transponer para visualización

# Graficar algunos pesos como imágenes
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    if i < pesos_capa1.shape[0]:
        ax.imshow(pesos_capa1[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

plt.suptitle("Visualización de los filtros de la primera capa")
plt.show()
```

En este código, accedemos a los pesos de la primera capa densa con `model.layers[1].get_weights()`. Luego, transponemos la matriz para poder visualizar cada fila como una imagen de 28x28, lo que nos permite observar qué patrones está capturando cada neurona de la primera capa.

Si el modelo está correctamente entrenado, los filtros deberían parecerse a patrones de bordes o curvas, ya que la red está aprendiendo representaciones básicas de los dígitos.

##### **Visualización de activaciones de distintas capas**

Además de los pesos, es interesante analizar cómo se activan las neuronas al recibir una entrada. Esto nos permite ver qué características de la imagen están siendo enfatizadas en cada nivel de la red.

Para capturar las activaciones de una capa intermedia, utilizamos el modelo de Keras de la siguiente manera:

```python
# Seleccionar una imagen de prueba
imagen = x_test[0].reshape(1, 28, 28)

# Crear un modelo que devuelve las activaciones de la primera capa oculta
from tensorflow.keras.models import Model
modelo_intermedio = Model(inputs=model.input, outputs=model.layers[1].output)

# Obtener las activaciones
activaciones = modelo_intermedio.predict(imagen)

# Graficar las activaciones
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    if i < activaciones.shape[1]:
        ax.imshow(activaciones[0, i].reshape(8, 16), cmap='gray')
        ax.axis('off')

plt.suptitle("Activaciones de la primera capa oculta")
plt.show()
```

En este caso, creamos un **modelo intermedio** que toma la misma entrada que la MLP original, pero en lugar de devolver la predicción final, extrae la salida de la primera capa oculta. Al pasar una imagen de prueba por este modelo, obtenemos las activaciones de las neuronas, las cuales pueden visualizarse en forma de mapa de activaciones.

> [!note]
>
> La visualización de los pesos y activaciones en una red neuronal ofrece una visión más profunda sobre lo que el modelo está aprendiendo. Al observar los filtros de la primera capa, podemos intuir qué patrones están siendo detectados por la red. Del mismo modo, analizar las activaciones nos permite ver cómo las imágenes de entrada son transformadas a lo largo de las capas, proporcionando información clave para ajustar y mejorar el modelo.
>
> En problemas más complejos, como la clasificación de imágenes en redes convolucionales, estas técnicas de interpretación se vuelven aún más relevantes. En las siguientes sesiones, exploraremos arquitecturas más avanzadas, como las **Redes Convolucionales (CNN)**, que han demostrado ser especialmente eficaces en visión por computadora.



##### **Para reflexionar...**

> **¿Cómo podría ayudar la visualización de pesos y activaciones en el ajuste de hiperparámetros de la red?**
>  **Clave**: Reflexiona sobre cómo el análisis de los filtros aprendidos puede revelar posibles problemas, como la falta de diversidad en las activaciones o el ajuste excesivo a patrones específicos.



