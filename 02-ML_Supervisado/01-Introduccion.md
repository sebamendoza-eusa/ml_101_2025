# 1. Introducción

El **aprendizaje automático supervisado** es seguramente una de las ramas más importantes dentro del **machine learning** y se ha convertido en una herramienta esencial en la inteligencia artificial moderna. Ya hemos comentado a lo largo de este curso que su característica principal es la utilización de datos etiquetados para entrenar modelos predictivos que permite mapear entradas a salidas conocidas. El objeto es hacer predicciones precisas en nuevos datos. Este enfoque es aplicable tanto en problemas de clasificación como de regresión, y ha ganado una gran popularidad en la industria y en la investigación científica debido a su efectividad en una amplia gama de aplicaciones.

El aprendizaje supervisado se apoya en algoritmos capaces de aprender a partir de ejemplos previos, ajustando sus parámetros para minimizar el error entre sus predicciones y las etiquetas conocidas. Esta capacidad de aprendizaje permite a los modelos mejorar su rendimiento a medida que incorporan más datos. A continuación, profundizaremos en los fundamentos del aprendizaje supervisado, sus tipos de problemas, el flujo del proceso de aprendizaje, comparaciones con otros paradigmas de aprendizaje y las aplicaciones más comunes.

## Definición de Aprendizaje Automático Supervisado

Así pues, podemos definir el **aprendizaje automático supervisado** como:

> [!important]
>
> Subcampo del **machine learning** en el que un algoritmo que es  capaz de construir una función de mapeo $f(x)$ que asocia un conjunto de entradas $X$ (características o variables independientes) con un conjunto de salidas $Y$ (etiquetas o variables dependientes) a partir de un conjunto de ejemplos etiquetados.
>
> Formalmente, se busca encontrar una función $f: X \rightarrow Y$ que minimice una medida de error o pérdida entre las predicciones del modelo y las etiquetas verdaderas.

El proceso de entrenamiento consiste en ajustar los parámetros del modelo para minimizar una función de coste, como podría ser la **entropía cruzada** en problemas de clasificación o el **error cuadrático medio** (MSE) en problemas de regresión. Este ajuste de los parámetros se realiza utilizando un método de optimización, que puede variar según la naturaleza del problema

Para que el modelo generalice correctamente y no solo memorice los datos de entrenamiento, el proceso se lleva a cabo en diferentes fases e incorporando progresivamente distintos subconjuntos de datos:

- **Conjunto de entrenamiento**: Este conjunto se utiliza para ajustar los pesos del modelo.
- **Conjunto de validación**: Un subconjunto que permite evaluar el rendimiento del modelo durante el entrenamiento y ajustar hiperparámetros, como la tasa de regularización o la arquitectura del modelo.
- **Conjunto de prueba**: Se reserva para una evaluación final y permite medir la capacidad de generalización del modelo en datos no vistos.

> **Ejemplo**
>
> Un sistema de clasificación de correos electrónicos utiliza un conjunto de datos etiquetados de correos como "spam" o "no spam". El algoritmo aprende a reconocer patrones en los correos (como palabras clave o características del remitente) que indican si es probable que un correo sea spam. Utiliza estos patrones para predecir la categoría de nuevos correos electrónicos.

## Generalización: La clave del éxito de un modelo de aprendizaje automático supervisado

Una buena **generalización** en un modelo de aprendizaje automático supervisado es crucial para que el modelo pueda hacer predicciones precisas **en datos no vistos o nuevos**, más allá del conjunto de entrenamiento. La generalización depende de varios factores que afectan la capacidad del modelo para evitar el **sobreajuste** (cuando el modelo se ajusta demasiado a los datos de entrenamiento) y **subajuste** (cuando el modelo es demasiado simple para capturar la complejidad de los datos). A continuación, se describen los factores más importantes que influyen en la capacidad de generalización de un modelo supervisado:

### Complejidad del Modelo

En primer lugar, el equilibrio entre la **complejidad** del modelo y la cantidad de datos disponibles es fundamental para una buena generalización. Por un lado podemos encontrar **modelos simples** (como por ejemplo la regresión lineal), que pueden tener una capacidad limitada para capturar patrones complejos en los datos y llevar al **subajuste**, es decir, la incapacidad del modelo de captar correctamente los patrones subyacentes de los datos.

En otras ocasiones tendremos que trabajar con **modelos complejos** (como por ejemplo las redes neuronales profundas). Estos modelos tienen una gran capacidad para aprender patrones complejos, pero también tienen un mayor riesgo de **sobreajuste**, es decir, aprender el ruido específico del conjunto de entrenamiento en lugar de los patrones generales.

Al final, la clave para una buena generalización es encontrar un modelo con la complejidad adecuada, lo suficientemente flexible para capturar los patrones relevantes pero no tan complejo que modele las fluctuaciones aleatorias o el ruido en los datos.

### Cantidad y Calidad de los Datos

Ya sabemos lo importante de la cantidad y calidad de los datos de entrenamiento para conseguir una buena generalización. Hemos visto como, en general, **a medida que aumenta la cantidad de datos de entrenamiento**, la capacidad de generalización del modelo mejora. Ello es porque más datos proporcionan una representación más precisa del espacio de entrada, permitiendo que el modelo capture patrones más robustos y representativos.

Sin embargo, más datos por sí solos no garantizan una buena generalización. También es importante que los datos sean **representativos** de la distribución real que se quiere modelar. 

Por otro lado hemos de vigilar la **calidad de los datos**. Los datos deben ser limpios, sin errores, inconsistencias o valores atípicos. Datos de baja calidad, con etiquetas incorrectas o variables mal definidas, pueden llevar a que el modelo aprenda patrones incorrectos, lo que afecta negativamente su capacidad de generalización. El preprocesamiento de datos, como la eliminación de valores faltantes, la normalización y la transformación adecuada de características, es esencial para garantizar que el modelo generalice bien.

### Mejorar la capacidad de generalización

Existen técnicas y herramientas variadas para mejorar la capacidad de generalización de nuestro modelo y evitar los problemas ya identificados  de sobreajuste o subajuste. Todas ellas pueden agruparse como sigue:

#### Regularización

La **regularización** es una técnica para evitar el sobreajuste **limitando la complejidad del modelo**. Existen diferentes tipos de regularización que se pueden aplicar dependiendo del algoritmo. Algunas, como la **regularización L1 (Lasso)** o la **regularización L2 (Ridge)**, se introduce una penalización en la función de pérdida del modelo. Otras, como el ***dropout***, aplicada a redes neuronales, desactivan aleatoriamente neuronas de la red durante el entrenamiento para evitar que la red dependa demasiado de una neurona específica y así forzar la red a aprender patrones más distribuidos.

#### Selección de características

Una buena selección de características también mejora la capacidad de generalización de un modelo. Si se incluyen demasiadas características irrelevantes o ruidosas, el modelo puede verse sobreajustado a estas características, lo que perjudica su rendimiento en datos nuevos. Para evitar lo anterior, podremos optar en ocasiones por realizar **ingeniería de características**, es decir, crear nuevas características basadas en el conocimiento del dominio del problema para mejorar la capacidad del modelo para capturar relaciones subyacentes importantes. En otras ocasiones podremos optar por **reducir la dimensionalidad** en aquellos casos donde hay muchas características. Mediante técnicas como el **Análisis de Componentes Principales (PCA)** o la **Selección de Características** (usando algoritmos como Lasso o Árboles de Decisión), podremos eliminar aquellas que aportan poca información.

#### Hiperparámetros y ajuste de hiperparámetros

El **ajuste de hiperparámetros** es otro conjunto de herramientas importante para mejorar la capacidad de generalización. Los hiperparámetros son **aquellos parámetros del modelo que no se aprenden durante el entrenamiento**, sino que deben definirse antes de entrenar. Estos hiperparámetros influyen directamente en la capacidad de generalización del modelo. Por ejemplo,  la **tasa de aprendizaje** es un hiperparámetro clave en muchos algoritmos, que determina cuán grandes son los pasos que toma el modelo al ajustar los pesos. Una tasa de aprendizaje demasiado alta puede llevar a un aprendizaje inestable, mientras que una tasa demasiado baja puede hacer que el modelo tarde mucho en converger o incluso quede atrapado en mínimos locales. Otro hiperparámetro habitual es el **número de neuronas o capas** en modelos de redes neuronales. El número de capas y neuronas afecta directamente la complejidad del modelo. Más neuronas o capas pueden permitir capturar patrones más complejos, pero también aumentan el riesgo de sobreajuste.

En general, el ajuste de estos hiperparámetros se realiza utilizando un conjunto de validación o técnicas de **validación cruzada**.

Concretamente la **validación cruzada** es una técnica clave para evaluar la capacidad de generalización de un modelo. La más común es la denominada **validación cruzada k-fold**, en la que los datos se dividen en $k$ subconjuntos. El modelo se entrena en $k-1$ subconjuntos y se evalúa en el subconjunto restante, repitiendo este proceso $k$ veces. La media de los resultados proporciona una estimación más robusta del rendimiento del modelo en datos no vistos.

Este proceso es especialmente **útil cuando se dispone de un conjunto de datos limitado**, ya que permite que todos los datos sean utilizados tanto para el entrenamiento como para la evaluación.

### Sesgo y Varianza

El equilibrio entre **sesgo** y **varianza** es otro aspecto clave para conseguir una buena generalización. Idealmente, un modelo debe tener un bajo sesgo y una baja varianza para lograr un buen rendimiento en datos no vistos.

El **sesgo** hace referencia a los errores que introduce el modelo debido a suposiciones simplificadas sobre los datos. Un modelo con alto sesgo es demasiado rígido y es probable que subajuste. Por su parte, la **varianza** refleja la sensibilidad del modelo a las pequeñas variaciones en los datos de entrenamiento. Un modelo con alta varianza es probable que sobreajuste, aprendiendo detalles y ruido específico de los datos de entrenamiento que no generalizan bien.

Un buen modelo debe encontrar un equilibrio entre el sesgo y la varianza para lograr una generalización óptima. Las técnicas de regularización, como las mencionadas anteriormente, ayudan a reducir la varianza, mientras que modelos más simples pueden reducir el sesgo.

> **Ejemplo**
>
> Supongamos que estamos entrenando modelos para predecir el precio de una casa en función de características como el tamaño, el número de habitaciones y la ubicación.
>
> **Modelo con alto sesgo**: Si utilizamos un modelo muy simple, como la regresión lineal sin términos polinómicos, el modelo podría subajustarse. Esto significa que el modelo es demasiado rígido para capturar las relaciones complejas entre las características de entrada y el precio de la casa. Como resultado, el modelo tendrá un error de predicción alto tanto en los datos de entrenamiento como en los datos de prueba. Este tipo de error es el resultado de un alto **sesgo**, ya que el modelo hace suposiciones simplificadas sobre los datos.
>
> **Modelo con alta varianza**: Si, en cambio, utilizamos un modelo muy complejo, como una red neuronal profunda o un polinomio de alto grado, el modelo puede ajustarse demasiado a los datos de entrenamiento. En este caso, el modelo aprenderá no solo los patrones generales, sino también las fluctuaciones y el ruido en los datos de entrenamiento. Como resultado, aunque el modelo tenga un error bajo en los datos de entrenamiento, su error en los datos de prueba será alto. Esto es un ejemplo de **alta varianza**, ya que el modelo es muy sensible a las variaciones en los datos de entrenamiento y no generaliza bien a nuevos datos.
>
> El objetivo final será encontrar un equilibrio entre el sesgo y la varianza para que el modelo pueda capturar los patrones generales en los datos de entrenamiento sin sobreajustarse, de manera que tenga un buen rendimiento en los datos de prueba.

### Conjunto de datos representativo

Finalmente, es fundamental que el conjunto de datos utilizado para entrenar el modelo sea representativo de la población general de datos. Si los datos de entrenamiento no reflejan correctamente la distribución de los datos del mundo real, el modelo tendrá dificultades para generalizar. El origen de los problemas con la representatividad de los datos de entrenamiento hay que buscarlo, por un lado, en el **sesgo en la recolección de datos**: Si los datos de entrenamiento están sesgados o no son representativos de la población en su conjunto, el modelo aprenderá patrones incorrectos y no generalizará bien. Por otro lado, en el **desbalanceo de clases**. En problemas de clasificación, si una clase está desproporcionadamente representada en los datos de entrenamiento (por ejemplo, en la detección de fraudes donde la clase "fraude" es mucho menos común que la clase "no fraude"), el modelo puede estar sesgado hacia la clase mayoritaria. Ya hemos mencionado como esto se puede mitigar con técnicas como el sobremuestreo de la clase minoritaria o el submuestreo de la clase mayoritaria.

## Tipos de problemas en aprendizaje supervisado

El aprendizaje supervisado aborda principalmente dos tipos de problemas: los denominados de **clasificación** y los denominados de **regresión**.

En los problemas de clasificación, el objetivo es predecir una etiqueta que pertenece a una categoría o clase predefinida. Por ejemplo, en tareas de reconocimiento de imágenes, el modelo tiene que decidir si una imagen contiene un gato o un perro, o bien, en el ámbito del correo electrónico, clasificar si un mensaje es spam o no. Esta técnica es ampliamente utilizada en sectores, por ejemplo, como la visión por computadora, donde el modelo debe identificar objetos en imágenes, la detección de fraudes financieros, donde se busca identificar transacciones anómalas, o en el ámbito médico, para diagnosticar enfermedades a partir de los síntomas del paciente.

Por otro lado, los problemas de **regresión** tienen como objetivo **predecir un valor continuo**, en lugar de asignar una clase. Un ejemplo clásico de este tipo de problema es la predicción del precio de una vivienda, que se realiza basándose en variables como el tamaño, la ubicación o el número de habitaciones. Este enfoque se aplica también en áreas como la predicción de precios de acciones, el pronóstico del clima o la estimación de la demanda de productos en series temporales.

Es importante destacar que el tipo de problema al que se enfrenta el modelo supervisado no solo determina la naturaleza de los algoritmos utilizados, sino también las métricas con las que se evaluará su rendimiento. En problemas de clasificación, se suelen utilizar métricas como la precisión o la exactitud, mientras que en regresión es común emplear el error cuadrático medio (MSE) o el error absoluto medio (MAE) para medir la precisión de las predicciones.

## Flujo del aprendizaje supervisado

Ya vimos en el tema anterior como cualquier proyecto de aprendizaje automático sigue un flujo de trabajo muy determinado. En el caso del aprendizaje supervisado la estructura es prácticamente la misma que ya se vio. Recordemos que los pasos esenciales para desarrollar un modelo supervisado eran:

1. **Recolección de datos**: Recopilación de un conjunto de datos etiquetados que será utilizado para entrenar y probar el modelo. Es importante que los datos sean representativos del problema que se está tratando de resolver.

2. **Análisis exploratorio de datos**: Técnicas utilizadas para comprender mejor los datos, identificar patrones, detectar anomalías, probar hipótesis y resumir sus características clave mediante estadísticas descriptivas y visualizaciones.

3. **Preprocesamiento de datos**:  Conjunto de técnicas que se aplican a los datos con el objetivo de **transformarlos y prepararlos** para ser utilizados por modelos de machine learning

4. **División del conjunto de datos**: Obtención de los conjuntos de entrenamiento, validación y prueba a partir del conjunto inicial, asegurando que el modelo se entrene en una parte de los datos y se evalúe en una parte distinta no vista.

5. **Entrenamiento del modelo**: Proceso de aprendizaje de los patrones subyacentes en los datos. El algoritmo ajusta sus parámetros internos para minimizar el error entre sus predicciones y las etiquetas reales.

6. **Validación y ajuste del modelo**: Ejecución del modelo con los datos de validación. Las métricas utilizadas, que permiten evaluar la capacidad de generalización, podrán variar dependiendo de si se trata de un problema de clasificación o de regresión.

7. **Ajuste de hiperparámetros**: Fase de ajuste para optimizar el rendimiento de aquellos elementos que no intervienen en el proceso de aprendizaje y que fueron decididos con anterioridad.

8. **Predicción final**: Una vez que el modelo ha sido ajustado y validado, puede ser utilizado para hacer predicciones con nuevos datos no etiquetados.

## Diferencias con otros paradigmas de aprendizaje

El aprendizaje supervisado es solo uno de los diversos paradigmas dentro del aprendizaje automático, concretamente diseñado para abordar un tipo de problema y su solución basada en datos. Una de las principales diferencias del aprendizaje supervisado en comparación con otros enfoques es la **disponibilidad de etiquetas para guiar el proceso de entrenamiento**. Sin embargo, ya hemos visto como existen otros enfoques alternativos para encontrar soluciones a problemas de distinta naturaleza. A diferencia del aprendizaje supervisado, el **aprendizaje no supervisado** no requiere etiquetas. En este enfoque, el modelo debe identificar patrones o estructuras subyacentes en los datos de manera autónoma, sin ningún tipo de guía previa. Un ejemplo clásico de este paradigma es el **agrupamiento** o **clustering**, donde el objetivo es dividir los datos en grupos o clústeres de objetos similares, aunque se desconoce de antemano a qué grupo pertenecen. Algoritmos como **K-means** son ejemplos típicos utilizados para esta tarea.

En medio de ambos paradigmas se encuentra el **aprendizaje semi-supervisado**, que combina lo mejor de ambos mundos. Este enfoque aprovecha una pequeña cantidad de datos etiquetados junto con una gran cantidad de datos no etiquetados para mejorar la capacidad de generalización del modelo. Es especialmente útil en situaciones donde etiquetar los datos es costoso o difícil, pero los datos sin etiquetar son fáciles de recolectar en grandes cantidades.

Por último, el **aprendizaje por refuerzo** representa un enfoque completamente distinto. Aquí, un agente interactúa con un entorno y aprende a tomar decisiones para maximizar una recompensa acumulada a lo largo del tiempo. En lugar de basarse en ejemplos estáticos, como en los paradigmas supervisado o no supervisado, el agente del aprendizaje por refuerzo aprende mediante ensayo y error en un proceso dinámico. Este enfoque es especialmente útil en contextos donde las decisiones presentes influyen en el estado futuro del sistema, como sucede en los videojuegos o en la conducción autónoma. A través de la interacción con el entorno, el agente refuerza aquellas acciones que resultan en recompensas positivas y evita las que generan penalizaciones, optimizando su estrategia de comportamiento en función de las experiencias previas.

> **Ejemplo**
>
> En un entorno de aprendizaje por refuerzo, como en el caso de un robot que debe aprender a navegar por una habitación, el robot recibe una recompensa cuando alcanza su objetivo, lo que refuerza la idea de que ciertas acciones son preferibles a otras.

## Aplicaciones del Aprendizaje Supervisado

El aprendizaje supervisado tiene una amplia gama de aplicaciones en diversas industrias, y es utilizado para resolver problemas que requieren predicciones precisas basadas en datos históricos. A continuación, presentamos algunas de las áreas más destacadas donde se emplea:

- **Visión por computadora**: Se utiliza para tareas como el reconocimiento de objetos, detección de rostros y clasificación de imágenes. Modelos como las redes neuronales convolucionales (CNN) se entrenan para identificar patrones visuales en grandes conjuntos de imágenes etiquetadas.

- **Procesamiento del lenguaje natural (NLP)**: En este campo, se aplican algoritmos supervisados para tareas como la clasificación de texto, análisis de sentimiento y traducción automática. Modelos como los Transformers han revolucionado el campo con aplicaciones como GPT-3, que puede generar lenguaje natural coherente a partir de entradas textuales.

- **Detección de fraudes**: En el sector bancario y de pagos, los modelos supervisados ayudan a identificar patrones anómalos en transacciones financieras, lo que permite detectar y prevenir fraudes en tiempo real.

- **Diagnóstico médico**: El aprendizaje supervisado se emplea para analizar imágenes médicas, predecir el desarrollo de enfermedades o ayudar en la clasificación de tumores. Modelos entrenados con datos médicos pueden asistir a los profesionales en la toma de decisiones clínicas.

> **Ejemplo**
>
> En un hospital, un sistema basado en aprendizaje supervisado puede analizar radiografías y ayudar a los médicos a diagnosticar enfermedades pulmonares con una precisión similar a la de los especialistas humanos.

## Limitaciones del aprendizaje supervisado

El aprendizaje supervisado es una técnica poderosa y ampliamente utilizada, pero no está exenta de limitaciones que pueden afectar su efectividad en ciertos contextos. Una de las principales restricciones es la **necesidad de grandes volúmenes de datos etiquetados**. Para entrenar correctamente un modelo supervisado, se requiere un conjunto de datos suficientemente amplio y con etiquetas precisas. La obtención de estos datos puede ser un desafío, tanto en términos de coste como de tiempo, especialmente en áreas donde las etiquetas deben ser generadas manualmente por expertos. Este proceso puede ralentizar el desarrollo del modelo y aumentar considerablemente los recursos necesarios para su implementación.

Otra limitación importante es el fenómeno ya conocido como **sobreajuste** (o **overfitting)**.Ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento, aprendiendo no solo los patrones subyacentes, sino también las particularidades y el ruido inherente a los datos. Como resultado, el modelo pierde su capacidad para generalizar correctamente cuando se enfrenta a datos nuevos. Esto es especialmente común en modelos muy complejos en relación con el volumen de datos disponible. El sobreajuste puede ser mitigado mediante técnicas de regularización, selección adecuada de características o ajustando la complejidad del modelo.

Además de estos problemas técnicos, el uso del aprendizaje supervisado en ciertos contextos presenta también **implicaciones éticas** importantes. En áreas donde los modelos son utilizados para predecir comportamientos humanos, como en la contratación de personal o en sistemas judiciales, el uso de datos sesgados puede resultar en predicciones injustas o discriminatorias. Por ejemplo, si un modelo se entrena con datos que contienen sesgos raciales o de género, estos sesgos pueden no solo ser replicados, sino incluso amplificados en las predicciones, afectando negativamente a ciertos grupos. Esto hace que el diseño y la validación de estos modelos requieran un análisis cuidadoso para garantizar que las decisiones automatizadas sean justas y equitativas.

> **Ejemplo**
>
> Un sistema de contratación automatizada basado en aprendizaje supervisado puede discriminar contra ciertos grupos si los datos históricos utilizados para entrenarlo no están equilibrados o son parciales.

**Para reflexionar...**

> **¿Cuáles son las principales limitaciones del aprendizaje supervisado en proyectos de machine learning?** 
> **Clave**: Reflexiona sobre la necesidad de grandes cantidades de datos etiquetados y los costos asociados a obtenerlos. Piensa en problemas donde las etiquetas correctas no están fácilmente disponibles.
