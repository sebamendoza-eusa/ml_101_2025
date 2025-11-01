# 4. Etapas en un proyecto ML

El desarrollo de un proyecto de **aprendizaje automático (ML)** sigue una serie de fases cruciales, cada una contribuyendo al éxito del modelo y su implementación final. En esta ampliación, se introduce también la **puesta en producción** del modelo y su **mantenimiento** durante su ciclo de vida, aspectos fundamentales que garantizan la utilidad y longevidad de un proyecto ML.

El resumen de todas la fases del ciclo de vida proyecto ML típico sería el siguiente:

> [!important]
>
> 1. **Definición del problema**: Clarificar el objetivo del proyecto, qué problema se va a resolver y qué tipo de modelo se requiere (supervisado o no supervisado).
>
> 2. **Recolección de datos**: Recopilar datos relevantes y suficientes para entrenar el modelo. Estos datos pueden ser estructurados o no estructurados.
>
> 3. **Análisis exploratorio de datos (EDA):** Entender las características principales de un conjunto de datos mediante técnicas estadísticas y visualizaciones, con el objetivo de descubrir patrones, detectar anomalías, probar hipótesis y verificar supuestos.
>
> 4. **Preprocesamiento de datos**: Limpiar, normalizar y transformar los datos para prepararlos para el modelo. Incluye la gestión de valores faltantes y la codificación de variables.
>
> 5. **Selección del modelo**: Elegir el algoritmo adecuado en función del tipo de problema y los datos disponibles.
>
> 6. **Entrenamiento del modelo**: Utilizar los datos de entrenamiento para ajustar los parámetros del modelo.
>
> 7. **Validación**: Evaluar el rendimiento del modelo en datos de validación para evitar sobreajuste y optimizar hiperparámetros.
>
> 8. **Evaluación**: Probar el modelo con datos de prueba para medir su capacidad de generalización.
>
> 9. **Despliegue**: Implementar el modelo en un entorno de producción para hacer predicciones o automatizar decisiones.
>
> 10. **Monitoreo y mantenimiento**: Supervisar el rendimiento del modelo en el tiempo y actualizarlo si los datos o el entorno cambian.

## Definición del problema

La **definición del problema** es la fase fundamental en cualquier proyecto de machine learning (ML), ya que establece el enfoque del modelo y determina qué tipo de aprendizaje será más adecuado. En esta etapa, se identifican los objetivos del proyecto, las preguntas clave que se deben responder, y se clarifican las metas de negocio o científicas que se pretenden alcanzar con el uso de ML.

Uno de los aspectos más importantes es **entender el problema en términos de negocio o ciencia** antes de formularlo como un problema de ML. Esto implica reunir a expertos del dominio específico y a científicos de datos para asegurar que el proyecto se oriente hacia la resolución de un problema concreto y no hacia la construcción de un modelo innecesariamente complejo. 

Otro elemento clave es la **definición clara de las entradas y salidas** esperadas del modelo. Se debe decidir si el problema se puede resolver con un enfoque **supervisado** (donde los datos de entrenamiento incluyen las etiquetas correctas) o con un enfoque **no supervisado** (donde el objetivo es encontrar patrones ocultos sin etiquetas). También es importante determinar si se busca resolver un problema de **clasificación**, **regresión**, o de **agrupamiento**.

Además, en esta fase se establece cómo se medirá el éxito del modelo. Esto puede implicar la elección de una **métrica de evaluación** adecuada según la naturaleza del problema (precisión, error cuadrático medio (MSE), etc.). Definir correctamente los indicadores de rendimiento garantiza que se pueda evaluar de manera objetiva si el modelo está cumpliendo o no con los objetivos planteados.

Por último, la fase de definición del problema debe tomar en cuenta los **recursos** disponibles, como la cantidad y calidad de los datos, el tiempo disponible y las limitaciones computacionales, lo que puede influir en la elección de los algoritmos y la estrategia de modelado.

> **Ejemplo**: Si se quiere construir un modelo para predecir el abandono de clientes en una empresa de telecomunicaciones, es importante definir el problema como un problema de **clasificación binaria**, donde la variable de salida puede tomar dos valores: "abandono" o "no abandono". 

> **Ejemplo**: En un proyecto de **regresión** para predecir el precio de viviendas en una ciudad, el problema debe definirse claramente como uno en el que se buscan **predicciones continuas** basadas en variables de entrada como el tamaño de la casa, el número de habitaciones y la ubicación.

> **Ejemplo**: Si el objetivo es segmentar a los clientes de un e-commerce para personalizar campañas de marketing, se define el problema como uno de **clustering no supervisado**, donde el modelo debe agrupar a los clientes según patrones de comportamiento sin etiquetas predeterminadas.

**Para reflexionar...**

> **¿Cómo afecta una mala definición del problema al éxito de un proyecto de machine learning?**
>
> **Clave**: Un problema mal definido puede llevar a la recolección de datos incorrectos, selección inadecuada de métricas de éxito y, en consecuencia, a la construcción de un modelo que no resuelva el problema real.

## Recolección de datos

La **recolección de datos** es una fase crítica en cualquier proyecto de machine learning (ML), ya que la calidad y cantidad de los datos recolectados influirán directamente en el rendimiento del modelo. En esta etapa, el objetivo es obtener los datos adecuados que reflejen fielmente el fenómeno que se busca modelar. Los datos deben ser representativos, completos y relevantes para el problema en cuestión.

Uno de los primeros pasos es determinar **las fuentes de datos**. Dependiendo del proyecto, los datos pueden provenir de diversas fuentes. Hablamos de bases de datos internas de una empresa, APIs, sensores, redes sociales, o incluso ser generados por simulaciones. En algunos casos, se pueden utilizar datasets abiertos disponibles en plataformas como **Kaggle** o **UCI Machine Learning Repository**. La selección de las fuentes depende de la accesibilidad y la calidad de los datos, así como de las restricciones éticas y legales que puedan estar involucradas, como la privacidad y la protección de datos sensibles.

Es importante también considerar la **cantidad de datos**. Para muchos algoritmos de ML, como los basados en deep learning, se necesitan grandes volúmenes de datos para entrenar modelos efectivos. Sin embargo, más datos no siempre implican mejor rendimiento. Si los datos no son relevantes o de buena calidad, pueden introducir ruido y llevar a resultados imprecisos. Por lo tanto, la **calidad de los datos** es esencial y esto implica asegurarse de que los datos sean precisos, completos y no contengan errores o valores atípicos.

Una vez recolectados los datos, es crucial realizar una **exploración inicial** para entender sus características. Esto incluye analizar la distribución de las variables, la existencia de datos faltantes, y el formato en el que se encuentran. Este análisis ayuda a determinar si los datos necesitan transformaciones adicionales antes del entrenamiento, como la limpieza o normalización.

Finalmente, se deben tomar en cuenta aspectos éticos y legales, como el **consentimiento de los usuarios** cuando los datos provienen de fuentes personales o sensibles, así como la normativa aplicable, como el **Reglamento General de Protección de Datos (GDPR)** en Europa o la normativa específica en materia de uso de la IA.

> **Ejemplo**: En un proyecto de ML para predecir el precio de automóviles usados, los datos pueden recolectarse de **plataformas en línea de venta de coches**. Es crucial asegurarse de que los datos contengan características relevantes, como el año de fabricación, el kilometraje y la marca del automóvil.

> **Ejemplo**: Para un modelo de detección de fraudes en transacciones bancarias, los datos se obtienen de las bases de datos de transacciones de un banco. En este caso, es importante garantizar la **seguridad y confidencialidad** de los datos, cumpliendo con las regulaciones de protección de datos.

> **Ejemplo**: En un proyecto de **análisis de sentimientos** sobre productos en redes sociales, los datos pueden extraerse mediante **web scraping** de plataformas como Twitter o Instagram. En este caso, es fundamental tener en cuenta las políticas de uso de las plataformas y el manejo adecuado de datos personales.

**Para reflexionar...**

> **¿Cuáles son los riesgos de utilizar datos de baja calidad o mal recolectados en un proyecto de machine learning?**
>
> **Clave**: Los datos de baja calidad pueden introducir sesgos, errores en las predicciones y afectar negativamente la capacidad del modelo para generalizar a nuevos datos, comprometiendo su utilidad en aplicaciones reales.

## Análisis exploratorio de datos (EDA)

El **análisis exploratorio de datos (EDA)** es la primera etapa en cualquier proyecto de machine learning, ya que permite comprender mejor las características y estructura de los datos antes de aplicar cualquier modelo. El principal objetivo del EDA es detectar patrones, relaciones entre variables, distribuciones y anomalías en los datos, proporcionando una base sólida para la toma de decisiones en las fases posteriores del proyecto.

El EDA se realiza mediante dos enfoques principales: **análisis descriptivo** y **análisis gráfico**. El análisis descriptivo incluye el cálculo de estadísticas que resumen los datos numéricos, como la **media**, **mediana**, **moda**, **desviación estándar**, **percentiles** y **rango intercuartílico**. Estas herramientas permiten tener una visión inicial de cómo se distribuyen las variables y cómo varían entre sí. Este análisis también es útil para identificar variables que tienen valores extremos o una gran variabilidad, lo que puede afectar el rendimiento del modelo.

Por otro lado, el análisis gráfico facilita la visualización de las relaciones y distribuciones. Entre las herramientas más utilizadas para el EDA gráfico se incluyen:
- **Histogramas**, que muestran cómo se distribuyen los valores de una variable continua.
- **Diagramas de caja (boxplots)**, útiles para detectar outliers y visualizar la dispersión de los datos.
- **Gráficos de dispersión (scatter plots)**, que permiten observar correlaciones entre dos variables numéricas.
- **Gráficos de barras**, empleados para mostrar frecuencias en variables categóricas.

Estas herramientas no solo ayudan a detectar posibles problemas en los datos, sino que también ofrecen pistas sobre qué tipo de relaciones pueden existir entre las variables y cómo estas influencian el resultado que queremos predecir. Un análisis más detallado puede incluir **mapas de calor de correlación**, que muestran visualmente las correlaciones entre las diferentes variables numéricas del conjunto de datos. Estos gráficos permiten identificar rápidamente qué pares de variables tienen una relación fuerte, lo que puede ser útil para simplificar el modelo eliminando variables redundantes o poco relevantes.

El análisis exploratorio también incluye la **detección de patrones**, como la **distribución normal** de los datos o la presencia de relaciones lineales o no lineales. Identificar estos patrones es fundamental para seleccionar las técnicas de modelado adecuadas. Si los datos no siguen una distribución normal, por ejemplo, se podrían considerar transformaciones adicionales para hacer que las variables se ajusten mejor a los requisitos del modelo que se utilizará.

Otra técnica importante dentro del EDA es el **análisis de correlación**, que mide la relación entre dos variables numéricas. Esta herramienta es fundamental en problemas de regresión, donde la fuerza y dirección de las relaciones entre variables explicativas y la variable dependiente determinan el diseño y la complejidad del modelo. Una correlación fuerte y positiva indicaría que a medida que una variable aumenta, la otra también lo hace, mientras que una correlación negativa implicaría que una aumenta a medida que la otra disminuye.

Finalmente, el EDA también nos ayuda a comprender si los datos están **equilibrados o desbalanceados**. Este análisis es especialmente importante en problemas de clasificación, donde es crucial que las clases estén equilibradas para evitar sesgos en el modelo. Si, por ejemplo, un conjunto de datos contiene un 95% de un tipo de clase y solo un 5% de otra, el modelo podría tener dificultades para aprender correctamente sobre la clase minoritaria.

> **Ejemplo**: En un proyecto de análisis de datos de salud, un histograma de la variable "edad" puede mostrar cómo está distribuida esta variable entre los pacientes y detectar si hay un sesgo hacia ciertos grupos etarios que podría afectar la representatividad del modelo.

> **Ejemplo**: En un estudio sobre el consumo energético de edificios, un gráfico de dispersión entre el área del edificio y su consumo total de energía podría revelar una correlación lineal, lo que sugiere que a medida que aumenta el tamaño del edificio, también lo hace su consumo energético.

> **Ejemplo**: En una investigación sobre precios de viviendas, un mapa de calor de correlaciones podría mostrar una alta correlación entre el número de habitaciones y el precio de venta, lo que indicaría que esta variable debe ser tomada en cuenta en la construcción del modelo.

**Para reflexionar...**

> **¿Por qué es importante realizar tanto análisis descriptivos como gráficos en el EDA y cómo se complementan entre sí?**
> **Pistas**: Reflexiona sobre cómo los análisis descriptivos permiten resumir numéricamente los datos mientras que los gráficos ofrecen una representación visual que facilita la detección de patrones y anomalías.

## Preprocesamiento de los datos

El **preprocesamiento de datos** es una etapa compleja, ya que los modelos de ML requieren datos limpios, consistentes y bien estructurados para poder aprender patrones de manera eficiente. Sin un preprocesamiento adecuado, los modelos pueden obtener resultados sesgados, incorrectos o poco confiables. El preprocesamiento se refiere a una serie de técnicas y procesos que transforman los datos brutos en un formato adecuado para el modelado.

### **Limpieza de datos**
El primer paso en el preprocesamiento es la **limpieza de datos**. Los datos reales suelen contener errores, valores faltantes o inconsistencias. Los datos incompletos, como registros que tienen valores nulos o ausentes, pueden sesgar los resultados del modelo. Existen varias técnicas para tratar los valores faltantes, como:

- **Eliminación** de registros o variables con datos faltantes. Esto es útil cuando la cantidad de datos faltantes es pequeña en relación con el tamaño del conjunto de datos.
- **Imputación**, donde se reemplazan los valores faltantes por otros valores, como la media, la mediana o la moda de la variable en cuestión, o utilizando métodos más avanzados como la imputación por modelos predictivos.

Los datos también pueden tener **outliers**, que son valores anómalos que pueden distorsionar el rendimiento del modelo. Estos se pueden identificar mediante técnicas estadísticas (como el uso de percentiles) y se pueden eliminar o ajustar.

### Escalado de datos
En muchos algoritmos de machine learning, especialmente **aquellos que se basan en la distancia** (como las redes neuronales o los modelos de regresión), es importante que los datos estén en una escala similar. Las variables que tienen rangos de valores muy diferentes pueden tener un impacto desproporcionado en el modelo, por lo que se debe aplicar **normalización** o **estandarización**.

La **normalización** convierte los valores de las variables a un rango entre 0 y 1. Esto se utiliza cuando se espera que los datos sigan una distribución uniforme o, como se ha señalado antes, cuando se trabaja con algoritmos basados en distancias, como el k-NN.

$$
x_{\text{norm}} = \dfrac{x - \min(x)}{\max(x) - \min(x)}
$$


La **estandarización** convierte los datos a una escala con media 0 y desviación estándar 1. Esto es útil cuando los datos siguen una distribución gaussiana o en métodos como la regresión logística o las redes neuronales profundas.

$$
x_{\text{std}} = \dfrac{x - \mu}{\sigma}
$$

### **Codificación de variables categóricas**
Los modelos de machine learning solo entienden datos numéricos, por lo que las **variables categóricas** deben ser convertidas a números. Existen dos técnicas principales para ello:

- ***One-hot encoding***: Transforma una variable categórica con $n$ categorías en $n$ características binarias. Esta técnica es útil cuando no existe un orden inherente entre las categorías, como en el caso de colores o nombres de ciudades.
- ***Label encoding***: Asigna un número entero a cada categoría. Esto es útil cuando las categorías tienen un orden natural, como "bajo", "medio" y "alto".

### **Ingeniería de características**
En algunos casos, es necesario crear nuevas variables a partir de las existentes para mejorar el rendimiento del modelo. Esto se conoce como **ingeniería de características**. Por ejemplo, en un conjunto de datos de transacciones financieras, se podría crear una nueva variable que represente el promedio de transacciones por día a partir de los datos de las transacciones totales y los días activos.

### **Reducción de dimensionalidad**
Cuando el conjunto de datos contiene muchas variables, algunos algoritmos pueden sufrir un problema denominado  **maldición de la dimensionalidad**, derivado precisamente de la dispersión debida a la gran cantidad de características. Para evitarlo, se puede aplicar técnicas como **PCA (Análisis de Componentes Principales)** o **selección de características**, que permiten reducir el número de variables conservando la mayor parte de la información relevante.

> **Ejemplo:** En un proyecto de predicción de precios de viviendas, es fundamental imputar los valores faltantes en la variable 'tamaño del terreno', ya que muchos registros tienen datos incompletos. Reemplazar estos valores con la media o la mediana puede mejorar la calidad del conjunto de datos.

> **Ejemplo:** En un problema de clasificación de imágenes de ropa, es necesario normalizar los píxeles de las imágenes a un rango de 0 a 1. Esto garantiza que los diferentes colores no afecten de manera desproporcionada a los modelos de clasificación.

> **Ejemplo:** Para un sistema de recomendación de películas, podría aplicarse *one-hot encoding* a las variables categóricas, como el género de la película o el idioma, para que los algoritmos puedan utilizarlas de manera efectiva.

**Para reflexionar...**

> **¿Cómo influye la correcta elección de las técnicas de preprocesamiento en el rendimiento de un modelo de machine learning?**
>
> **Clave**: Piensa en cómo el preprocesamiento afecta la precisión y generalización del modelo, considerando los riesgos de introducir sesgo o perder información valiosa durante esta fase.

## Selección del modelo

La selección del modelo es importante debido a que éste define la estructura y el algoritmo que se utilizarán para transformar los datos de entrada en predicciones, clasificaciones o agrupamientos deseados. La selección adecuada depende de factores como el tipo de problema, la naturaleza de los datos, la capacidad de generalización y la complejidad computacional. Este proceso en su conjunto implica un análisis profundo de las características del problema, así como un balance entre precisión, interpretabilidad y eficiencia computacional. Vamos a ver un poco más en detalle qué consideraciones, al menos en teoría, intervienen en la selección del modelo

### Naturaleza del problema

La elección del modelo se ve fuertemente influenciada por el tipo de problema que se desea resolver. Si es un problema de **regresión** para predecir un valor continuo, como los precios de la vivienda, es recomendable utilizar modelos de **regresión lineal** o **regresión polinómica**. Si es un problema de **clasificación**, como detectar correos spam, modelos como **árboles de decisión**, **máquinas de soporte vectorial (SVM)** o **redes neuronales** son más adecuados.

### Capacidad de generalización

Es fundamental que el modelo no solo se ajuste a los datos de entrenamiento, sino que pueda **generalizar** a datos nuevos. Los modelos demasiado complejos pueden sobreajustarse (capturar el ruido y no los patrones subyacentes), mientras que los modelos simples pueden subajustarse (no captar la complejidad del problema). Aquí entrarían en juego las **técnicas de regularización**, que buscan controlar la complejidad del modelo.

### Cantidad de datos disponibles

Algunos modelos, como las **redes neuronales profundas**, requieren grandes cantidades de datos para ser efectivos. Modelos como los **k-vecinos más cercanos (k-NN)** o los **árboles de decisión** pueden funcionar adecuadamente con conjuntos de datos más pequeños.

### Interpretabilidad vs precisión

Modelos como la **regresión lineal** y los **árboles de decisión** son fáciles de interpretar, lo que los hace útiles en aplicaciones donde la explicabilidad es importante (p. ej., medicina o finanzas). Sin embargo, estos modelos pueden no ser los más precisos para problemas complejos. Por otro lado, las **redes neuronales profundas** pueden ofrecer alta precisión a costa de una menor interpretabilidad.

### Dimensionalidad de los datos

Cuando entran en juego un gran número de características, la **maldición de la dimensionalidad** puede hacer que algunos modelos, como los **k-NN**, sean ineficientes. En estos casos, alternativas como algunos tipos de **SVM** o las **redes neuronales** pueden manejar mejor la alta dimensionalidad al aprender representaciones más abstractas.

### Costo computacional

Los recursos computacionales también influyen. Algoritmos como las **redes neuronales** o algunos tipos de **SVM** requieren una considerable cantidad de tiempo y potencia de cómputo, mientras que otros, como la **regresión lineal** o los **árboles de decisión**, son más eficientes.

> **Ejemplo:** En un proyecto de predicción de precios de viviendas, se opta por un modelo de **regresión lineal** porque la relación entre las características (superficie, número de habitaciones) y el precio es mayormente lineal. Un modelo más complejo, como una red neuronal, podría llevar a sobreajuste dado que el conjunto de datos es relativamente pequeño.

> **Ejemplo:** Para clasificar documentos en categorías temáticas, se utiliza una **Máquina de Soporte Vectorial (SVM)**, ya que puede trabajar bien con alta dimensionalidad (múltiples términos de texto) y es capaz de definir fronteras claras entre diferentes categorías.

> **Ejemplo:** En un sistema de recomendación para un servicio de streaming, se decide utilizar un modelo de **redes neuronales profundas**. Aunque es difícil de interpretar, este modelo tiene la capacidad de captar relaciones complejas entre las características de los usuarios y las preferencias de contenido, ofreciendo recomendaciones personalizadas.

**Para reflexionar...**

> **¿Cómo podemos decidir entre priorizar la interpretabilidad o la precisión en la selección de un modelo de machine learning?**
>
> **Clave:** Reflexiona sobre las implicaciones en áreas críticas como la salud, donde los modelos deben ser explicables, o en áreas comerciales, donde puede ser más importante maximizar la precisión para aumentar el rendimiento.

## Entrenamiento del Modelo

El **entrenamiento del modelo** es el proceso mediante el cual el modelo ajusta sus parámetros para aprender de los datos. Durante esta fase, el algoritmo de machine learning ajusta su comportamiento basándose en los datos de entrenamiento, para luego ser capaz de realizar predicciones o clasificaciones cuando se enfrenta a nuevos datos. Este proceso requiere de una estrategia cuidadosa para evitar problemas como el **sobreajuste** o el **subajuste** y garantizar que el modelo sea capaz de **generalizar** adecuadamente.

### División del dataset: Entrenamiento, Validación y Test

Uno de los pasos iniciales y fundamentales en esta fase es la división del conjunto de datos en **tres subconjuntos** principales: **entrenamiento**, **validación** y **test**. Esta división es crucial para asegurar que el modelo se entrene correctamente y sea evaluado de manera precisa.

1. **Entrenamiento**: Este subconjunto se utiliza para ajustar los parámetros del modelo. El algoritmo iterativamente ajusta los pesos y los coeficientes internos en base a estos datos, buscando minimizar una función de coste que dependerá de la naturaleza del problema en cuestión.  Con este conjunto de datos se realizará la mayor parte del trabajo de aprendizaje.

2. **Validación**: Durante el entrenamiento, es importante verificar cómo se comporta el modelo con un conjunto de datos que no ha visto antes. Esto se realiza con el conjunto de **validación**, que permite ajustar los hiperparámetros del modelo (por ejemplo, el grado de regularización o la tasa de aprendizaje) y evita el sobreajuste. La validación es útil para decidir cuándo detener el entrenamiento si el rendimiento en los datos de validación deja de mejorar, un concepto conocido como ***early stopping***.

3. **Test**: Finalmente, después de haber ajustado el modelo utilizando los datos de entrenamiento y validación, se realiza una evaluación final con el conjunto de **test**. Este conjunto no se utiliza en ninguna parte del proceso de entrenamiento y sirve para medir de manera imparcial la capacidad de generalización del modelo. El rendimiento en este conjunto da una idea realista de cómo se comportará el modelo en situaciones del mundo real.

### Proceso de optimización

Ya hemos comentado en varias ocasiones que el **entrenamiento del modelo** implica la **optimización** de una función de coste, que mide el error del modelo con respecto a los datos de entrenamiento. Uno de los algoritmos de optimización más comunes es el **gradiente descendente**, que ajusta los parámetros del modelo gradualmente para minimizar esta función de error. Existen variantes del gradiente descendente, como el **gradiente descendente estocástico (SGD)**, donde los parámetros se actualizan por cada ejemplo o por pequeños lotes de datos, lo que puede acelerar el proceso de convergencia en grandes conjuntos de datos.

Durante este proceso, el modelo ajusta sus **pesos** en función de cómo afectan a la predicción de la variable dependiente. Este ajuste se realiza utilizando las derivadas parciales de la función de coste respecto a cada parámetro, lo que permite que el algoritmo sepa en qué dirección ajustar los pesos para reducir el error. Esta técnica es especialmente efectiva en **modelos lineales** y **redes neuronales**.

### Evitar el sobreajuste y subajuste

Uno de los desafíos principales durante el entrenamiento es encontrar un equilibrio entre la capacidad del modelo para aprender los patrones en los datos de entrenamiento y su capacidad para generalizar a nuevos datos. Un modelo que se ajuste demasiado bien a los datos de entrenamiento (capturando incluso el ruido) sufrirá de **sobreajuste** y no generalizará bien. Por otro lado, un modelo que no capture suficientes patrones en los datos sufrirá de **subajuste**. Para evitar esto, se utilizan técnicas como la **regularización**, o métodos como **dropout** en en el caso de redes neuronales.

> [!tip]
>
> La **regularización** es una técnica utilizada para evitar el **sobreajuste**, tratando de penalizar modelos demasiado complejos. Introduce un término adicional en la función de coste que restringe los valores de los parámetros, haciendo que el modelo se enfoque en patrones más generales en lugar de los detalles específicos del conjunto de entrenamiento. El **dropout** es una técnica de regularización en redes neuronales que, durante el entrenamiento, apaga aleatoriamente un porcentaje de neuronas en cada capa. Esto previene la **co-adaptación** excesiva entre las neuronas, mejorando la generalización del modelo y reduciendo el riesgo de sobreajuste.

> **Ejemplo:** En un proyecto de predicción de precios de viviendas, se divide el conjunto de datos en un 70% para entrenamiento, un 15% para validación y un 15% para test. El modelo de regresión se entrena iterativamente en el conjunto de entrenamiento, ajustando sus coeficientes para minimizar el error cuadrático medio (MSE). Posteriormente, se utiliza el conjunto de validación para ajustar el parámetro de regularización y mejorar la generalización.

> **Ejemplo:** Para un sistema de reconocimiento facial, se utiliza un algoritmo de red neuronal convolucional (CNN). El conjunto de imágenes se divide en tres partes. Durante el entrenamiento, el modelo ajusta sus parámetros utilizando el gradiente descendente, mientras que se evalúa continuamente su rendimiento en el conjunto de validación para evitar el sobreajuste.

> **Ejemplo:** En un problema de clasificación de correos electrónicos como spam o no spam, se entrena un clasificador con datos etiquetados. Después del entrenamiento inicial, se utiliza el conjunto de validación para ajustar la tasa de aprendizaje y el número de épocas de entrenamiento, garantizando que el modelo no se sobreajuste a los datos de entrenamiento.

**Para reflexionar...**

> **¿Cómo podemos encontrar un equilibrio entre el rendimiento en los datos de entrenamiento y la capacidad de generalización a nuevos datos en un proyecto de machine learning?**
>
> **Clave:** Considera cómo las técnicas de validación cruzada, el ajuste de hiperparámetros y la regularización pueden ayudar a equilibrar el rendimiento del modelo para evitar tanto el sobreajuste como el subajuste.

## Validación del Modelo

En el proyecto de machine learning no sólo es esencial que el modelo ajuste bien los datos de entrenamiento, sino que también pueda hacer predicciones precisas en datos desconocidos. En este sentido, la **validación del modelo** es la etapa clave para asegurar la capacidad del modelo para generalizar sus predicciones a nuevos datos no vistos previamente. Para lograr esto, ya hemos visto que es común dividir el conjunto de datos en una parte para **entrenamiento**, otra para **validación** y una última para **test**.

La validación implica entrenar el modelo con los datos de entrenamiento y luego verificar su rendimiento con los datos de validación. Los modelos se ajustan o afinan en base al rendimiento en el conjunto de validación, lo que ayuda a evitar el **sobreajuste** (cuando el modelo se ajusta demasiado a los datos de entrenamiento y falla en generalizar). Para medir el rendimiento del modelo en esta fase se utilizarán varias **métricas que volverán a utilizarse en fase de evaluación**, como la precisión, el recall, el F1-score, y para modelos de clasificación, y métricas como el **error cuadrático medio** (MSE) o el **error absoluto medio** (MAE) para modelos predictivos

### Métodos de Validación

Con seguridad el **método *Hold-Out*** es una de las técnicas más simples y ampliamente utilizadas para la **validación de modelos de machine learning**. Su propósito, como podemos adivinar, es validar el rendimiento del modelo con datos que no fueron utilizados durante el proceso de entrenamiento, con el fin de estimar cómo generalizará a datos nuevos. Es fácil de implementar y entender y rápido, ya que solo se entrena una vez y se evalúa una vez, lo que ahorra tiempo computacional en comparación con otros métodos. Sin embargo, la calidad del modelo final va a depender fuertemente de la manera en que se realiza la partición de los datos. Una división no representativa puede llevar a resultados engañosos (como siempre, sobreestimando o subestimando el rendimiento del modelo).

Otro de los métodos habituales de validación es la **validación cruzada k-fold**. En este enfoque, el conjunto de datos se divide en *k* partes, donde *k-1* partes se utilizan para el entrenamiento y 1 parte se usa para la validación. Este proceso se repite *k* veces, cambiando la parte de los datos utilizada para la validación en cada iteración. Al final, el rendimiento del modelo se promedia a lo largo de todas las iteraciones, lo que ofrece una estimación más confiable de su capacidad de generalización.

Durante la fase de **validación** no se vuelve a aplicar la **optimización de la función de coste**. Este es un proceso que únicamente se lleva a cabo en la **fase de entrenamiento** del modelo, donde los parámetros del modelo (como los pesos en un modelo polinómico, el sesgo en una red neuronal o los coeficientes en una regresión) se ajustan para minimizar dicha función.

Ahora bien, si se observa que el rendimiento en la validación es deficiente, podría llevarse a cabo un ajuste de **hiperparámetros**, para luego repetir el entrenamiento con esos nuevos hiperparámetros. Este proceso, denominado ***tuning* de hiperparámetros**, no implica en ningún caso una re-optimización directa de la función de coste sobre el conjunto de validación. Los hiperparámetros son los valores que determinan cómo funciona el modelo y no se aprenden directamente del entrenamiento.

Es muy importante entender la diferencia entre **parámetros** e **hiperparámetros** en un modelo, ya que juegan roles complementarios pero distintos en el proceso de aprendizaje. Los **parámetros** son los valores internos que el modelo **aprende automáticamente** a partir de los datos durante el proceso de **entrenamiento**. Estos valores se ajustan para minimizar la **función de coste** y mejorar el rendimiento del modelo. Un parámetro es ajustado directamente por el algoritmo de optimización. En modelos como la **regresión lineal**, los parámetros son los coeficientes de la función lineal (las pendientes o el término independiente), mientras que en las redes neuronales, los parámetros son los pesos de las conexiones entre neuronas. Los **hiperparámetros**, en cambio, **no son aprendidos por el modelo directamente de los datos**. Son **valores externos** que se establecen antes del proceso de entrenamiento y definen la estructura y el comportamiento del modelo. Los hiperparámetros son clave para controlar el proceso de aprendizaje, influenciando cómo se ajustan los parámetros. Ejemplos comunes de hiperparámetros pueden incluir la **tasa de aprendizaje** en el gradiente descendente, que controla el tamaño de los pasos en la optimización; el **número de capas** o **número de neuronas** en cada capa en una red neuronal, o la **profundidad** o **número de árboles** en un modelo de bosque aleatorio. La búsqueda de los mejores hiperparámetros se realiza evaluando el rendimiento del modelo en el conjunto de validación y ajustando dichos valores para maximizar el rendimiento sin incurrir en sobreajuste.

## Evaluación del modelo

En el ciclo de vida de un proyecto de **machine learning**, la **evaluación del modelo** permite medir la calidad de éste, asegurando que generalice bien a nuevos datos y no solo a los datos de entrenamiento o validación. En esta etapa, el objetivo es evaluar el rendimiento del modelo en un conjunto de **datos de prueba (o  *test*)** que el modelo no ha procesado previamente. Esto asegurará que el modelo no se sobreajuste (memorice) los datos de entrenamiento y que sea capaz de hacer buenas predicciones con datos futuros o no vistos.

Un aspecto central en la evaluación del modelo es el uso de **métricas** que cuantifican su rendimiento. Las métricas varían dependiendo del tipo de problema y del tipo de modelo (predicción o clasificación, por ejemplo). Elegir la métrica adecuada es fundamental para obtener una evaluación precisa y útil. Aplicar una métrica consiste básicamente en comparar el *target* de los datos de test con el *target* predicho por el modelo. 

## Métricas comunes en la evaluación de modelos

### Métricas para modelos de clasificación

1. **Exactitud**: Mide el porcentaje de predicciones correctas sobre el total de predicciones realizadas. Es útil cuando las clases están balanceadas, pero puede ser engañosa en casos de clases desbalanceadas, ya que puede parecer que el modelo tiene un buen rendimiento incluso si falla en las clases minoritarias.
2. **Precisión**: Calcula la proporción de verdaderos positivos sobre el total de predicciones positivas. Esta métrica es útil cuando los **falsos positivos** son costosos y se necesita asegurar que, cuando el modelo predice un resultado positivo, lo sea realmente.
3. **Sensibilidad (Recall)**: Mide la proporción de verdaderos positivos sobre todos los casos verdaderamente positivos. Es importante cuando los **falsos negativos** son costosos, por ejemplo, en diagnósticos médicos donde es crucial no pasar por alto casos positivos.
4. **F1-Score**: Es la media armónica entre precisión y sensibilidad. Es útil cuando se necesita un equilibrio entre la precisión y la capacidad del modelo para detectar positivos. Es especialmente importante en problemas con clases desbalanceadas.

### Ejemplo:

> **Ejemplo:** En un sistema de clasificación de correos electrónicos como spam o no spam, la precisión es crucial porque un falso positivo (un correo legítimo clasificado como spam) es más costoso que un falso negativo (un spam que no fue detectado).

### Métricas para modelos de regresión (predicción)

1. **Error Cuadrático Medio (MSE)**: Mide el promedio de los errores al cuadrado entre las predicciones y los valores reales. Penaliza más los errores grandes que los pequeños, por lo que es adecuado en situaciones donde los grandes errores son críticos.

2. **Error Absoluto Medio (MAE)**: Mide el promedio de los errores absolutos. A diferencia del MSE, no da tanto peso a los errores grandes, lo que puede ser útil cuando todos los errores se desean tratar por igual.

3. **R² (Coeficiente de determinación)**: Mide la proporción de la varianza explicada por el modelo respecto a la varianza total. Indica cuán bien las variables independientes explican la variabilidad de la variable dependiente.

> **Ejemplo:** En un modelo de predicción de precios de viviendas, el MSE es útil para penalizar grandes errores, como cuando el modelo predice un precio de vivienda muy bajo o alto en comparación con el valor real.

> **Ejemplo:** Si estamos prediciendo el tiempo que una máquina estará operativa antes de fallar, el **MAE** puede ser más útil que el **MSE**, ya que el error en horas no debe tener un peso excesivo sobre los valores atípicos, y es preferible tratar todos los errores de manera similar.

**Para reflexionar...**

> **¿Cómo afecta la naturaleza del problema (clasificación o regresión) en la elección de las métricas de evaluación?**
>
> **Clave:** Piensa en la importancia de medir diferentes aspectos, como la capacidad de predecir correctamente en problemas de clasificación, frente a medir la precisión de predicción numérica en problemas de regresión.

### Importancia de la evaluación

La evaluación del modelo permite **detectar problemas como el sobreajuste** (cuando el modelo tiene un buen rendimiento en el conjunto de entrenamiento, pero falla con los datos de prueba). Al dividir los datos en conjuntos de entrenamiento, validación y prueba, se garantiza que el modelo se evalúa en datos no vistos, lo que da una buena indicación de cómo se desempeñará en producción.

La evaluación también juega un papel esencial en la **comparación de modelos**. Cuando se prueban múltiples modelos o se ajustan varios hiperparámetros, las métricas de evaluación ayudan a seleccionar el mejor modelo para el problema en cuestión. En este proceso es común realizar una validación cruzada, como la **validación cruzada k-fold**, que divide los datos en diferentes subconjuntos para probar la estabilidad del modelo en diferentes muestras de datos.

### Técnicas avanzadas de evaluación

Además de las métricas básicas, hay técnicas avanzadas de evaluación como la **curva ROC-AUC**, que es útil para evaluar el rendimiento de modelos de clasificación binaria a diferentes umbrales de decisión. La **matriz de confusión** también es una herramienta visual muy utilizada en problemas de clasificación para analizar el rendimiento del modelo en cada clase, permitiendo identificar falsos positivos y falsos negativos.

> **Ejemplo:** En un problema de clasificación de spam, se podría utilizar una **matriz de confusión** para visualizar cuántos correos electrónicos fueron correctamente clasificados como spam o no spam, y cuántos fueron incorrectamente clasificados, para ajustar el modelo si hay un alto número de falsos positivos.

> **Ejemplo:** Al entrenar un modelo de regresión para predecir el precio de automóviles, la métrica **MSE** es útil para medir el error promedio entre los precios predichos y los reales. Un MSE bajo indicará que el modelo tiene un buen rendimiento en este problema de predicción.

> **Ejemplo:** En un modelo de detección de fraude en transacciones bancarias, la métrica de **precisión** es crítica, ya que es importante que todas las predicciones de fraude sean correctas, minimizando los falsos positivos.

**Para reflexionar...**

> **¿Cómo influyen las métricas seleccionadas en la interpretación del rendimiento del modelo?**
>
> **Clave:** Reflexiona sobre cómo algunas métricas pueden proporcionar una imagen incompleta del rendimiento del modelo y cómo elegir las adecuadas según la naturaleza del problema, como el balance de clases o la importancia de falsos positivos o negativos.

## Despliegue del modelo

El **despliegue del modelo** es la fase donde el modelo entrenado y validado se traslada a un entorno de producción para generar predicciones o decisiones sobre datos, ya sea en tiempo real o en procesos por lotes (*batch*). Este paso no debe minusvalorarse porque el modelo comienza a interactuar directamente con usuarios, sistemas o procesos, y debe integrarse adecuadamente en la infraestructura de negocio existente.

Existen varias formas de desplegar un modelo, dependiendo del caso de uso y la arquitectura del sistema. Un método común es la implementación mediante **APIs** (interfaces de programación de aplicaciones), que permite que otros sistemas envíen datos al modelo y reciban predicciones en respuesta. Estas APIs pueden ser utilizadas en aplicaciones web, móviles o sistemas empresariales.

Otra opción es el despliegue del modelo como parte de un **pipeline de datos**, donde el modelo se integra en un flujo de procesamiento de datos que maneja grandes volúmenes de información en tiempo real o en lotes. En este caso, es importante que el modelo sea eficiente y capaz de manejar grandes cantidades de datos sin afectar el rendimiento general del sistema.

El despliegue también implica tomar decisiones sobre **dónde** se ejecutará el modelo. En algunos casos, el modelo puede estar alojado en la nube, aprovechando la escalabilidad y potencia de servicios como AWS, Azure o GCP. En otros casos, el modelo debe desplegarse ***en el borde*** (*edge computing*), especialmente cuando las predicciones deben realizarse rápidamente y cerca de donde los datos se generan. El **edge computing** no es exactamente lo mismo que *desplegar en local*, aunque ambos enfoques comparten algunas similitudes. Cuando desplegamos en local nos referimos a ejecutar el modelo en servidores o infraestructuras locales, como en un centro de datos de la empresa o en dispositivos controlados directamente por la organización. El procesamiento tendrá lugar pues **dentro de la red privada o en las máquinas físicas de la empresa**. Por su parte *despliegue en el borde* hace referencia al hecho de **ejecutar el modelo en dispositivos cercanos al lugar donde se generan los datos**, como **sensores IoT, dispositivos móviles, routers o gateways locales**. El procesamiento se realiza cerca del origen de los datos, lo que permite reducir la latencia y evitar enviar grandes cantidades de datos a la nube o a servidores centrales para su procesamiento. El objetivo del despliegue en el borde es tomar decisiones rápidamente y de manera más eficiente en tiempo real.

Durante el despliegue, se deben tener en cuenta consideraciones técnicas como la **latencia**, que puede ser crítica en aplicaciones de tiempo real (por ejemplo, en sistemas de prevención de fraudes bancarios) o la **escalabilidad**, cuando se trata de servir a miles o millones de usuarios simultáneamente.

El despliegue también requiere la creación de un entorno que permita la **reproducción** del modelo entrenado. Esto incluye la gestión de dependencias, como bibliotecas de software y versiones de hardware o software, para garantizar que el modelo funcione de manera consistente. En muchos casos, se utilizan software contenedores como **Docker** para encapsular el modelo y todas sus dependencias, facilitando su despliegue en diferentes entornos sin problemas de compatibilidad.

> **Ejemplo:** Un modelo de reconocimiento de voz implementado en un asistente virtual utiliza una API para convertir la voz del usuario en texto. Esta API se despliega en la nube, lo que permite que el sistema procese las solicitudes de miles de usuarios simultáneamente, devolviendo respuestas en tiempo real.

> **Ejemplo:** En una plataforma de comercio electrónico, un modelo de recomendación de productos se despliega directamente en los servidores que gestionan el sitio web. El modelo procesa los datos de navegación de los usuarios y actualiza las recomendaciones de productos en tiempo real mientras el usuario navega por la web.

> **Ejemplo:** Un modelo para optimizar rutas de entrega en una empresa de logística se despliega en la nube. Los datos de los pedidos y las ubicaciones de los vehículos se envían al modelo, que genera rutas óptimas que son utilizadas por los conductores en sus dispositivos móviles.

**Para reflexionar...**

> **¿Cómo afecta la ubicación del despliegue (nube, edge, servidores locales) al rendimiento y la capacidad de respuesta del modelo en un entorno de producción?**
>
> **Clave:** Considera factores como la latencia, los costos asociados, la disponibilidad de recursos computacionales y la necesidad de tomar decisiones en tiempo real.

## Monitoreo y mantenimiento del modelo

El **monitoreo y mantenimiento de un modelo de machine learning** asegura que el modelo continúe funcionando de manera efectiva después de su despliegue. Una vez que un modelo ha sido implementado en producción, es esencial realizar un seguimiento constante de su desempeño para detectar cualquier degradación en su rendimiento y tomar medidas correctivas.

Una de las razones principales para el monitoreo es que los **datos del entorno de producción pueden cambiar** con el tiempo, fenómeno conocido como ***drift* de datos o de concepto**. Este cambio puede afectar la precisión del modelo, lo que significa que un modelo que inicialmente fue muy efectivo podría volverse menos preciso con el tiempo. Por ello, el monitoreo de métricas clave, como la precisión, el recall, el F1-Score, el error cuadrático medio (MSE) o el área bajo la curva ROC, es fundamental para garantizar la eficacia del modelo a lo largo del tiempo.

El mantenimiento del modelo también incluye procesos de **reentrenamiento** cuando el rendimiento del modelo disminuye. Esto implica volver a entrenar el modelo con nuevos datos, adaptándolo a las condiciones actuales. Este proceso puede ser **automático** (mediante la implementación de pipelines automatizados de machine learning) o **manual**, dependiendo de los requerimientos del sistema y los recursos disponibles.

También es importante tener en cuenta el **registro de datos** y el seguimiento de cambios. Se debe llevar un control detallado de todas las versiones del modelo, datos utilizados y modificaciones realizadas, lo cual es esencial no solo para la gestión del rendimiento, sino también para cumplir con las normativas de auditoría y trazabilidad.

> [!important]
>
> **Elementos clave en el monitoreo del modelo**
>
> - **Desempeño del modelo**: Monitorear la exactitud y otras métricas relevantes en intervalos regulares.
> - **Deriva de datos (data drift)**: Evaluar si los datos de entrada cambian con el tiempo, lo que puede afectar la efectividad del modelo.
> - **Disponibilidad y latencia**: Garantizar que el modelo siga respondiendo en el tiempo requerido.
> - **Mantenimiento proactivo**: Establecer un plan de reentrenamiento o ajuste periódico.



> **Ejemplo:** En un sistema de predicción de demanda de productos, el modelo puede haber sido entrenado con patrones estacionales. Si las tendencias de compra cambian debido a eventos imprevistos, como la pandemia de COVID-19, el modelo deberá ajustarse para reflejar las nuevas tendencias.

> **Ejemplo:** Un sistema de detección de fraudes en una plataforma de pagos puede experimentar una caída en su precisión debido a cambios en el comportamiento de los usuarios o nuevas tácticas de fraude. Monitorear las tasas de falsos positivos y falsos negativos es esencial para activar un proceso de reentrenamiento cuando sea necesario.

> **Ejemplo:** En un modelo de recomendación de contenidos en una plataforma de streaming, el modelo puede volverse obsoleto si no se actualiza con nuevas preferencias del usuario. El monitoreo continuo del rendimiento asegura que las recomendaciones sigan siendo relevantes.

**Para reflexionar...**

> **¿Cómo se puede garantizar que un modelo de machine learning siga siendo eficaz con el paso del tiempo, a pesar de los cambios en los datos y el entorno de producción?**
>
> **Clave:** Reflexiona sobre la importancia del monitoreo continuo, la identificación del *data drift* y la planificación de un proceso de reentrenamiento regular para mantener la precisión del modelo.

> ##### ¿Qué riesgos implica implementar un modelo en producción sin monitoreo adecuado?
> **Pistas**: Considera cómo los cambios en los datos pueden degradar el rendimiento del modelo con el tiempo. Piensa en problemas como la **deriva de datos** o la aparición de nuevos patrones que no se reflejan en el entrenamiento inicial.

> ##### ¿Es siempre necesario reentrenar un modelo en producción cuando su rendimiento comienza a disminuir?
> **Pistas**: Reflexiona sobre cómo pequeñas adaptaciones o ajustes en los hiperparámetros, junto con la recolección de nuevos datos, pueden mejorar el rendimiento sin necesidad de un reentrenamiento completo.

