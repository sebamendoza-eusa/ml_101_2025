# 6. Codificar la información

## Introducción a la codificación de datos

En el campo del **machine learning** y la **inteligencia artificial**, los modelos procesan y aprenden a partir de datos numéricos. Sin embargo, no todos los datos del mundo real llegan en un formato adecuado para ser utilizados directamente por los algoritmos. Muchos tipos de información, como texto, imágenes, sonidos o categorías, deben transformarse o **codificarse** para que los modelos puedan extraer patrones y aprender de ellos. Este proceso es fundamental para la **preparación de los datos** y puede impactar directamente en la capacidad del modelo para generalizar correctamente.

**La codificación de datos tiene como objetivo convertir información cualitativa o categórica en valores cuantitativos. De esta manera, el modelo puede procesar y relacionar las entradas con las salidas de manera efectiva**. En muchos casos, la forma en que se realiza esta codificación afecta el rendimiento y la precisión del modelo, ya que los algoritmos de machine learning se comportan de manera diferente según el tipo de datos que reciben.

### Importancia de la codificación

La codificación es especialmente importante en proyectos de **aprendizaje supervisado** y **no supervisado** donde los algoritmos necesitan interpretar datos heterogéneos. Para los modelos de regresión y clasificación, por ejemplo, es fundamental transformar las variables categóricas y textuales en un formato que el algoritmo pueda procesar y analizar. Sin una codificación adecuada, el modelo no sería capaz de identificar relaciones significativas, lo que afectaría la calidad de las predicciones.

Además, la codificación de datos es esencial en el caso de las redes neuronales y los modelos de aprendizaje profundo. Estos modelos dependen de los valores numéricos para entrenarse y aprender patrones a través de capas de neuronas. Las variables no numéricas, como el texto o las imágenes, requieren técnicas avanzadas de codificación para que el modelo pueda procesarlas adecuadamente.

### Tipos de datos y necesidades de codificación

Diferentes tipos de datos requieren diferentes enfoques y técnicas en cuanto a su codificación. Entre los escenarios más comunes se encuentran:

- **Variables categóricas**: Estas son variables que representan categorías discretas y no tienen un orden inherente. Se suelen convertir en valores numéricos utilizando técnicas como **label encoding** o **one-hot encoding**. Estas transformaciones permiten que los modelos interpreten las categorías de manera cuantitativa.
  
- **Datos ordinales**: Variables que poseen un orden jerárquico, pero no una magnitud numérica definida entre los niveles. La **codificación ordinal** es ideal en estos casos, ya que mantiene la estructura de orden en los datos.

- **Datos textuales**: Las palabras y frases deben ser transformadas en vectores numéricos para que los modelos de machine learning puedan analizarlas. Métodos como **bag-of-words**, **TF-IDF** o **embeddings** son populares en la codificación de datos textuales.

- **Datos continuos**: Estos datos ya están en un formato numérico, pero pueden requerir **normalización** o **escalado** para asegurarse de que las diferentes características tienen magnitudes comparables.

### Desafíos en la codificación

La codificación de datos no es un proceso trivial y presenta varios desafíos. Uno de ellos es **el sesgo**, especialmente cuando se trata de datos categóricos con muchos niveles o categorías. Elegir la técnica de codificación incorrecta puede llevar a un modelo ineficaz o a introducir sesgos en las predicciones. Otro desafío importante es la **dimensionalidad**. En técnicas como **one-hot encoding**, donde cada categoría se convierte en una columna separada, la dimensionalidad del conjunto de datos puede aumentar drásticamente, lo que puede causar problemas de **sobreajuste** o dificultar la interpretación del modelo.

Además, la **información implícita** que contienen los datos puede perderse durante la codificación. Un ejemplo clásico es cuando se utiliza label encoding para convertir categorías en números. Si las categorías no tienen una relación inherente entre ellas, el modelo puede asumir incorrectamente que una categoría es mayor que otra debido a su valor numérico asignado.

Por último, es fundamental tener en cuenta el **contexto de los datos** y los objetivos del proyecto a la hora de seleccionar el método de codificación. La misma técnica no siempre es adecuada para todos los casos, y a menudo es necesario realizar pruebas y validaciones para encontrar la mejor opción para un proyecto en particular.

> **Ejemplo:** En un proyecto de predicción de precios de vivienda, las variables categóricas como el tipo de vivienda o la localización geográfica pueden ser codificadas utilizando **one-hot encoding** para que el modelo interprete estas categorías correctamente sin asumir un orden entre ellas.

> **Ejemplo:** Para un sistema de procesamiento de lenguaje natural, las palabras en un texto pueden ser representadas usando **word embeddings** como Word2Vec o GloVe, que capturan la relación semántica entre diferentes palabras, proporcionando al modelo una representación numérica útil.

> **Ejemplo:** En un proyecto de clasificación de clientes para un banco, la variable "estado civil" se puede codificar usando **label encoding**. Esto asigna a cada categoría (soltero, casado, etc.) un número diferente, que luego será procesado por el modelo.

**Para reflexionar...**

> **¿Cómo puede la codificación de datos influir en el rendimiento y la precisión de un modelo de machine learning?**
>
> **Clave:** Piensa en cómo la elección de la técnica de codificación afecta la representación de la información para el modelo, el posible sesgo introducido y la magnitud de los datos

## Codificación de variables categóricas

Las **variables categóricas** son aquellas que contienen un conjunto limitado de valores discretos y que, en muchos casos, no tienen una relación numérica natural entre ellas. Ejemplos comunes incluyen características como el **estado civil**, **tipo de vivienda**, **categorías de productos**, o el **género**. En **machine learning**, los modelos no pueden procesar estas variables categóricas directamente, ya que trabajan con datos numéricos. Por tanto, es esencial transformar estas variables mediante técnicas de **codificación** para que los modelos puedan aprender a partir de ellas.

### Tipos de codificación de variables categóricas

Existen varias formas de codificar variables categóricas. La selección de la técnica depende del tipo de variable, la naturaleza del problema, y el modelo que se esté utilizando. A continuación, se detallan las técnicas más comunes:

#### Label Encoding

La **codificación de etiquetas** (label encoding) es una técnica sencilla que asigna un valor numérico a cada categoría. Si se tiene una variable con los valores {‘bajo’, ‘medio’, ‘alto’}, esta técnica asignaría 0 a ‘bajo’, 1 a ‘medio’, y 2 a ‘alto’. Aunque es fácil de implementar, tiene desventajas importantes. La principal es que introduce un **orden implícito** entre las categorías que no siempre es real o deseable. Algunos modelos pueden interpretar que el valor ‘alto’ es mayor que ‘medio’ de una manera que no tiene sentido en el contexto de la variable.

> **Ejemplo:** En un conjunto de datos que contiene la variable ‘color’ con los valores {‘rojo’, ‘azul’, ‘verde’}, el label encoding podría asignar 0 a ‘rojo’, 1 a ‘azul’, y 2 a ‘verde’. Sin embargo, no hay una relación numérica real entre estos colores, lo que podría inducir un sesgo en el modelo.

#### One-Hot Encoding

La **codificación one-hot** es una técnica que evita los problemas de orden implícito de label encoding al crear una columna separada para cada categoría. Cada categoría se convierte en un vector binario en el que solo un valor es 1 y el resto son 0. Esto permite que el modelo trate las categorías como independientes, sin introducir relaciones no deseadas.

Sin embargo, una desventaja de one-hot encoding es que aumenta significativamente la **dimensionalidad** del conjunto de datos, especialmente cuando se trabaja con variables que tienen muchas categorías. Esto puede llevar a problemas de **escasez de datos** y un aumento en los tiempos de entrenamiento.

> **Ejemplo:** Si tenemos la variable ‘género’ con las categorías {‘hombre’, ‘mujer’, ‘no binario’}, one-hot encoding transformaría estas categorías en tres columnas: {‘hombre’ = [1, 0, 0], ‘mujer’ = [0, 1, 0], ‘no binario’ = [0, 0, 1]}. El modelo interpretaría cada categoría de forma independiente, sin asumir ningún orden entre ellas.

#### Ordinal Encoding

La **codificación ordinal** es apropiada para variables categóricas que tienen un orden lógico, como las escalas de satisfacción (bajo, medio, alto) o niveles educativos (primaria, secundaria, universitaria). En este caso, se puede utilizar una codificación numérica en la que cada categoría tiene un valor asignado que refleja su posición en el orden. A diferencia de label encoding, en ordinal encoding el orden sí tiene sentido, ya que hay una **jerarquía natural** en las categorías.

> **Ejemplo:** Si una variable representa el nivel de satisfacción de un cliente, con los valores {‘muy insatisfecho’, ‘insatisfecho’, ‘neutral’, ‘satisfecho’, ‘muy satisfecho’}, ordinal encoding asignaría números crecientes, como 0 a ‘muy insatisfecho’, 1 a ‘insatisfecho’, 2 a ‘neutral’, 3 a ‘satisfecho’, y 4 a ‘muy satisfecho’. En este caso, el orden es importante, ya que refleja el nivel de satisfacción.

Su uso es relevante en problemas donde las diferencias entre los valores codificados tienen un impacto directo en la predicción o clasificación. Sin embargo, uno de los riesgos de la codificación ordinal es que puede introducir **sesgos** si las diferencias entre los valores no reflejan adecuadamente las distancias reales entre las categorías.

Por ejemplo, si codificamos las categorías de satisfacción del cliente {‘muy insatisfecho’, ‘insatisfecho’, ‘neutral’, ‘satisfecho’, ‘muy satisfecho’} como 0, 1, 2, 3 y 4, estamos asumiendo que las diferencias entre cada nivel de satisfacción son **iguales**. En la práctica, la distancia entre ‘muy insatisfecho’ y ‘neutral’ podría ser mayor que entre ‘satisfecho’ y ‘muy satisfecho’. Si el modelo no tiene en cuenta estas posibles **disparidades**, los resultados pueden no ser óptimos.

Como se ha dicho, comparada con el **label encoding**, la codificación ordinal tiene la ventaja de **preservar el orden jerárquico** en las categorías, lo que puede ser beneficioso en problemas donde el orden es significativo. En cambio, en situaciones donde no hay un orden intrínseco, el uso de técnicas como **one-hot encoding** o **target encoding** sería más adecuado, ya que evitan imponer una estructura ordinal que no tiene sentido.

Como también hemos comentado, uno de los principales **desafíos** de la codificación ordinal es que **asume una igualdad en las distancias entre categorías**, algo que no siempre está de acorde a la realidad del problema. De hecho ello puede llevar a **errores en la interpretación** del modelo, sobre todo en algoritmos sensibles a las magnitudes de los valores, como los modelos de regresión lineal. Además, la codificación ordinal es propensa a errores en problemas donde las categorías no tienen una estructura de orden clara, lo que puede introducir **sesgos** si se aplica incorrectamente. También es importante considerar la naturaleza del **modelo** que se está utilizando. Algoritmos lineales, como la **regresión lineal** o los **modelos lineales generalizados**, pueden interpretar los valores ordinales de manera más adecuada, pero en **modelos basados en árboles**, como **random forests**, donde el enfoque es más local, el orden puede no ser tan significativo. Por lo tanto, la selección del método de codificación debe estar alineada con el tipo de modelo y la estructura de los datos.

#### Target Encoding

El **target encoding** es una técnica de codificación de variables categóricas en la que cada categoría se reemplaza por un valor basado en la **variable objetivo** (target) **en un problema de machine learning supervisado**. En lugar de asignar un valor arbitrario a cada categoría, como en el caso de **one-hot encoding** o **label encoding**, el target encoding utiliza **el promedio de la variable objetivo para cada categoría**, capturando la relación entre las categorías y la variable que se desea predecir. Esto permite que la codificación esté directamente relacionada con el comportamiento de la variable objetivo, lo que puede mejorar la capacidad predictiva del modelo.

Por ejemplo, si estamos intentando predecir si un cliente comprará o no un producto (con un target binario: 0 o 1), podemos calcular el porcentaje de compras (1s) en cada categoría de la variable "fuente de tráfico" (como "orgánico", "pago", "referido"). Luego, cada valor categórico se codifica con ese porcentaje.

Esta técnica, como todas, presenta ventajas e inconvenientes dependiendo del uso que vayamos a darle. Como ventajas podemos argumentar, en primer lugar que **captura relaciones complejas**, En efecto, el target encoding permite que el modelo capture relaciones entre las categorías y la variable objetivo, que podrían ser difíciles de detectar con técnicas de codificación más simples. También es una técnica que **reduce la dimensionalidad**. A diferencia del one-hot encoding, que crea una columna para cada categoría, target encoding reemplaza todas las categorías por un solo valor, lo que reduce significativamente el número de variables en el modelo. De esto último se desprende que en problemas con **alta cardinalidad** esta técnica mejora mucho el rendimiento.

Por contra, también el target encoding tiene sus riesgos. En primer lugar hemos de citar el efecto **fuga de datos (data leakage)**, que se produce  cuando la información del conjunto de validación o prueba se utiliza para calcular los promedios durante el entrenamiento. Esto puede ocurrir si las categorías captan información de la variable objetivo que no deberían ver antes de hacer predicciones, lo que puede llevar a un sobreajuste. Por otra parte, esta técnica también puede producir **sobreajuste en conjuntos pequeños**, ya que los promedios calculados para esas categorías no generalizarán bien.

Para evitar estos problemas, pueden usarse técnicas como la **suavización de la codificación** (smoothing), en la que se ajustan los promedios categóricos utilizando **una combinación del promedio global de la variable objetivo y el promedio específico de la categoría**, con un mayor peso dado al promedio global cuando hay pocas observaciones para una categoría específica.

> **Ejemplo:** Imagina que estás intentando predecir la probabilidad de que un cliente compre un producto (0 o 1) en función de su país de residencia. Para cada país, calculas el promedio de compras realizadas (porcentaje de "1s") y utilizas este valor para codificar el país. Si el país "EE.UU." tiene un promedio de compras de 0.6 y el país "España" tiene un promedio de 0.3, reemplazas "EE.UU." por 0.6 y "España" por 0.3.

#### Codificación basada en frecuencias

De modo parecido al target encoding, la **codificación basada en frecuencias** es una técnica para transformar variables categóricas en valores numéricos, asignando a cada categoría un valor basado en la **frecuencia** de su aparición en el conjunto de datos. Este método es particularmente útil cuando se trabaja con grandes volúmenes de datos y una alta cardinalidad en las variables categóricas, ya que reduce significativamente el número de características, permitiendo una mayor eficiencia en el proceso de entrenamiento de los modelos de machine learning.

A diferencia de técnicas como el **one-hot encoding**, que crea una nueva columna para cada categoría, la codificación basada en frecuencias utiliza una sola columna numérica, lo que reduce la dimensionalidad del conjunto de datos. Cada categoría en una variable categórica se reemplaza por el porcentaje o el número absoluto de veces que aparece en el conjunto de datos.

La codificación basada en frecuencias es ideal para escenarios donde los valores categóricos tienen una **distribución desigual** y donde la aparición de ciertas categorías es más relevante para el problema que otras. Al usar las frecuencias, el modelo puede captar la importancia de cada categoría con base en su prevalencia en el conjunto de datos.

Como se ha comentado, esta técnica también es adecuada cuando se busca un enfoque más simple para manejar categorías de alta cardinalidad, donde el **one-hot encoding** generaría una explosión de dimensiones, complicando el entrenamiento del modelo. Sin embargo, es importante recordar que la codificación basada en frecuencias puede no ser adecuada para todos los problemas, ya que en algunos casos las frecuencias pueden no estar relacionadas con la variable objetivo, lo que podría introducir **sesgos**.

En comparación con el **target encoding**, donde los valores categóricos se reemplazan por el promedio de la variable objetivo para esa categoría, la codificación basada en frecuencias es menos propensa a **fugas de datos**, ya que no incorpora información de la variable objetivo en la transformación, lo que puede ser una ventaja en algunos casos. No obstante, la codificación basada en frecuencias puede no capturar relaciones profundas entre las categorías y la variable objetivo, lo que limita su aplicabilidad en problemas más complejos.

La codificación basada en frecuencias tiene por supuesto sus limitaciones. Uno de los principales desafíos es que puede no ser la mejor opción en problemas donde todas las categorías tienen una **distribución uniforme** o donde las frecuencias no son relevantes para la predicción. En esos casos, la técnica ya no solo podría no añadir valor, sino que además podría deteriorar el rendimiento del modelo. Por otro lado hay que tener muy en cuenta que, en casos donde las categorías tienen una **distribución sesgada**, las categorías con frecuencias más altas dominarán la codificación, lo que puede llevar a que el modelo favorezca ciertas categorías de manera injustificada. Para mitigar este riesgo, se puede combinar la codificación basada en frecuencias con técnicas de normalización o escalado, asegurando que los valores no distorsionen la importancia de las categorías.

> **Ejemplo:** En un conjunto de datos sobre transacciones en línea, la variable ‘método de pago’ podría ser codificada usando la frecuencia de cada método. Si el método ‘tarjeta de crédito’ aparece en el 60% de las transacciones, se le asignaría un valor de 0.6, mientras que al método ‘Paypal’, que aparece en el 30% de las transacciones, se le asignaría un valor de 0.3.

> **Ejemplo:** En un análisis de compras de supermercado, si la variable ‘categoría del producto’ tiene una alta cardinalidad (por ejemplo, cientos de categorías de productos), la codificación basada en frecuencias podría asignar un valor de acuerdo con la cantidad de veces que cada categoría de producto fue comprada. Así, las categorías de productos más comprados tendrían un valor mayor, mientras que los productos menos frecuentes tendrían valores más bajos.

> **Ejemplo:** En un análisis de tráfico web, la variable ‘fuente de tráfico’ (directo, orgánico, pagado) podría ser codificada por frecuencias para reflejar cuántas veces cada fuente ha traído visitantes al sitio. Esto permite reducir la dimensionalidad de los datos sin perder la relevancia de las diferentes fuentes de tráfico en la predicción de conversiones.

**Para reflexionar...**

> **¿Cuáles son las ventajas y riesgos de usar codificación basada en frecuencias en lugar de otras técnicas como one-hot encoding o target encoding?**
>
> **Clave:** Reflexiona sobre cómo las frecuencias pueden preservar la estructura del conjunto de datos y reducir la dimensionalidad, pero también considera los riesgos de sesgos y pérdida de información al no capturar todas las relaciones con la variable objetivo.

### Codificación binaria

La **codificación binaria** es una técnica utilizada para convertir variables categóricas en una representación numérica que pueda ser procesada por algoritmos de machine learning. En lugar de crear una columna por cada categoría, como ocurre en el *one-hot encoding*, la codificación binaria transforma las categorías en secuencias de bits, lo que permite una representación más compacta y eficiente en términos de espacio.

Este enfoque es particularmente útil cuando se trabaja con variables que contienen un gran número de categorías, ya que reduce la cantidad de columnas necesarias, minimizando así el riesgo de sobreajuste y mejorando la capacidad de generalización del modelo. Además, al utilizar una menor cantidad de columnas, la codificación binaria evita problemas comunes como la multicolinealidad, lo que facilita la interpretación del modelo.

> [!tip]
>
> La **multicolinealidad** es una situación en la que dos o más variables independientes en un modelo de regresión están altamente correlacionadas entre sí. Esto puede dificultar la estimación precisa de los coeficientes de las variables, ya que el modelo no puede distinguir de manera efectiva el impacto individual de cada variable sobre la variable dependiente. La multicolinealidad puede aumentar la varianza de las estimaciones y reducir la interpretabilidad del modelo.

El proceso de codificación binaria comienza asignando un valor numérico a cada categoría de la variable. Este valor se convierte en su equivalente binario y luego se separa en varios bits, con cada bit representado en una nueva columna.

> **Ejemplo:** Imagina que tenemos una variable categórica llamada "Tipo de vehículo" con las categorías: "Coche", "Moto", "Camión" y "Autobús". Si utilizáramos la codificación binaria, cada categoría se convertiría en un número binario. A continuación, se muestran los pasos:
>
> 1. **Asignación de valores numéricos**:  
>    - Coche = 1  
>    - Moto = 2  
>    - Camión = 3  
>    - Autobús = 4  
>
> 2. **Conversión a binario**:
>    - Coche (1) = 01  
>    - Moto (2) = 10  
>    - Camión (3) = 11  
>    - Autobús (4) = 100  
>
> 3. **Codificación binaria en columnas**: El valor binario resultante se divide en varias columnas, una por cada bit.
>   
> | Tipo de vehículo | Columna 1 | Columna 2 | Columna 3 |
> | ---------------- | --------- | --------- | --------- |
> | Coche            | 0         | 1         | 0         |
> | Moto             | 1         | 0         | 0         |
> | Camión           | 1         | 1         | 0         |
> | Autobús          | 1         | 0         | 1         |
>
> En este ejemplo práctico, hemos representado las categorías "Coche", "Moto", "Camión" y "Autobús" utilizando tres columnas binarias, en lugar de crear una columna por cada categoría como haríamos en una codificación *one-hot*.

Una de las principales ventajas de la codificación binaria es que optimiza el uso de memoria, especialmente cuando se manejan grandes cantidades de categorías. Esto resulta en conjuntos de datos menos dimensionales y facilita el entrenamiento de modelos, como árboles de decisión o redes neuronales. Al reducir la cantidad de columnas, el modelo puede evitar el sobreajuste y manejar mejor los datos con menos redundancia. Por otro lado, la codificación binaria resulta particularmente eficaz cuando se trabaja con variables categóricas que no tienen un orden intrínseco. En este contexto, la reducción de dimensionalidad y el uso eficiente de las columnas pueden facilitar significativamente el procesamiento de grandes volúmenes de datos, mejorando tanto la precisión del modelo como su tiempo de entrenamiento.

A pesar de sus ventajas, la codificación binaria presenta algunos desafíos. Uno de los principales es la interpretación de los datos resultantes, ya que al comprimir la información en secuencias binarias, se puede perder parte del significado original de las categorías. Esto puede dificultar la capacidad del modelo para identificar patrones claros entre las diferentes categorías. Otro problema es que la codificación binaria no es capaz de capturar relaciones jerárquicas entre las categorías, lo que puede limitar su aplicabilidad en ciertos casos.

Además, la codificación binaria depende completamente del conjunto de datos con el que se entrene el modelo. Si el conjunto de datos está sesgado o es limitado, las representaciones binarias también reflejarán esos sesgos, lo que puede resultar problemático en aplicaciones sensibles como el procesamiento de datos médicos o legales.

> **Ejemplo**: En el análisis de géneros musicales, donde hay más de 100 categorías, la codificación binaria puede reducir la representación de estos géneros a solo unas pocas columnas, disminuyendo el tamaño del conjunto de datos sin perder la capacidad de clasificar correctamente los géneros.

> **Ejemplo**: En un proyecto de clasificación de productos en una tienda en línea con miles de categorías, la codificación binaria puede representar eficientemente todas esas categorías en un número significativamente menor de columnas, facilitando así el entrenamiento de los modelos.

> **Ejemplo**: En el análisis de diagnósticos médicos, donde se manejan miles de categorías de enfermedades, la codificación binaria permite representar cada diagnóstico de manera más eficiente, lo que facilita el procesamiento de los datos y mejora el rendimiento de los modelos.

**Para reflexionar...**

> **¿Qué tipos de problemas crees que se beneficiarían más de la codificación binaria, en comparación con otras técnicas de codificación?**
>
> **Pistas**: Considera cómo la codificación binaria puede reducir la dimensionalidad en conjuntos de datos con muchas categorías y cómo puede afectar a la precisión del modelo en ciertos contextos.

### Consideraciones finales a tener en cuenta en la codificación de variables categóricas

Una de las decisiones clave al trabajar con variables categóricas es seleccionar el método de codificación más apropiado para el problema y el modelo. Técnicas como one-hot encoding son ideales para **modelos lineales** o cuando no existe una relación natural entre las categorías, mientras que ordinal encoding es más adecuado cuando hay una jerarquía evidente.

También es fundamental **prestar atención al sesgo** introducido por la codificación. El label encoding puede ser útil para variables con un orden claro, pero en los casos en que no lo hay, podría llevar a que el modelo asuma relaciones inexistentes entre las categorías. 

Finalmente, es importante tener en cuenta la **dimensionalidad**. Cuando se trabaja con un número muy alto de categorías, técnicas como one-hot encoding pueden inflar la dimensionalidad del conjunto de datos y provocar problemas de rendimiento en los modelos. En estos casos, otras técnicas como la codificación basada en frecuencias pueden ser más adecuadas para reducir la complejidad sin perder información.

> **Ejemplo:** En un proyecto de análisis de encuestas, las respuestas de los participantes a preguntas de opción múltiple pueden ser transformadas utilizando **ordinal encoding** para que el modelo pueda aprender a partir de ellas, preservando el orden de preferencia en las respuestas.

> **Ejemplo:** En una tienda en línea, se utiliza **one-hot encoding** para la variable ‘categoría del producto’, asegurando que el modelo no introduzca sesgos innecesarios al asumir un orden entre las categorías, como ‘ropa’, ‘electrónica’ o ‘libros’.

> **Ejemplo:** En un análisis de riesgo de crédito, las variables categóricas como el tipo de empleo pueden ser codificadas utilizando **frecuencia**, asignando valores numéricos basados en cuántas veces cada categoría aparece en el conjunto de datos.

**Para reflexionar...**

> **¿Cómo influye la elección de la técnica de codificación en el rendimiento de los modelos de machine learning, y qué sesgos pueden introducirse?**
>
> **Clave:** Piensa en cómo la elección del método de codificación afecta a la representación de las categorías, el aumento de la dimensionalidad, y la posibilidad de introducir ordenaciones no deseadas en los datos.

## Codificación de imágenes

La **codificación de imágenes** en proyectos de inteligencia artificial es fundamental para que los modelos puedan interpretar visualmente la información. A diferencia de los datos tabulares o textuales, las imágenes están formadas por una enorme cantidad de información en bruto que debe ser procesada y transformada en representaciones numéricas manejables. Este proceso de **codificación** es esencial para reducir la dimensionalidad de las imágenes y extraer características relevantes que los modelos puedan utilizar para tomar decisiones. En esta sección, abordaremos las técnicas más comunes de codificación de imágenes, con un enfoque basado en la **reducción dimensional** y la **representación de características**.

### Punto de partida

Cada imagen digital está compuesta por una matriz de píxeles, donde cada píxel representa un valor numérico asociado con la intensidad de luz y, en imágenes en color, con combinaciones de colores primarios (RGB). Para que un modelo de aprendizaje automático procese una imagen, es necesario convertir esta matriz de píxeles en una representación numérica que sea interpretable por el algoritmo. Podría pensarse que la matriz numérica pixel a pixel podría ser una buena idea, sin embargo, esta matriz suele ser muy grande, especialmente en imágenes de alta resolución. Por tanto el primer reto que se plantea en la codificación de imágenes es el de gestionar la **alta dimensionalidad** de los datos.

> **Ejemplo**: Una imagen en color de 256x256 píxeles tiene 196,608 valores (256 x 256 x 3). Este volumen de datos puede ser ineficiente y costoso de procesar para muchos modelos, por lo que se hace necesario aplicar técnicas de **reducción de dimensionalidad** para simplificar la representación de la imagen sin perder las características esenciales que influyen en la tarea a resolver, como clasificación, detección de objetos o segmentación de imágenes.

### Reducción dimensional

La **reducción de dimensionalidad** es una técnica utilizada para disminuir la cantidad de información de una imagen mientras se mantienen las características más importantes. En el contexto de imágenes, se busca encontrar representaciones más compactas que retengan las propiedades visuales significativas para el aprendizaje de los modelos. A continuación, se describen dos de las técnicas más comunes para la reducción dimensional en imágenes: El análisis de componentes principales y los *autoencoders*

#### Análisis de componentes principales (PCA)

El **Análisis de Componentes Principales (PCA)** es una técnica estadística que transforma los datos originales en conjunto de variables menor no correlacionadas entre sí. Este nuevo conjunto de variables llamadas **componentes principales**, son combinaciones lineales de las variables originales. Estas componentes están ordenadas según la cantidad de varianza (información) que explican en los datos. La primera componente principal captura la mayor varianza, la segunda la siguiente mayor cantidad, y así sucesivamente. El objetivo es reducir el número de variables mientras se conserva la estructura esencial de los datos, lo que es útil para simplificar el modelo, reducir el ruido y mejorar la eficiencia.

En la práctica, PCA puede ser útil cuando se trabaja con grandes conjuntos de imágenes donde es necesario simplificar la información. Por ejemplo, en la clasificación de imágenes médicas, PCA puede ayudar a reducir el tamaño de las imágenes manteniendo las características que son cruciales para el diagnóstico.

> [!note]
>
> Imagina que tienes un conjunto de datos que contiene las **alturas** y **pesos** de un grupo de personas. Estos datos forman un gráfico en dos dimensiones, donde cada punto representa a una persona. Al observar los datos, puedes notar una relación entre la altura y el peso: a mayor altura, mayor peso, aunque no es una relación perfecta.
>
> Aplicando **PCA** a estos datos, podrías encontrar que la mayoría de la variabilidad en el conjunto de datos (la información más relevante) se puede capturar en una única línea diagonal que atraviesa el gráfico. Esta línea representa la **componente principal** o **primer componente**, que refleja la tendencia general de que las personas más altas tienden a pesar más. El segundo componente capturaría cualquier variación adicional que no esté en esa tendencia principal, como ligeras diferencias en altura y peso entre individuos de una misma altura.
>
> En este caso, **PCA** permite transformar los dos ejes originales (altura y peso) en dos **nuevas variables**: una que captura la mayor parte de la variación conjunta (altura y peso combinados) y otra que contiene variaciones más pequeñas. Al reducir el número de variables (pasando de dos a una), se simplifica el análisis de los datos sin perder demasiada información importante. En aplicaciones prácticas, este proceso ayuda a **reducir el ruido**, hacer los datos más manejables y mejorar el rendimiento de los algoritmos de machine learning.

> **Ejemplo:** Al analizar imágenes de resonancias magnéticas, se puede aplicar PCA para reducir la cantidad de datos de cada imagen, centrándose en las regiones cerebrales más relevantes, lo que permite que el modelo se entrene de manera más eficiente y con menos recursos computacionales.

#### Autoencoders

Al igual que la técnica PCA, los **autoencoders** son un tipo de red neuronal utilizada principalmente para la **reducción de dimensionalidad** en proyectos de reconocimiento de imágenes. Se componen de dos partes principales: el **encoder** y el **decoder**. El encoder comprime los datos de entrada a una representación más pequeña, mientras que el decoder intenta reconstruir la entrada original a partir de esa representación comprimida. El objetivo es minimizar la diferencia entre la entrada original y la reconstrucción, de modo que se capturen las características más importantes de los datos.

El proceso de **reducción de dimensionalidad se da en el encoder**, donde los datos se transforman a un espacio de menor dimensión llamado **código latente**. Este código latente retiene la información esencial de los datos, descartando detalles innecesarios o redundantes. Este enfoque es especialmente útil en datos con muchas variables, como es el caso de imágenes, en donde mantener todas las dimensiones puede ser ineficiente y causar sobreajuste en modelos complejos.

> **Ejemplo:** En sistemas de reconocimiento facial, los *autoencoders* pueden aprender a codificar rostros humanos en una representación de características compacta, lo que permite al modelo reconocer patrones faciales clave y hacer predicciones con mayor precisión.

### Representación de características

Una vez que la imagen ha sido codificada y su dimensionalidad reducida, es crucial que el modelo pueda **extraer características** que le permitan identificar patrones y estructuras dentro de la imagen. Estas características pueden ser de bajo nivel, como bordes y texturas, o de alto nivel, como objetos y formas complejas.

En la **representación de características**, se utilizan diferentes herramientas para transformar los píxeles de la imagen en vectores de características que el modelo pueda analizar. Entre las técnicas más populares para extraer características se encuentran las **redes neuronales convolucionales (CNNs)**.

#### Redes neuronales convolucionales (CNN)

Las **redes neuronales convolucionales** son uno de los métodos más eficaces para la codificación y extracción de características de imágenes. A través de capas convolucionales, las CNN aplican filtros sobre la imagen para identificar patrones locales, como bordes, texturas o detalles de un objeto. Estas capas convolucionales permiten reducir la dimensionalidad espacial de las imágenes a medida que avanzan por la red, sin perder la información más relevante.

La estructura jerárquica de las CNN es clave para su éxito. Las primeras capas suelen aprender características simples como líneas o bordes, mientras que las capas más profundas capturan representaciones más complejas, como formas y objetos completos. Además, las CNN aplican técnicas de **pooling** para reducir aún más la dimensionalidad de la imagen, manteniendo las características esenciales y mejorando la eficiencia computacional.

Imaginemos que queremos entrenar una **CNN** para clasificar imágenes de dígitos escritos a mano, como en el conocido dataset **MNIST**. Al ingresar una imagen de un dígito en escala de grises, que tiene un tamaño de 28x28 píxeles, cada píxel representa una intensidad de color que oscila entre 0 y 255. La CNN comienza aplicando una serie de filtros pequeños, también conocidos como *kernels*, sobre la imagen. Estos filtros recorren la imagen para detectar patrones locales, como bordes o texturas. Por ejemplo, si un filtro detecta un borde vertical, activará fuertemente esa región, permitiendo al modelo reconocer formas características del dígito.

Después de aplicar estos filtros, se utiliza una función de activación del tipo **ReLU** (Rectified Linear Unit) para convertir los valores negativos en ceros. Esto permite introducir no linealidad en la red y ayuda al modelo a captar relaciones complejas. A continuación, se aplica un proceso llamado **pooling**, cuyo objeto es reducir la resolución de la imagen, seleccionando solo la información más importante de áreas pequeñas. Este proceso nuevamente reduce la dimensionalidad, haciendo que el modelo sea más eficiente y que se preserven las características clave sin perder detalles críticos.

Este proceso de aplicar filtros, activaciones y reducir la resolución se repite varias veces en la red. A medida que se avanza por las capas, las características que la red aprende se vuelven más abstractas. En las primeras capas, puede detectar líneas simples, mientras que en las capas posteriores puede identificar patrones más completos, como bucles o formas que son propias de los dígitos.

Una vez que la red ha extraído suficientes características de la imagen, pasa esta información a una capa completamente conectada. Aquí, cada neurona se conecta a todas las neuronas de la capa anterior y combina las características aprendidas para generar una predicción. Al final de la red, existe una capa con 10 salidas, cada una representando uno de los dígitos posibles (del 0 al 9). La red elige el dígito que tenga la mayor probabilidad.

> [!tip]
>
> Puedes ver este vídeo resumen acerca de CNN para aclarar los concepto de esta sección: https://youtu.be/V8j1oENVz00?si=ad-UKxREsB0hFO7o

> [!tip]
>
> El **MNIST** (Modified National Institute of Standards and Technology) es un conjunto de datos ampliamente utilizado en el aprendizaje automático, especialmente en el campo de la clasificación de imágenes. Contiene **70.000 imágenes** de dígitos escritos a mano, divididas en 60.000 para entrenamiento y 10.000 para prueba. Cada imagen está en escala de grises con un tamaño de **28x28 píxeles**, donde los dígitos varían de 0 a 9. MNIST es utilizado frecuentemente para entrenar y evaluar algoritmos de clasificación, como redes neuronales, SVMs o KNN, siendo un estándar para medir el rendimiento en tareas de reconocimiento de patrones.

> **Ejemplo:** En un sistema de visión por computadora para vehículos autónomos, las CNN pueden procesar imágenes de cámaras en tiempo real para identificar objetos como peatones, señales de tráfico y otros vehículos. Las capas convolucionales permiten detectar estos elementos clave, ayudando al sistema a tomar decisiones seguras y precisas.

#### Codificación con SIFT y HOG

Aunque las CNN son la técnica dominante en la representación de características de imágenes, existen otros enfoques tradicionales que siguen siendo útiles en ciertos contextos. Dos de ellos son el **Scale-Invariant Feature Transform (SIFT)** y el **Histograma de Gradientes Orientados (HOG)**. El primero de ellos trata de detectar puntos clave en una imagen que son invariantes a cambios de escala, rotación y transformación. Estas características son extremadamente útiles para tareas de reconocimiento de objetos donde los objetos pueden aparecer en diferentes posiciones o tamaños dentro de la imagen. Con el método HOG, se busca detectar objetos mediante el cálculo de gradientes de la imagen y analizar la orientación de las líneas dentro de la misma. HOG es particularmente efectivo para tareas como la detección de personas en imágenes o videos.

> **Ejemplo:** En un sistema de seguridad para detección de intrusos, HOG podría utilizarse para identificar formas humanas en las imágenes de cámaras de vigilancia, alertando sobre la presencia de personas en áreas restringidas.

> [!important]
>
> La codificación de imágenes y la representación de características son elementos necesariamente a tener en cuenta en los sistemas de aprendizaje automático que trabajan con datos visuales. La **reducción dimensional** permite gestionar grandes volúmenes de datos y hacer más eficiente el entrenamiento de los modelos, mientras que las técnicas de **extracción de características** proporcionan la base para que los algoritmos identifiquen patrones en las imágenes. Desde PCA y autoencoders hasta CNN y métodos como SIFT o HOG, cada técnica tiene su papel en la creación de representaciones eficaces y robustas para la clasificación, detección y segmentación de imágenes.

**Para reflexionar...**

> **¿Cómo influye la calidad de la codificación de las imágenes en el rendimiento de los modelos de machine learning?**
> **Pistas**: Considera cómo la extracción de características y la reducción de dimensionalidad pueden afectar la precisión y eficiencia de un modelo en diferentes tareas.

## Codificación de textos: embeddings y representaciones distribuidas

El procesamiento de texto en machine learning y modelos de inteligencia artificial ha avanzado de manera significativa en los últimos años gracias al uso de **embeddings** y **representaciones distribuidas**. Estos enfoques han permitido una representación más eficiente y rica del texto, capturando relaciones semánticas entre palabras y conceptos que las técnicas tradicionales, como el **one-hot encoding**, no podían ofrecer. El texto, a diferencia de los datos numéricos o categóricos, es inherentemente complejo. Las palabras pueden tener múltiples significados según el contexto, y el simple hecho de representar una palabra con un número o vector binario (como en **one-hot encoding**) no captura las relaciones semánticas entre las palabras.

Los **embeddings** y las **representaciones distribuidas** resuelven este problema al proyectar palabras en un espacio vectorial continuo donde las palabras con significados similares están cerca unas de otras. Esta propiedad es crucial para tareas como **clasificación de textos**, **traducción automática**, **análisis de sentimientos**, y muchos otros campos del **procesamiento del lenguaje natural (NLP)**.

Sin embargo, antes de los embbedings la codificación de textos se basó en una técnica denominada ***Bag of words***

### Primeros pasos en la codificación de textos

#### Bag of Words

La **codificación Bag of Words (BoW)** es uno de los primeros enfoques utilizados para representar texto de manera que los algoritmos de machine learning puedan procesarlo. Este método convierte el texto en una representación numérica simple, basada en la **frecuencia de aparición** de palabras dentro de un documento. En su forma más básica, BoW ignora la estructura gramatical y el orden de las palabras, considerando solo cuántas veces aparece cada palabra en un documento. Cada documento se representa como un vector en el que cada dimensión corresponde a una palabra del vocabulario del conjunto de datos, y el valor de cada dimensión es la cantidad de veces que esa palabra aparece en el documento.

Aunque BoW es una técnica extremadamente sencilla, es un **precedente importante** para las codificaciones más avanzadas, como los **embeddings**. Introdujo el concepto de **representación numérica** de texto, lo que facilitó el uso de modelos de machine learning para tareas como la **clasificación de textos** o el **análisis de sentimientos**. A pesar de su simplicidad, BoW presenta limitaciones claras: no captura el orden de las palabras, no refleja las relaciones semánticas entre términos, y genera representaciones tipo *sparse* (matrices con muchos elementos nulos) de escasa aportación y de alta dimensionalidad.

Sin embargo, su simplicidad hizo que fuera ampliamente utilizado en los primeros sistemas de procesamiento de lenguaje natural. Con el tiempo, los investigadores desarrollaron métodos más avanzados, como **TF-IDF** (Term Frequency-Inverse Document Frequency) y las **representaciones distribuidas** como **Word2Vec** o **BERT**, que abordan las limitaciones de BoW y capturan la semántica del texto de manera más efectiva.

> **Ejemplo sencillo de implementación de la codificación BoW**
>
> Imaginemos que tenemos tres frases simples:
>
> 1. "El gato está durmiendo"
> 2. "El perro está jugando"
> 3. "El gato y el perro están corriendo"
>
> Primero, creamos un vocabulario con todas las palabras únicas en el conjunto de frases:
>
> **Vocabulario**: [El, gato, está, durmiendo, perro, jugando, y, corriendo, están]
>
> Luego, representamos cada frase como un vector de frecuencias, donde cada dimensión corresponde a la presencia (o ausencia) de una palabra en esa frase:
>
> 1. "El gato está durmiendo" → [1, 1, 1, 1, 0, 0, 0, 0, 0]  
> 2. "El perro está jugando" → [1, 0, 1, 0, 1, 1, 0, 0, 0]  
> 3. "El gato y el perro están corriendo" → [1, 1, 0, 0, 1, 0, 1, 1, 1]
>
> Cada vector tiene tantos elementos como palabras en el vocabulario. El valor es 1 si la palabra aparece en la frase y 0 si no aparece. También puede usarse la frecuencia de las palabras en vez de solo la presencia en el caso en que una palabra aparezca varias veces en la misma frase.
>
> Vemo cómo BoW **representa cada frase ignorando el orden y la gramática, centrándose solo en la aparición de las palabras.**

**Para reflexionar...**

> **¿Cuáles son las limitaciones de BoW en la representación del significado de las palabras, y cómo los métodos más avanzados, como los embeddings, superan estas limitaciones?**
>
> Reflexiona sobre cómo BoW ignora el contexto y la semántica de las palabras, mientras que los embeddings capturan relaciones más complejas y representaciones continuas.

#### TF-IDF

La **codificación TF-IDF (Term Frequency-Inverse Document Frequency)** ha sido y es una técnica ampliamente utilizada en el procesamiento del lenguaje natural (NLP) para representar textos de manera numérica. A diferencia de la codificación basada en la frecuencia de términos simple, como el **Bag of Words**, TF-IDF pondera las palabras en función de su relevancia dentro de un documento y su frecuencia en un conjunto de documentos (corpus).

El **TF (frecuencia de término)** mide cuántas veces aparece una palabra en un documento particular. Sin embargo, palabras comunes como "el", "y" o "de" aparecerán con frecuencia en muchos documentos, lo que podría darles más peso del que deberían tener. Aquí es donde entra el término **IDF (frecuencia inversa de documento)**, que reduce el peso de palabras que aparecen en muchos documentos del corpus, asignando más importancia a las palabras que son relevantes pero no demasiado comunes.

La fórmula del TF-IDF es

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

Donde $t$ es un término, $d$ es un documento y $D$ es el corpus de documentos.

Esta técnica es útil para tareas como la clasificación de textos, búsqueda de información y detección de similitud de documentos, ya que ayuda a identificar las palabras clave más significativas en cada documento, mejorando el rendimiento de los modelos que dependen de estas representaciones numéricas de texto.

Por contra, las principales desventajas de **TF-IDF** son su incapacidad para capturar el significado debido a que trata cada término de forma independiente, sin considerar las relaciones entre ellos. Además, **TF-IDF** no tiene en cuenta el orden de las palabras en un documento. Las dos cuestiones anteriores pueden limitar su eficacia en problemas donde el contexto es importante. También puede ser sensible a documentos muy largos o muy cortos, y no maneja adecuadamente las palabras que no están en el corpus de entrenamiento, afectando la representación de términos raros o fuera de vocabulario.



> **Ejemplo:** Imagina que tenemos un pequeño corpus de tres documentos:
>
> 1. **Documento 1**: "El gato está durmiendo"
> 2. **Documento 2**: "El perro está jugando"
> 3. **Documento 3**: "El gato y el perro están corriendo"
>
> **Paso 1: Calcular la Frecuencia de Términos (TF)**
>
> Vamos a contar cuántas veces aparece cada palabra en cada documento:
>
> - **Documento 1**: "El" (1), "gato" (1), "está" (1), "durmiendo" (1)
> - **Documento 2**: "El" (1), "perro" (1), "está" (1), "jugando" (1)
> - **Documento 3**: "El" (1), "gato" (1), "y" (1), "perro" (1), "están" (1), "corriendo" (1)
>
> **Paso 2: Calcular la Frecuencia Inversa de Documento (IDF)**
>
> La **IDF** reduce la importancia de las palabras comunes. Se calcula usando la fórmula:
>
> $$
> IDF(t) = \log\left(\frac{N}{1 + DF(t)}\right)
> $$
>
> Donde $N$ es el número total de documentos y $DF(t)$ es el número de documentos que contienen el término $t$.
>
> - "El" aparece en los 3 documentos, así que su IDF es $ \log\left(\frac{3}{1+3}\right) = 0$
> - "Gato" aparece en 2 documentos, su IDF es $ \log\left(\frac{3}{1+2}\right) = 0.18$
> - "Durmiendo", "jugando", "corriendo", que aparecen solo en un documento, tendrán una IDF mayor.
>
> **Paso 3: Multiplicar TF por IDF**
>
> Finalmente, multiplicamos la frecuencia de cada término en cada documento por su IDF para obtener el **TF-IDF**.
>
> Por ejemplo, en el Documento 1, la palabra "gato" tendrá un **TF-IDF** de:
>
> $$
> \text{TF-IDF}(gato, Documento 1) = 1 \times 0.18 = 0.18
> $$
>
> Este valor sería mucho menor para palabras comunes como "El".

### ¿Qué son los embeddings?

Los **embeddings** son vectores de baja dimensión que representan palabras, frases o incluso documentos en un espacio continuo, donde la proximidad entre los vectores refleja la similitud semántica entre los elementos. A diferencia de las representaciones tipo *sparse* de técnicas como BoW, los embeddings son **densos** (baja dimensionalidad en comparación) y permiten capturar relaciones semánticas y contextuales entre palabras.

#### Word2Vec

Uno de los primeros métodos de generación de embeddings en aparecer fue **Word2Vec**, desarrollado por Google. El objetivo principal de Word2Vec es aprender representaciones vectoriales de palabras que mantengan su contexto semántico. El principio detrás de Word2Vec es que las palabras con significados similares aparecerán en contextos similares. Por ejemplo, en frases como "El gato está durmiendo" y "El perro está durmiendo", Word2Vec aprendería que "gato" y "perro" tienen un significado similar porque aparecen en contextos similares.

Word2Vec tiene dos enfoques principales para aprender las representaciones de palabras. El primero es el **Continuous Bag of Words (CBOW)**, donde el modelo predice una palabra objetivo a partir de las palabras que la rodean. Por ejemplo, si se tiene una frase incompleta como "El _ está durmiendo", el modelo intentará adivinar que la palabra faltante podría ser "gato" o "perro". Este método permite que el modelo asocie palabras que tienden a aparecer en contextos similares, mejorando así su comprensión de patrones lingüísticos. El segundo enfoque es **Skip-Gram**, que funciona de manera inversa al CBOW. En lugar de predecir una palabra a partir de su contexto, el modelo predice las palabras que suelen aparecer alrededor de una palabra dada. Por ejemplo, dado "gato", intentará predecir qué palabras aparecen cerca de "gato", como "está" o "durmiendo". Este enfoque es particularmente útil para capturar relaciones semánticas entre palabras, incluso si están a cierta distancia dentro de una oración.

Ambos enfoques generan embeddings que capturan las relaciones semánticas entre las palabras. Para generar los embbedings Word2Vec utiliza **un modelo basado en una red neuronal simple con capas ocultas**. La entrada del modelo es un **one-hot vector** (una representación en la que solo la posición correspondiente a la palabra está marcada como 1, y el resto son 0s) para la palabra del contexto o palabra objetivo. Durante el entrenamiento, el modelo ajusta los pesos para minimizar un **error** (función de coste), que puede calcularse como la diferencia entre las predicciones del modelo y las verdaderas palabras del contexto (CBOW) o las palabras objetivo (Skip-gram). La optimización se realiza a través de **gradiente descendente** y retropropagación. A medida que el modelo ve más ejemplos, ajusta los pesos, de manera que las palabras que suelen aparecer juntas en el texto terminan teniendo representaciones (vectores) similares en el espacio vectorial.

El objetivo final es que las palabras se transformen en vectores numéricos (embeddings) en un espacio de alta dimensión. **Palabras con contextos similares tendrán vectores cercanos**, lo que significa que el modelo ha capturado sus relaciones semánticas. Por ejemplo, si “rey” y “reina” aparecen frecuentemente en contextos similares, sus embeddings estarán próximos en el espacio vectorial. Seguramente los vectores de las palabras "gato", "perro" y "mascota" también estarían relativamente cerca entre sí, reflejando su similitud semántica. Al final, todo este proceso permite que las relaciones semánticas implícitas en los datos de entrenamiento se capturen sin intervención explícita.

También es importante señalar que Word2Vec tiene la capacidad de capturar no solo relaciones semánticas simples entre palabras, sino también relaciones complejas. Por ejemplo, puede aprender que la diferencia entre "hombre" y "mujer" es similar a la diferencia entre "rey" y "reina". Esta propiedad es extremadamente útil en tareas como **análisis de sentimientos** en donde es necesario capturar matices en el significado de las palabras en función del contexto; **sistemas de recomendación**, ya que al representar ítems (productos, películas, etc.) como vectores, puede recomendar elementos similares; y en **traducción automática**, ya que, aunque con limitaciones, es capaz de captar las relaciones entre palabras de diferentes lenguajes.

El entrenamiento de embeddings se realiza mediante el aprendizaje de una función de costo, que se optimiza para predecir contextos de palabras con precisión. En el caso de **Word2Vec**, se entrenan modelos **en grandes corpus de texto**, ajustando los pesos de los vectores de las palabras para minimizar la diferencia entre las predicciones de palabras contextuales y las palabras reales del corpus.

#### Más técnicas de embedding: GloVe y FastText

Al igual que Word2Vec, **GloVe** y **FastText** son técnicas populares de generación de **embeddings** para palabras que son utilizadas en **Procesamiento de Lenguaje Natural (NLP)**. Aunque todas comparten el objetivo de convertir palabras en vectores numéricos, existen diferencias clave en cómo logran este objetivo y cómo manejan la información semántica. Vamos a reseñar en primera aproximación las principales similitudes y diferencias.

Primero hay que insistir en el hecho de que los tres métodos buscan generar representaciones vectoriales de palabras, de tal manera que **palabras con significados similares tengan vectores cercanos en el espacio vectorial**. Así pues, los tres modelos tienen la capacidad de capturar relaciones semánticas entre palabras. Por ejemplo, las relaciones de tipo "rey - reina" o "hombre - mujer" pueden capturarse porque las palabras se colocan de manera similar en el espacio vectorial. Los tres modelos crean **espacios vectoriales** donde la semántica y las relaciones entre las palabras están presentes a través de las distancias y direcciones entre vectores. Por último también hay que destacar que los tres modelos **se entrenan de manera no supervisada utilizando grandes corpus de texto**.

Hasta aquí las similitudes. Sin embargo los tres modelos tienen rasgos que los diferencian claramente entre sí. En primer lugar el **método de aprendizaje**. Ya hemos comentado que Word2Vec usa una red neuronal simple y se entrena mediante dos enfoques. En CBOW, el modelo predice una palabra objetivo dado su contexto, mientras que en Skip-gram predice el contexto dado una palabra objetivo. Ahora bien, en el caso de **GloVe**, en lugar de una red neuronal, el modelo se basa **en matrices de co-ocurrencia global de palabras**. La idea central es que se entrena creando una matriz que representa la frecuencia con que las palabras co-ocurren en un gran corpus. Luego, se factoriza esta matriz para obtener los embeddings. Por su parte, **FastText** sí que se basa en el enfoque de Word2Vec (particularmente en Skip-gram) pero, a diferencia de Word2Vec, **FastText** descompone las palabras en **n-gramas de caracteres**. Esto le permite capturar información morfológica de las palabras, lo que es especialmente útil para manejar palabras raras o formas no vistas antes.

También existen diferencias en cuanto al **tratamiento de palabras fuera del vocabulario (OOV)**. En este caso **Word2Vec** no maneja bien las palabras que no aparecen en el corpus de entrenamiento (palabras fuera del vocabulario), algo que comparte con **GloVe**. Sin embargo **FastText** ofrece una ventaja significativa dado que FastText descompone las palabras en **n-gramas**, y así puede generar embeddings para palabras no vistas previamente al combinar los embeddings de los subcomponentes (n-gramas) de la palabra.

> **Ejemplo:** si FastText no ha visto la palabra "descomposición" en su fase de entrenamiento, pero ha aprendido representaciones para los n-gramas de caracteres como "des-", "-com-", "-pos-", "-ición", puede generar un embedding aproximado para la palabra "descomposición" a partir de los n-gramas que la componen. Así, aunque la palabra completa no esté en su vocabulario, FastText puede deducir su significado basándose en las partes de la palabra que sí conoce.

> [!Note]
>
> **Word2Vec** fue propuesto en 2013 por **Tomas Mikolov** y su equipo en **Google**. Este modelo marcó un hito en el procesamiento del lenguaje natural, ya que permitió generar representaciones vectoriales de palabras en las que era tenido en cuenta el significado de las mismas. Su desarrollo facilitó el aprendizaje de relaciones semánticas entre palabras, mejorando las tareas de comprensión y generación de lenguaje natural. Un año después, en 2014, un equipo de la **Universidad de Stanford**, liderado por **Jeffrey Pennington**, **Richard Socher** y **Christopher Manning**, presentó **GloVe** (Global Vectors for Word Representation). Este modelo, a diferencia de Word2Vec, utilizaba estadísticas globales de co-ocurrencia de palabras, ofreciendo una visión diferente para capturar las relaciones semánticas en grandes corpus de texto. Finalmente, en 2016, **Facebook AI Research (FAIR)**, con **Armand Joulin**, **Edouard Grave**, **Piotr Bojanowski**, y nuevamente **Tomas Mikolov**, lanzó **FastText**. Este modelo, basado en Word2Vec, introdujo la capacidad de manejar palabras fuera del vocabulario (OOV) mediante la descomposición de palabras en **n-gramas** de caracteres, lo que mejoró la representación de palabras raras o morfológicamente complejas.

**Para reflexionar...**

> **¿Cómo impacta la capacidad de FastText para manejar palabras fuera del vocabulario en la robustez de los modelos en aplicaciones del mundo real, como el análisis de texto en redes sociales?**
>
> Reflexiona sobre el valor de manejar palabras raras y nuevas, como abreviaciones, jerga o errores tipográficos.

### ¿Qué son las representaciones distribuidas? Transformers

Las **representaciones distribuidas** se refieren a la idea de representar una palabra, frase o documento como un vector que **captura múltiples características semánticas y sintácticas, distribuyendo esa información entre las diferentes dimensiones del vector**. El término fue popularizado por **Geoffrey Hinton** y es fundamental en la idea de que el significado de una palabra no se representa de manera aislada, sino a través de su relación con otras palabras.

A medida que las palabras se proyectan en un espacio vectorial mediante embeddings, el sistema puede aprender relaciones complejas, como sinónimos, asociaciones y similitudes contextuales. Las representaciones distribuidas son la base de los modelos más modernos de NLP, como los **transformers**, que permiten el procesamiento de secuencias de texto con gran precisión.

Un **transformer** es un tipo de arquitectura de red neuronal diseñada para procesar secuencias de datos y especialmente eficaz en tareas de **procesamiento del lenguaje natural (NLP)**. Esta arquitectura fue introducida en 2017 por Vaswani en su artículo ***Attention is All You Need*** y revolucionó el campo al superar las limitaciones de las redes recurrentes (RNN) y las redes neuronales convolucionales (CNN) al manejar relaciones a larga distancia en las secuencias de datos.

> [!tip]
>
> Cuando tuvo lugar la publicación del artículo "Attention is All You Need" en 2017, miembros de la comunidad científica fueron escépticos respecto a la sustitución de las redes recurrentes y convolucionales, que eran predominantes en ese momento en NLP. Sin embargo, los autores del artículo, Ashish Vaswani y su equipo, confiaron en la simplicidad y efectividad del **mecanismo de atención** y siguieron adelante con su enfoque. A la postre y a pesar de las dudas iniciales, el modelo de transformers demostró un rendimiento superior en diversas tareas de NLP, sorprendiendo a muchos investigadores hasta entonces escépticos. Esta apuesta resultó finalmente buena, quedando reflejada en uno de los artículos más influyentes en la historia reciente del campo de la IA.
>
> Un vídeo muy interesante sobre esta nueva tecnología: https://youtu.be/aL-EmKuB078?si=ctUXaEY4Beata5Zu

La clave del transformer es el **mecanismo de atención**, que permite al modelo enfocarse en diferentes partes de la secuencia de entrada con diferentes "pesos" en función de su relevancia para una tarea en particular. Esto elimina la necesidad de procesar los datos de manera secuencial, como lo hacen las RNN, permitiendo un entrenamiento más paralelo y eficiente.

El transformer utiliza dos componentes principales: un **codificador**, que procesa la entrada, y un **decodificador**, que genera la salida. Gracias a su capacidad para capturar dependencias a largo plazo y gestionar secuencias de datos de forma más eficiente, el transformer se ha convertido en la base de modelos avanzados como **BERT**, **GPT** y **T5**, que dominan el procesamiento del lenguaje natural en tareas como traducción automática, generación de texto y comprensión de lenguaje.

> [!note]
>
> **BERT y GPT, dos formas de entender el lenguaje**
>
> Dentro de los modelos basados en **transformers**, dos nombres destacan por su impacto: **BERT** y **GPT**. Ambos aprenden del lenguaje a gran escala, pero lo hacen con enfoques distintos.
>
> **BERT** (de Google) está diseñado para **entender el texto**, analizando cada palabra en relación con las que la rodean, tanto antes como después. Por eso se dice que su entrenamiento es **bidireccional**. Este enfoque lo hace ideal para tareas de comprensión, como responder preguntas o analizar sentimientos.
>
> **GPT** (de OpenAI), en cambio, se especializa en **generar texto**. Aprende a predecir la palabra siguiente en una secuencia, avanzando **de izquierda a derecha**. Gracias a esto puede producir textos largos, coherentes y creativos.
>
> En resumen, **BERT interpreta** y **GPT crea**: ambos usan la misma base técnica —los transformers—, pero con objetivos complementarios dentro del procesamiento del lenguaje natural.

### Conclusiones: Fortalezas y desafíos de los embeddings

Los **embeddings** ofrecen varias fortalezas que los hacen muy útiles en el procesamiento del lenguaje natural (NLP). La primera, como hemos explicado, es su capacidad para **capturar relaciones semánticas** entre palabras. En lugar de tratar a cada palabra como una entidad independiente, los embeddings permiten que las palabras con significados similares se representen de manera cercana en un espacio vectorial. Esto mejora significativamente la comprensión del texto por parte de los modelos y facilita tareas como la clasificación de texto o la traducción automática.

Otra ventaja importante es la **reducción de la dimensionalidad**. Mientras que técnicas tradicionales como el one-hot encoding generan vectores muy largos y llenos de elementos nulos, los embeddings permiten comprimir esta información en vectores de menor dimensión sin perder las relaciones relevantes entre las palabras. Esto hace que los modelos sean más eficientes y manejables, especialmente cuando se trabaja con grandes cantidades de datos.

A todo lo anterior hay que sumar que los embeddings mejoran la **generalización** de los modelos. Por ejemplo, técnicas como FastText permiten manejar palabras nuevas o fuera del vocabulario entrenado al descomponer palabras en subcomponentes más pequeños, lo que facilita su reconocimiento y clasificación, incluso en situaciones no vistas durante el entrenamiento.

Sin embargo, **los embeddings también presentan algunos desafíos**. Uno de los más notables es su **opacidad**. Aunque son potentes, las representaciones vectoriales que crean son abstractas y difíciles de interpretar directamente. A diferencia de los métodos simbólicos, que ofrecen relaciones explícitas entre conceptos, los embeddings encapsulan significados de manera implícita, lo que dificulta su comprensión.

Otro reto es su **dependencia del corpus de entrenamiento**. Si los datos utilizados para entrenar los embeddings contienen sesgos, estos pueden transferirse al modelo, lo que es especialmente problemático en dominios sensibles como el procesamiento de textos médicos o legales. Además, si el corpus es limitado, los embeddings pueden no capturar adecuadamente la riqueza del lenguaje, afectando el rendimiento del modelo en ciertos contextos.

### Ejemplos

> **Ejemplo**: Un modelo de clasificación de correos electrónicos como spam o no spam podría utilizar embeddings para representar cada palabra en los correos electrónicos, permitiendo que el modelo capture relaciones semánticas entre palabras como "oferta", "gratuito" y "fraude", que son típicas de mensajes de spam.

> **Ejemplo**: En la traducción automática, los embeddings permiten a los modelos comprender la similitud entre palabras en diferentes idiomas. Por ejemplo, las palabras "casa" en español y "house" en inglés estarían cerca en el espacio vectorial, facilitando la traducción entre estos términos.

> **Ejemplo**: Un sistema de recomendación de productos podría utilizar embeddings para mapear descripciones de productos en un espacio vectorial, de modo que productos con descripciones similares (como "teléfonos móviles" y "smartphones") se recomienden de manera más precisa a los usuarios.

**Para reflexionar...**

> **¿Cómo afectan los sesgos en los corpus de entrenamiento al comportamiento de los embeddings en tareas como la clasificación o el análisis de sentimientos?**
>
> Reflexiona sobre cómo los sesgos presentes en los datos originales pueden ser amplificados en los embeddings, afectando la capacidad de los modelos para tomar decisiones justas y precisas.

## Codificación de series temporales

El análisis y la codificación de series temporales es un área crucial dentro del aprendizaje automático y la ciencia de datos, ya que muchas aplicaciones reales involucran datos que evolucionan a lo largo del tiempo. Las series temporales son secuencias de datos recolectadas a intervalos de tiempo regulares, como los precios de las acciones, registros meteorológicos o mediciones de sensores en tiempo real. La codificación de estos datos de manera adecuada es fundamental para garantizar que los modelos de machine learning puedan capturar las dependencias temporales y realizar predicciones precisas.

### Características de las series temporales

Las series temporales poseen ciertas características únicas que las diferencian de otros tipos de datos, lo que hace necesario utilizar técnicas de preprocesamiento y codificación específicas. Podríamos enumerar las siguientes:

1. **Dependencia temporal**: Los datos de series temporales tienen una relación entre observaciones sucesivas. Esta dependencia entre los valores previos y futuros es clave para construir modelos predictivos efectivos.
2. **Estacionalidad**: En muchas series temporales, los patrones se repiten a intervalos regulares (diarios, semanales, anuales), lo que se denomina estacionalidad. Por ejemplo, los patrones de temperatura tienen estacionalidad anual, mientras que el tráfico web puede tener un comportamiento diario y semanal.
3. **Tendencia**: Las series temporales a menudo muestran una tendencia, que puede ser una dirección general ascendente o descendente en el tiempo, como el crecimiento de usuarios en una plataforma de e-commerce.
4. **Ruido**: A menudo, las series temporales contienen ruido que no está relacionado con la estructura subyacente del problema y debe ser filtrado para mejorar el rendimiento del modelo.

### Técnicas de codificación y preprocesamiento de series temporales

#### Ventaneo de datos (windowing)

Una de las técnicas más comunes para transformar una serie temporal en un formato adecuado para el aprendizaje automático es el "ventaneo" o ***sliding window***. Esta técnica consiste en dividir la serie temporal en segmentos o ventanas de datos consecutivos para capturar la dependencia entre puntos en el tiempo. El enfoque *sliding window* genera un conjunto de ventanas que luego se utiliza como entradas a los modelos.

> **Ejemplo**: En la predicción de la demanda energética diaria, la técnica de sliding window permite utilizar la demanda de los últimos 7 días para predecir la demanda del día siguiente. Esto crea múltiples subconjuntos de datos que se pueden utilizar para entrenar un modelo de regresión.

#### Codificación temporal con *time lags*

Las series temporales requieren capturar la correlación entre observaciones anteriores y futuras. Los *time lags* o retrasos temporales son una técnica para codificar explícitamente las observaciones pasadas en el modelo. Al incluir variables de retraso, como el valor de la serie temporal en los días previos, se mejora la capacidad del modelo para hacer predicciones más precisas basadas en el historial reciente.

> **Ejemplo**: En la predicción de ventas, podemos utilizar como característica el volumen de ventas de los últimos tres días. Esto crea una nueva matriz de entrada con las observaciones pasadas como variables predictoras.

#### Codificación de características derivadas

Otra técnica útil es la creación de características derivadas, donde se calculan nuevas variables a partir de los datos originales de la serie temporal. Ejemplos comunes incluyen el cálculo de la media móvil, que suaviza los datos, o el cálculo de la diferencia entre valores consecutivos para capturar la tasa de cambio de la serie temporal.

> **Ejemplo**: En una serie temporal que mide el precio de las acciones, se puede incluir la diferencia entre el precio de hoy y el precio de ayer como una nueva variable que refleje el cambio diario en el precio de las acciones.

#### Codificación estacional

Las series temporales que exhiben estacionalidad requieren un tratamiento especial para capturar estas variaciones periódicas. Esto puede implicar codificar la estacionalidad explícitamente utilizando variables que representen el mes, la semana o el día del año. También se pueden emplear **Fourier transforms** o **one-hot encoding** para representar componentes estacionales complejas.

> **Ejemplo**: En una serie temporal de demanda de productos en un supermercado, podemos incluir el mes o la estación del año como una variable categórica para capturar la variación estacional en las ventas.

### Modelos y algoritmos para series temporales

La codificación efectiva de las series temporales permite el uso de una amplia gama de algoritmos de aprendizaje automático. Entre los modelos más utilizados para series temporales podemos encontrar los de **regresión lineal**. Estos pueden ser usada en series temporales cuando las características relevantes, como los *lags* y las variables derivadas, han sido correctamente codificadas. Aún así, su capacidad para manejar dependencias temporales es limitada. Otro de los algoritmos comúnmente utilizados es **ARIMA** (AutoRegressive Integrated Moving Average). Este modelo combina componentes de regresión, manejo de tendencias y promedios móviles para capturar la dependencia de errores pasados.

Sin embargo, son las **Redes neuronales recurrentes (RNN)** y sus variantes, como las **LSTM** (Long Short-Term Memory), las que están diseñadas específicamente para manejar la naturaleza secuencial de las series temporales. Estas redes permiten que el modelo capture dependencias de largo plazo, algo que no es fácilmente manejado por modelos tradicionales.

> **Ejemplo**: En la predicción de la temperatura diaria, una RNN puede procesar la secuencia completa de temperaturas pasadas y aprender a predecir la temperatura futura basándose en patrones que ocurren en lapsos temporales largos.

Por último, los **transformers**, aunque inicialmente diseñados para el procesamiento de lenguaje natural, han mostrado ser efectivos en series temporales gracias a su capacidad para manejar secuencias y capturar dependencias complejas a largo plazo.

### Manejo de series temporales irregulares

En muchas aplicaciones, los datos no son recolectados a intervalos regulares. Esto plantea un reto adicional a la codificación de series temporales, ya que muchas técnicas tradicionales asumen una periodicidad constante. Para tratar series temporales irregulares, se pueden aplicar técnicas como el **imputado de valores faltantes**, interpolación o agregar marcadores temporales adicionales para reflejar la naturaleza irregular de los datos.

> **Ejemplo**: En el monitoreo de sensores industriales, los datos pueden no ser recolectados en intervalos regulares debido a fallos en la transmisión. Para abordar esto, se puede usar interpolación lineal o técnicas de reamostrado para crear una serie temporal continua antes de entrenar el modelo.

> [!tip]
>
> Puedes acceder a una pequeña guía de como funcionan las series temporales en este vídeo: https://youtu.be/6VvYgPXnB40?si=GpcjgQgdF-jLVjjj

**Para reflexionar...**

> **¿Cómo se puede manejar la estacionalidad en una serie temporal que tiene ciclos irregulares o de duración variable?**
>
> **Pistas**: Considera técnicas de análisis de Fourier o el uso de transformaciones específicas para capturar componentes estacionales variables.

## Codificación de datos desbalanceados

Los **datos desbalanceados** son un problema común en proyectos de machine learning, especialmente en tareas de clasificación. Un conjunto de datos se considera desbalanceado cuando las clases no están distribuidas de manera uniforme, es decir, una o más clases tienen significativamente más instancias que otras. Este desbalance puede sesgar el modelo hacia la clase mayoritaria, lo que conduce a predicciones inadecuadas para las clases minoritarias. Para abordar este problema, existen diversas técnicas de codificación y procesamiento que mejoran el rendimiento del modelo y su capacidad para generalizar correctamente en las clases minoritarias.

### Retos que plantean los datos desbalanceados

Uno de los principales desafíos con los **datos desbalanceados** es que los modelos tienden a enfocarse en la clase mayoritaria, ya que al minimizar el error general, pueden llegar a ignorar las instancias de la clase minoritaria. Esto se traduce en que un modelo que presenta un buen rendimiento global, presenta un mal desempeño en la predicción de las clases menos representadas. Por ejemplo, en un conjunto de datos para la detección de fraudes bancarios, la clase de transacciones fraudulentas podría representar menos del 1% de las transacciones totales, lo que dificultaría la capacidad del modelo para identificarlas de manera correcta.

Otro reto común es la interpretación de las métricas de evaluación. Cuando los datos están desbalanceados, métricas como la **exactitud** pueden ser engañosas. Un modelo que predice siempre la clase mayoritaria puede tener una alta exactitud, pero un rendimiento deficiente al clasificar la clase minoritaria. Por ello, es crucial usar métricas más adecuadas, como el **F1-score**, o el **recall**, que ofrecen una mejor comprensión del comportamiento del modelo frente a datos desbalanceados.

### Estrategias de codificación para datos desbalanceados

Existen varias estrategias para manejar los datos desbalanceados, tanto a nivel de preprocesamiento como a nivel de diseño del modelo. Las más comunes incluyen el **submuestreo**, el **sobremuestreo** y el uso de **métodos basados en costes**.

#### Sobremuestreo de la clase minoritaria

El **sobremuestreo** consiste en aumentar el número de instancias de la clase minoritaria duplicando las observaciones existentes o generando nuevas instancias sintéticas. Uno de los métodos más populares para sobremuestrear es **SMOTE (Synthetic Minority Over-sampling Technique)**. SMOTE genera nuevas instancias sintéticas de la clase minoritaria interpolando entre las observaciones existentes. De esta manera, el modelo tiene más oportunidades de aprender las características de la clase minoritaria.

> **Ejemplo:** En un conjunto de datos de clasificación de enfermedades raras, donde la cantidad de casos positivos es muy baja en comparación con los negativos, se puede aplicar SMOTE para generar nuevas instancias sintéticas de la clase positiva, equilibrando así el número de instancias en ambas clases.

#### Submuestreo de la clase mayoritaria

El **submuestreo** reduce el número de instancias de la clase mayoritaria para igualar su tamaño al de la clase minoritaria. Aunque esto puede ayudar a equilibrar el conjunto de datos, también puede llevar a la pérdida de información valiosa de la clase mayoritaria, lo que podría afectar el rendimiento general del modelo.

> **Ejemplo:** En un conjunto de datos de detección de correos electrónicos de spam, se podría reducir el número de ejemplos de correos legítimos (la clase mayoritaria) para igualarlo al número de correos de spam, creando así un conjunto de datos balanceado para el entrenamiento del modelo.

#### Métodos basados en costes

Otra técnica para manejar datos desbalanceados es el uso de **métodos basados en costes**. En lugar de equilibrar el conjunto de datos, estos métodos ajustan el modelo para penalizar más fuertemente los errores en la predicción de la clase minoritaria. Esto se logra mediante la asignación de pesos desiguales a las clases en la función de pérdida, aumentando el costo de los errores en la clase minoritaria.

> **Ejemplo:** En un sistema de predicción de fraudes bancarios, se puede asignar un costo más alto a las predicciones incorrectas de transacciones fraudulentas, ya que los errores en esta clase pueden tener consecuencias graves. Al ajustar los pesos, el modelo se ve incentivado a prestar más atención a las clases minoritarias.

### Consideraciones finales

El tratamiento de datos desbalanceados también depende del tipo de problema y el contexto de la aplicación. En ciertos casos, como la medicina o la seguridad, los errores en la clasificación de la clase minoritaria pueden tener consecuencias graves, lo que justifica el uso de técnicas más agresivas para mejorar la predicción de estas clases. En otros contextos, el equilibrio entre las clases puede no ser tan crítico, y la elección de la técnica adecuada dependerá del impacto de los errores.

> **Ejemplo:** En una aplicación de diagnóstico médico, donde los falsos negativos (fallos al identificar una enfermedad) son más perjudiciales que los falsos positivos, el recall es una métrica más importante que la precisión.

**Para reflexionar...**

> **¿En qué escenarios puede ser más perjudicial utilizar el submuestreo de la clase mayoritaria que aplicar un modelo basado en costos?**
>
> **Pistas**: Reflexiona sobre cómo la reducción de datos puede limitar la capacidad del modelo para aprender patrones en la clase mayoritaria, afectando su capacidad para hacer predicciones precisas en datos futuros.

## Impacto de la codificación en el rendimiento del modelo

La **codificación de los datos** es un paso crucial en el preprocesamiento que puede influir significativamente en el rendimiento de un modelo de machine learning. La elección correcta de la técnica de codificación no solo afecta la precisión del modelo, sino también su capacidad de generalización, velocidad de entrenamiento y simplicidad de implementación. La forma en que los datos se codifican puede cambiar drásticamente los resultados del modelo, en particular en aquellos algoritmos que son sensibles a las relaciones numéricas o la dimensionalidad de los datos.

Uno de los principales impactos que la codificación tiene en el rendimiento del modelo está relacionado con cómo el modelo interpreta las **variables categóricas**. Algunos modelos, como los árboles de decisión, son más tolerantes a variables categóricas sin una codificación numérica compleja, mientras que otros, como los modelos de regresión lineal, requieren que los datos categóricos se conviertan en números para poder procesarlos correctamente. En estos casos, la codificación inapropiada puede introducir relaciones espurias entre las categorías, lo que llevaría a un deterioro en la capacidad del modelo para hacer predicciones precisas.

Las **técnicas de codificación como la codificación *one-hot*, ordinal o binaria** impactan de manera diferente en la dimensionalidad del conjunto de datos. En la codificación *one-hot*, cada categoría de una variable se convierte en una columna binaria adicional. Esto puede funcionar bien en problemas con pocas categorías, pero en situaciones donde la variable tiene muchas categorías, el aumento en la dimensionalidad puede llevar a lo que se conoce como la **maldición de la dimensionalidad**. Esto ocurre cuando un modelo tiene dificultades para capturar patrones en los datos debido al gran número de características, lo que puede resultar en **sobreajuste** y reducir la capacidad del modelo para generalizar a datos nuevos.

Por otro lado, técnicas de **codificación ordinal** pueden ser útiles en situaciones donde las categorías tienen un orden inherente, como "bajo", "medio" y "alto". Sin embargo, esta técnica puede ser problemática si el modelo asume una relación lineal entre las categorías cuando en realidad no existe tal relación. Esto puede distorsionar los patrones que el modelo intenta aprender, lo que impacta negativamente en el rendimiento.

Además de las variables categóricas, la **codificación de variables numéricas** también puede afectar el rendimiento. Técnicas como la normalización o estandarización son cruciales en modelos que dependen de las relaciones numéricas, como las redes neuronales o los métodos basados en distancia como *k-nearest neighbors* o SVM. Si los datos numéricos no están en la misma escala, los modelos pueden dar más importancia a variables con rangos más grandes, afectando negativamente su capacidad de aprender patrones significativos.

Finalmente, en problemas donde los datos están **desbalanceados**, la codificación puede jugar un papel fundamental. Por ejemplo, en problemas de clasificación donde una clase es mucho más frecuente que otra, una codificación adecuada puede ayudar a resaltar las clases minoritarias, evitando que el modelo sesgue las predicciones hacia la clase mayoritaria.

> **Ejemplo**: En un problema de clasificación de correos electrónicos, donde la variable categórica "tipo de remitente" puede tener cientos de valores, aplicar una codificación *one-hot* sin pensar en la dimensión resultante puede generar un modelo menos eficiente y difícil de entrenar. La codificación binaria, en cambio, podría reducir significativamente la dimensionalidad sin perder información.

> **Ejemplo**: En un proyecto de predicción de fraude financiero, la variable "tipo de transacción" puede tener múltiples categorías sin un orden específico. Si se aplicara una codificación ordinal, el modelo podría asumir una jerarquía inexistente entre las categorías, afectando negativamente el rendimiento.

> **Ejemplo**: En un modelo de clasificación de imágenes, la normalización de las variables que representan los píxeles es crucial para que los valores de los colores no afecten desproporcionadamente el entrenamiento. Una codificación incorrecta de los valores de los píxeles puede ralentizar el proceso de entrenamiento y llevar a un modelo menos preciso.

**Para reflexionar...**

> **¿Cómo influye la elección de la codificación en la capacidad del modelo para aprender patrones significativos a partir de los datos?**
>
> **Pistas**: Reflexiona sobre cómo técnicas como la codificación *one-hot* o la ordinal pueden alterar las relaciones que el modelo intenta aprender y cómo estas decisiones afectan el rendimiento del modelo en diferentes tipos de datos.

## Consideraciones éticas y sesgo en la codificación

En los proyectos de **machine learning**, la codificación de los datos puede tener importantes implicaciones éticas, especialmente cuando se trata de la forma en que los datos se representan y cómo las decisiones del modelo afectan a diferentes grupos de personas. El proceso de codificación, si no se maneja con cuidado, puede introducir o amplificar sesgos que impactan negativamente la equidad y la justicia en las predicciones de un modelo. Por ello, la **ética en la codificación** y el manejo del **sesgo** son áreas clave que los científicos de datos deben abordar para desarrollar modelos responsables y justos.

Uno de los desafíos más comunes es la **codificación de variables categóricas** que representan características sensibles, como género, raza, o nivel socioeconómico. Dependiendo de la técnica de codificación utilizada, estas variables pueden influir de manera desproporcionada en las predicciones del modelo, lo que podría derivar en decisiones discriminatorias o injustas. Por ejemplo, en sistemas de toma de decisiones como la **concesión de préstamos** o **evaluaciones laborales**, un sesgo en la codificación de estas características puede llevar a resultados que perpetúan la discriminación contra ciertos grupos.

Otro aspecto ético relevante es el tratamiento de los datos faltantes. La **imputación de valores** mediante técnicas como reemplazar con la media o la moda, aunque eficiente desde el punto de vista técnico, puede distorsionar los datos originales de una forma que afecte a ciertos grupos. Si los datos faltantes no se distribuyen uniformemente entre las distintas subpoblaciones, esta imputación podría terminar generando decisiones basadas en información distorsionada. Un ejemplo clásico de esta problemática es el sesgo hacia **minorías raciales** o **comunidades vulnerables**, que pueden tener datos faltantes o incompletos más a menudo que otros grupos, y cuya imputación incorrecta puede perpetuar la discriminación.

El **sesgo en los conjuntos de datos** también puede estar presente si los datos originales en sí mismos reflejan desigualdades históricas. Un claro ejemplo es el sesgo en los sistemas judiciales. Si se entrena un modelo de predicción de riesgo criminal utilizando datos históricos que ya están sesgados en contra de ciertos grupos raciales o étnicos, el modelo perpetuará estos sesgos. Al codificar estos datos sin una revisión crítica, se refuerzan patrones discriminatorios. En este contexto, la **codificación binaria** de categorías como “antecedentes penales” puede llevar a decisiones que impacten injustamente a grupos minoritarios que han sido desproporcionadamente afectados por sistemas de justicia previos.

La forma en que se codifican los datos también puede tener implicaciones en la **transparencia y explicabilidad** del modelo. Los modelos basados en **embeddings** o **representaciones distribuidas**, aunque poderosos para capturar patrones complejos, son a menudo opacos y difíciles de interpretar. Esto plantea problemas éticos, ya que las decisiones basadas en estos modelos pueden ser difíciles de justificar o auditar. En sectores como la **salud** o el **sector financiero**, donde las decisiones basadas en IA afectan directamente a las personas, la falta de explicabilidad puede minar la confianza pública y dificultar la identificación de sesgos ocultos en el modelo.

Para mitigar estos problemas, es fundamental que los equipos de desarrollo de modelos presten especial atención al **análisis de sesgo** durante el preprocesamiento de datos, asegurándose de que las técnicas de codificación no amplifiquen las desigualdades existentes. Además, es necesario realizar auditorías continuas del modelo para detectar posibles sesgos en el tiempo, ya que el comportamiento de los datos puede cambiar y afectar negativamente la equidad de las predicciones.

> **Ejemplo**: En un sistema de recomendación de empleo, si se utiliza una codificación binaria para características como el nivel educativo, es importante asegurar que esta codificación no excluya o penalice a personas de contextos socioeconómicos desfavorecidos que pueden no haber tenido acceso a ciertos niveles de educación formal, pero que poseen habilidades relevantes.

> **Ejemplo**: En un modelo de predicción de tasas de reincidencia criminal, un sesgo en los datos históricos podría hacer que el modelo recomiende sentencias más severas para ciertos grupos raciales. Si las variables de entrada están codificadas de manera que reflejan estos sesgos, el modelo continuará perpetuando la discriminación histórica.

> **Ejemplo**: En un sistema de diagnóstico médico basado en IA, una codificación incorrecta de síntomas o condiciones médicas en poblaciones minoritarias puede llevar a un sesgo en los resultados del diagnóstico, afectando negativamente a estos grupos al recibir tratamientos inadecuados o menos precisos.

**Para reflexionar...**

> **¿De qué manera pueden los científicos de datos mitigar los sesgos que pueden surgir durante la codificación de datos en modelos de machine learning?**
>
> **Pistas**: Reflexiona sobre la importancia de realizar auditorías regulares, seleccionar cuidadosamente las técnicas de codificación y considerar las implicaciones éticas de los datos antes de entrenar un modelo.
