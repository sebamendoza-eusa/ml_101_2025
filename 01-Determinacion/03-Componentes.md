# 3. Componentes de un proyecto de aprendizaje automático

Todo sistema de aprendizaje automático está compuesto de varios **elementos clave** que, en conjunto, permiten el desarrollo, ajuste y validación del modelo. Cada componente es esencial para lograr un modelo eficiente y preciso. A continuación, vamos a explorar en detalle los elementos principales.

Como se ha dicho, el éxito de un proyecto de aprendizaje automático va a depender de la integración adecuada de cada componente. Los datos deben estar bien representados, los algoritmos correctamente seleccionados y los modelos deben ser evaluados rigurosamente. La capacidad de generalización de un modelo es clave para su éxito en el mundo real, lo que exige un delicado equilibrio entre el ajuste a los datos de entrenamiento y la capacidad de responder bien a datos nuevos.

> [!important]
>
> **Datos-Algoritmo-Modelo: Cómo interaccionan**
>
> 1. **El algoritmo necesita los datos** para entrenar un modelo. Sin datos, no puede aprender ni ajustarse.
> 2. **Los datos pasan por el algoritmo**, que utiliza técnicas matemáticas para identificar patrones ocultos.
> 3. **El modelo entrenado** es el resultado del proceso de aprendizaje que el algoritmo ha realizado con los datos.
> 4. **Retroalimentación**: En muchos casos, el modelo generado se evalúa y ajusta con nuevos datos, permitiendo mejoras continuas.

## Datos

El **conjunto de datos** es la base sobre la que se construyen los modelos de ML. Estos datos, de diversa naturaleza, pueden ser numéricos, categóricos, imágenes, texto o señales de audio, y deben ser **representativos y de calidad** para evitar sesgos y mejorar la capacidad del modelo para generalizar a nuevos datos.

> **Ejemplo**: En la predicción de **rendimientos agrícolas**, los datos históricos del clima y el uso de fertilizantes se utilizan para entrenar un modelo capaz de predecir la cantidad de producción de cultivos en diferentes condiciones climáticas.

### Preprocesamiento de los datos

El **preprocesamiento de datos** es una etapa crítica en cualquier proyecto de machine learning, ya que garantiza la calidad y adecuación de los datos antes de que puedan ser utilizados por los algoritmos de aprendizaje. Este proceso incluye varias técnicas fundamentales que ayudan a transformar los datos crudos en un formato adecuado para ser procesado por un modelo.

#### **Limpieza de Datos**

La **limpieza de datos** se refiere a la identificación y corrección o eliminación de **valores perdidos, incorrectos o duplicados**. En muchos conjuntos de datos, es común encontrar valores faltantes o errores que pueden sesgar los resultados o impedir el correcto entrenamiento del modelo. Existen varias estrategias para manejar estos problemas, como **imputar** los valores faltantes (rellenarlos con promedios, medianas o con algún valor predicho) o simplemente eliminarlos si representan una porción pequeña del total de datos.

> **Ejemplo**: En un conjunto de datos médicos, si faltan valores en los registros de edad o sexo de algunos pacientes, es importante decidir si eliminar esos registros o reemplazar los valores faltantes con la media de los datos existentes.

#### **Escalado**

El **escalado de datos** es fundamental cuando los datos contienen variables con diferentes rangos o escalas. Por ejemplo, si una variable mide ingresos anuales en miles de dólares y otra mide edad en años, sus magnitudes pueden ser tan diferentes que afecten el rendimiento de ciertos algoritmos, como la regresión logística o las redes neuronales, que son sensibles a estas diferencias.

Existen diferentes técnicas de escalado, si bien dos técnicas son las más habituales. Se trata del **min-max scaling** (ajustar los valores entre 0 y 1) y el **escalado por estandarización** (ajustar los valores para que tengan media 0 y desviación estándar 1).

> **Ejemplo**: En un modelo que predice precios de viviendas, las variables como el tamaño en metros cuadrados y el número de habitaciones deben estar en escalas comparables para evitar que una domine sobre la otra.

#### **Codificación de Categorías**

En muchos casos, los datos contienen **variables categóricas** (por ejemplo, “bajo”, “medio” y “alto” o “rojo”, “verde” y “azul”) que deben ser transformadas en valores numéricos para que los modelos de machine learning puedan procesarlos. Existen varias técnicas para realizar esta conversión, siendo las más comunes la **codificación one-hot** (crear una columna binaria para cada categoría) o la **codificación ordinal** (asignar valores numéricos basados en un orden establecido).

> **Ejemplo**: En un modelo que predice si un cliente comprará un producto, la variable “nivel de ingresos” puede ser categórica (“bajo”, “medio”, “alto”). Para que el modelo la entienda, se podría convertir en 1, 2 y 3 respectivamente.

**Para reflexionar...**

> **¿Qué problemas pueden surgir si no se realiza una adecuada normalización de los datos?**
>
> Sin normalización, los modelos sensibles a las magnitudes de las variables, como las redes neuronales, podrían dar más peso a variables con mayores escalas, distorsionando los resultados.

> **¿Cuándo es preferible eliminar datos faltantes en lugar de imputarlos?**
>
> Considera la proporción de valores faltantes y si eliminarlos afectaría la representatividad del conjunto de datos o introduciría sesgos.

> **¿Qué diferencias existen entre las técnicas de codificación one-hot y ordinal, y cuándo es recomendable usar cada una?**
>
> Reflexiona sobre cómo las relaciones entre categorías influyen en la elección de la técnica adecuada, ya que la codificación ordinal asume un orden jerárquico, mientras que la one-hot trata cada categoría como independiente.

### Entrenamiento, test y validación

En un proyecto de machine learning es crucial separar los datos en **conjuntos de entrenamiento, validación y test**. Esta práctica asegura que el modelo se entrena correctamente, se evita el sobreajuste, y se evalúa de manera precisa su rendimiento. Cada conjunto cumple una función específica en el proceso de desarrollo del modelo. A continuación, detallamos estos tres conceptos clave:

#### **Conjunto de entrenamiento**

El **conjunto de entrenamiento** es el que se utiliza para ajustar los parámetros del modelo. Aquí, el modelo de ML aprende los patrones subyacentes en los datos. Durante el entrenamiento, el algoritmo ajusta sus parámetros internos (por ejemplo, los pesos en una red neuronal) con el fin de minimizar un error o maximizar un rendimiento, como la precisión o el rendimiento general en la tarea deseada.

> **Ejemplo:** En un problema de clasificación de imágenes con aprendizaje supervisado, el conjunto de entrenamiento incluiría imágenes etiquetadas (como “gato” o “perro”), y el modelo ajustaría sus parámetros para reconocer estas clases.

#### **Conjunto de validación**

El **conjunto de validación** se utiliza durante el entrenamiento para monitorizar el rendimiento del modelo y ajustar hiperparámetros (como la tasa de aprendizaje, el número de capas en una red neuronal, etc.). No se usa para entrenar el modelo directamente, sino para verificar su comportamiento en datos que no ha visto antes, pero que siguen siendo parte del proceso de optimización.

El conjunto de validación es fundamental para **evitar el sobreajuste** (overfitting), un problema común cuando el modelo aprende demasiado bien los detalles del conjunto de entrenamiento, y pierde capacidad de generalización en datos nuevos. Durante el proceso de entrenamiento, **si el rendimiento en el conjunto de entrenamiento es alto pero bajo en el de validación, es probable que el modelo esté sobreajustado.**

> **Ejemplo**: Durante el desarrollo de un modelo de predicción de ventas, el conjunto de validación se utilizaría para verificar si el modelo predice correctamente en diferentes meses o estaciones, ayudando a ajustar el modelo para mejorar su rendimiento en diferentes contextos temporales.

#### **Conjunto de test**

El **conjunto de test** es la evaluación final del modelo, y se utiliza **una vez completado el entrenamiento**. A diferencia del conjunto de validación, los datos de test son completamente nuevos para el modelo y permiten medir su rendimiento en un entorno simulado que represente datos reales en producción.

Una vez que el modelo ha sido ajustado utilizando el conjunto de validación, se aplica al conjunto de test para obtener una estimación precisa de su rendimiento general. **No se debe ajustar el modelo basándose en los resultados del conjunto de test**, ya que esto también podría generar sobreajuste.

> [!important]
>
> ##### El flujo de los datos en un proyecto ML...
>
> 1. **Entrenamiento**: Ajusta los parámetros del modelo.
> 2. **Validación**: Ayuda a ajustar hiperparámetros y monitorizar el sobreajuste.
> 3. **Test**: Proporciona una evaluación final del rendimiento.

### Cómo dividir el conjunto de datos de modo adecuado

Seleccionar adecuadamente los conjuntos de **entrenamiento**, **validación** y **test** es crucial para desarrollar modelos de machine learning robustos que generalicen correctamente a nuevos datos. Una cuestión que se nos planteará siempre a la hora de abordar un proyecto de machine learning es cómo dividir el conjunto de datos entre los datasets de entrenamiento, test y validación. Veamos algunas consideraciones generales.

#### Conjunto de entrenamiento
Este conjunto debería contener la mayoría de los datos (habitualmente entre el 60% y el 80%) ya que se utiliza para ajustar los parámetros del modelo. Es fundamental que los datos de entrenamiento reflejen la diversidad de casos posibles que el modelo encontrará en producción. Un conjunto de entrenamiento mal balanceado o poco representativo puede provocar que el modelo aprenda patrones incorrectos o sesgados.

#### Conjunto de validación
El conjunto de validación se integrará usualmente por un volumen de entre el 10% y el 20% de los datos originales. Este conjunto se utilizará para ajustar hiperparámetros y evaluar el rendimiento del modelo durante el proceso de desarrollo. También permite detectar problemas de **sobreajuste** (cuando el modelo se ajusta demasiado a los datos de entrenamiento y falla en generalizar). La **validación cruzada** es una técnica habitual donde **el conjunto de validación se rota**, utilizando diferentes porciones de los datos como validación en cada iteración, asegurando una evaluación más robusta.

#### Conjunto de test
Este conjunto, generalmente de entre el 10% y el 20% restante de los datos originales, evalúa el rendimiento final del modelo una vez completado su entrenamiento y ajuste. Los datos de test **no deben haber sido utilizados en ninguna parte del proceso de entrenamiento o validación**, ya que su objetivo es medir la capacidad de **generalización** del modelo a datos completamente nuevos. Un modelo que funcione bien en el conjunto de test es más probable que generalice bien en producción.

> [!important]
>
> Es esencial que los tres conjuntos sean representativos de los datos reales, contengan diversidad suficiente y no compartan instancias duplicadas o solapadas que puedan llevar a resultados de rendimiento “inflados”. 

> **Ejemplo**: En un proyecto de clasificación de imágenes, el 70% de las imágenes se pueden usar para entrenamiento, el 15% para validación y el 15% para test, asegurando que todas las clases estén equilibradas en cada conjunto.

**Para reflexionar...**

> **¿Qué problemas podrían surgir si no se utiliza un conjunto de validación?**
>
> Sin un conjunto de validación, el modelo podría sobreajustarse a los datos de entrenamiento y tener un mal rendimiento en datos nuevos.

> **¿Cómo asegurar que el conjunto de test refleja adecuadamente el entorno de producción?**
>
> Se deben elegir cuidadosamente los datos de test para que representen correctamente los casos de uso que el modelo encontrará en producción.

> **¿Qué riesgos podrían surgir al seleccionar conjuntos de entrenamiento, validación y test que no sean representativos?**
>
> Pueden generarse modelos sesgados o que no generalicen bien debido a una muestra no representativa de la población.

## Modelo

En general, un **modelo** es una representación simplificada de un sistema, proceso o fenómeno del mundo real. Los modelos se utilizan para comprender, predecir o controlar el comportamiento de sistemas complejos, y pueden ser expresados en diferentes formas, como matemáticas, diagramas, simulaciones o estructuras conceptuales.

Un modelo captura los aspectos esenciales de la realidad **mientras omite detalles innecesarios para un propósito específico**. Dependiendo del contexto, los modelos pueden ser descriptivos, prescriptivos o predictivos.

> **Ejemplo:** En física, un modelo puede describir la relación entre variables mediante ecuaciones matemáticas, mientras que en economía, un modelo puede predecir el comportamiento de mercados bajo ciertos supuestos.

**En machine learning**, un modelo es una representación matemática de un sistema que se utiliza para hacer predicciones o tomar decisiones basadas en datos. Intuitivamente, se puede imaginar como una "receta" que se aprende a partir de datos de entrenamiento. A través del aprendizaje de ejemplos previos, el modelo identifica patrones y reglas en los datos que luego le permiten predecir resultados para nuevos datos que no ha visto antes.

> **Ejemplo:** En un modelo para predecir el precio de una vivienda, el modelo aprende a relacionar características como el tamaño, la ubicación y el número de habitaciones con el precio. Luego, cuando recibe nuevos datos (una casa diferente con características similares), utiliza estas reglas aprendidas para estimar su precio. 

Todo modelo de machine learning toma **entradas** (datos de entrada) y produce **salidas** (predicciones o clasificaciones), con el objetivo de realizar inferencias sobre datos no observados. Este proceso implica encontrar la **función matemática** que mejor describe la relación entre las entradas y salidas, ajustando los parámetros del modelo para minimizar el **error de predicción**.

El **entrenamiento** del modelo consiste en iterar sobre un conjunto de datos etiquetados, ajustando los **parámetros** del modelo mediante un proceso de **optimización**, generalmente minimizando una **función de coste**. Esto asegura que el modelo sea capaz de generalizar, es decir, que no solo funcione bien con los datos de entrenamiento, sino que también sea preciso en los datos nuevos.

### Tipos de modelos en ML

En **machine learning**, los modelos se pueden clasificar en varias tipologías según el tipo de problema que abordan. Dos de las categorías más importantes son los **modelos de predicción** y los **modelos de clasificación**. Estos dos tipos de modelos se utilizan en diferentes situaciones según la naturaleza de las variables que estamos intentando predecir o clasificar.

#### Modelos de predicción

Los **modelos de predicción**, también conocidos como modelos de **regresión**, se utilizan cuando el objetivo es predecir un valor **continuo**. Estos modelos tratan de estimar una salida numérica a partir de los datos de entrada. Un ejemplo clásico es la **regresión lineal**, donde, para hacer predicciones, se modela la relación entre una variable dependiente y una o más variables independientes.

> **Ejemplo**: Un modelo de regresión lineal puede predecir el precio de una casa en función de características como el tamaño, el número de habitaciones o la ubicación.

#### Modelos de clasificación

Por otro lado, los **modelos de clasificación** se emplean cuando la salida es un valor **discreto** o una categoría. Estos modelos asignan una etiqueta o clase a las instancias de datos. Un ejemplo típico es el **clasificador de spam**, que clasifica los correos electrónicos en dos categorías: spam o no spam.

> **Ejemplo**: Un modelo de clasificación puede determinar si una imagen contiene un perro o un gato, asignando etiquetas "perro" o "gato".

#### Modelos de agrupación

Los **modelos de agrupación**, (o **clustering)**, se utilizan cuando el objetivo es **agrupar instancias sin etiquetas previas** en subconjuntos o **clusters** basados en su similitud. A diferencia de los modelos de predicción o clasificación, el clustering no busca predecir una etiqueta o valor, sino identificar patrones o estructuras ocultas dentro de los datos. Un ejemplo clásico es el **K-means**, donde el algoritmo divide los datos en $k$ grupos según la proximidad entre puntos. El objetivo es que los puntos dentro de un mismo grupo sean más similares entre sí que a los de otros grupos.

> **Ejemplo**: Un modelo de clustering puede agrupar a los clientes de una tienda en línea en diferentes segmentos según su comportamiento de compra, como la frecuencia de compras, el tipo de productos adquiridos y el monto gastado, permitiendo a la empresa personalizar campañas de marketing para cada grupo.

> [!important]
>
> - Los **modelos de predicción** generan resultados continuos, como un valor numérico.
> - Los **modelos de clasificación** generan resultados discretos o categóricos.

Cada tipo de modelo tiene aplicaciones específicas y se utiliza en problemas diferentes, dependiendo del formato de la salida que se desea obtener.

### Complejidad de los modelos de ML

Los modelos de ML pueden clasificarse según su complejidad.

En el contexto del aprendizaje automático, la distinción entre **modelos lineales** y **modelos no lineales** es crucial para entender cómo diferentes técnicas abordan el problema de modelar relaciones entre variables. Representan dos tipos diferentes de enfoques para modelar la relación entre las entradas (características) y las salidas (predicciones). La elección entre estos modelos dependerá evidentemente de la naturaleza del problema y los datos.

#### Modelos lineales

Un **modelo lineal** asume que la relación entre las variables de entrada (o características) y la salida es una combinación lineal de las entradas. Matemáticamente, se expresa como:

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$


donde:

- $y$ es la salida (la predicción),
- $w_0$ es el **término independiente**
- $w_1, w_2, \dots, w_n$ son los **coeficientes** o pesos asociados a las variables de entrada.
- $x_1, x_2, \dots, x_n$ son las variables de entrada (o características).

El modelo lineal ajusta estos parámetros ($w_0, w_1, \dots, w_n$) durante el entrenamiento, minimizando el error entre las predicciones $y$ y los valores reales. Los modelos lineales son fáciles de interpretar y entrenar, pero su capacidad para modelar relaciones complejas es limitada.

##### Ventajas e inconvenientes de los modelos lineales

###### **Ventajas**:

- Simplicidad y facilidad de interpretación: se puede ver fácilmente cómo cada entrada influye en la salida.
- Rápidos de entrenar y eficientes en problemas con relaciones sencillas entre las variables.

###### **Desventajas**:

- No pueden capturar relaciones complejas o no lineales entre las variables.
- Suponen que los efectos combinados de las entradas son aditivos y proporcionales, lo que limita su flexibilidad.

#### Modelos no lineales

En contraste, los **modelos no lineales** permiten que las relaciones entre las entradas y la salida sean mucho más complejas. Los modelos no lineales pueden capturar interacciones y relaciones no triviales que un modelo lineal no puede. Una función no lineal puede tomar muchas formas, como por ejemplo:

$$
y = w_0 + w_1 x_1^2 + w_2 \sin(x_2) + w_3 e^{x_3}
$$

Este tipo de modelos puede incluir términos polinómicos, funciones trigonométricas, exponenciales, o incluso relaciones definidas por redes neuronales. Los modelos no lineales son muy flexibles y pueden ajustar patrones complejos en los datos, pero también tienen mayor riesgo de **sobreajuste** si no se gestionan adecuadamente.

##### Ventajas y Desventajas de los Modelos No Lineales

###### **Ventajas**:

- Mayor capacidad para capturar relaciones complejas en los datos.
- Flexibilidad para adaptarse a una amplia variedad de patrones, independientemente de la forma que tomen.

###### **Desventajas**:

- Requieren más datos para entrenarse de manera eficaz, ya que son más propensos al **sobreajuste** (overfitting).
- Más difíciles de interpretar, lo que puede ser un problema en contextos donde se requiere explicabilidad del modelo.
- Típicamente más costosos computacionalmente.



> [!important]
>
> **Modelos lineales**: adecuados cuando las relaciones entre las variables son aproximadamente lineales. Son interpretables, fáciles de entrenar y computacionalmente baratos.
>
> **Modelos no lineales**: se utilizan cuando hay una relación compleja entre las características y la variable objetivo. Tienen mayor capacidad predictiva, pero son más difíciles de interpretar y entrenar, y a menudo requieren más datos para evitar sobreajuste.



> **Ejemplo:** Supongamos que queremos predecir el **precio de una vivienda** en función de una característica simple, como el **tamaño** de la casa ($x$). Un **modelo lineal** podría asumir que el precio ($y$) depende linealmente del tamaño de la vivienda, es decir:
> 
> $$
> y = w_0 + w_1 x
> $$
> 
> En este caso, el modelo asume que el precio aumenta proporcionalmente con el tamaño. Esto refleja una **relación lineal** entre el tamaño de la casa ($x$) y el precio ($y$), donde $w_0$ es el término independiente y $w_1$ es la pendiente de la recta.
>
> Sin embargo, el precio de una vivienda no suele depender únicamente de su tamaño. Factores más complejos, como la **ubicación**, el **número de habitaciones** y la **antigüedad** de la vivienda, pueden influir de forma no lineal. En este caso, un **modelo no lineal** sería más adecuado. Por ejemplo, un modelo podría tomar la forma:
>
> $$
> y = w_0 + w_1 x_1 + w_2 x_2^2 + w_3 \log(x_3)
> $$
>
> Aquí, la **ubicación** ($x_1$) tiene una relación lineal con el precio, el **número de habitaciones** ($x_2$) tiene una relación cuadrática, y la **antigüedad** ($x_3$) afecta de manera logarítmica al precio. Este tipo de modelo no lineal permitiría capturar mejor las interacciones complejas entre las distintas características de las viviendas.
>

> **Ejemplo:** Un ejemplo típico de **modelo No Lineal** sería el del uso de **redes neuronales profundas** para el **reconocimiento de imágenes**. En este caso, las relaciones entre los píxeles de una imagen y la etiqueta de la clase (por ejemplo, "gato" o "perro") son altamente no lineales, y el modelo debe aprender patrones complejos en las imágenes.

**Para reflexionar...**

> **¿Cuándo preferirías utilizar un modelo lineal sobre un modelo no lineal, y qué implicaciones tiene esa elección en términos de rendimiento y complejidad?**
>
> Reflexiona sobre cómo la simplicidad del modelo afecta la capacidad de generalización y el riesgo de sobreajuste.

> **¿En qué casos podría un modelo lineal ser insuficiente para capturar la relación entre las variables de entrada y salida?** 
>
> Reflexiona sobre situaciones en las que las entradas tienen efectos no aditivos o proporcionales en la salida, como en datos con interacciones complejas.

> **¿Cómo afecta la capacidad de generalización al comparar modelos lineales y no lineales?**
>
> Considera cómo un modelo lineal tiende a generalizar bien en conjuntos de datos pequeños, mientras que un modelo no lineal podría requerir más datos y ser propenso al sobreajuste.

### Entrenamiento y generalización

Durante el entrenamiento, el modelo ajusta sus parámetros para **minimizar el error** en las predicciones, utilizando algún algoritmo de optimización. El objetivo es lograr que el modelo no solo memorice los datos de entrenamiento, sino que sea capaz de **generalizar** a datos nuevos y no vistos previamente. Si el modelo es demasiado simple, puede sufrir de **subajuste** (no capturar adecuadamente los patrones), y si es demasiado complejo, puede **sobreajustar** (memorizar los datos sin generalizar).

Así pues, el ajuste de la complejidad del modelo en **machine learning** es esencial. Dos de las técnicas más comunes para gestionar este equilibrio son la **regularización** y la **validación cruzada**. Aunque ya se ampliaran estos conceptos en capítulos próximos, de momento podemos entender la  **regularización** como una técnica que introduce penalizaciones en el cálculo de los parámetros del modelo para evitar que se ajusten demasiado a los datos de entrenamiento, lo que mejora la capacidad de generalización. Por su parte, la **validación cruzada** es otra técnica que trata de dividir los datos en subconjuntos para entrenar y evaluar el modelo de manera repetida, asegurando que el rendimiento se mida de manera consistente en diferentes partes de los datos.

**Para reflexionar...**

> **¿Cómo influye la complejidad del modelo en su capacidad para generalizar a nuevos datos?**
>
> Reflexiona sobre el equilibrio entre un modelo que es lo suficientemente complejo para capturar patrones importantes, pero no tan complejo que se ajuste demasiado a los datos de entrenamiento.

> **¿Por qué un modelo lineal puede no ser adecuado para problemas con relaciones no lineales?**
>
> Considera ejemplos donde las variables tienen relaciones complejas o interacciones que no se pueden capturar con una simple línea recta.

### Evaluación del modelo

La **evaluación de modelos** es una etapa crítica en cualquier proyecto de **machine learning**. Su objetivo es medir el rendimiento del modelo y su capacidad para **generalizar** correctamente a partir de datos de entrada **que no ha visto antes**. Si un modelo se ajusta bien a los datos de entrenamiento pero falla al enfrentarse a nuevos datos no es útil en la práctica. Aquí es donde entran en juego las **métricas de evaluación**, que proporcionan **una forma cuantitativa** de medir el rendimiento del modelo.

Una **métrica** es una función que mide algún aspecto del rendimiento del modelo. Según el tipo de problema (clasificación, regresión, etc.), diferentes métricas se utilizan para proporcionar información sobre la calidad de las predicciones.

#### Métricas en modelos de predicción

Las métricas **Error Cuadrático Medio (MSE)** y **Error Absoluto Medio (MAE)** son dos de las más comunes utilizadas para evaluar el rendimiento de los **modelos de regresión** en machine learning. Ambas miden cuán cerca están las predicciones realizadas por un modelo de los valores reales, pero lo hacen de maneras diferentes, lo que las hace útiles en distintos contextos.

El **MSE** calcula el promedio de los errores al cuadrado entre las predicciones del modelo y los valores reales. La fórmula de cálculo sería la siguiente:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Donde:

- $y_i$ es el valor real,
- $\hat{y}_i$ es el valor predicho por el modelo,
- $n$ es el número de observaciones.

El MSE penaliza fuertemente los **errores grandes** debido al uso del cuadrado de las diferencias. Esto significa que es más sensible a valores atípicos, ya que los errores grandes tendrán un impacto mayor en la métrica final. El resultado está en unidades cuadradas del valor objetivo, lo que puede hacer que sea menos interpretable que otras métricas.

Por otro lado, el **MAE**, mide el promedio de los errores absolutos entre las predicciones y los valores reales. Su fórmula sería la siguiente:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

A diferencia del MSE, **el MAE no utiliza el cuadrado de las diferencias, sino el valor absoluto**. Esto significa que penaliza **todos los errores por igual**, sin importar si son grandes o pequeños. Es menos sensible a los valores atípicos que el MSE y su resultado se interpreta directamente en las mismas unidades que el valor objetivo, lo que a veces lo hace más intuitivo.

Ambas métricas miden el **error promedio** entre las predicciones y los valores reales y son,  como se ha comentado, utilizadas en **modelos de regresión** para evaluar la precisión de las predicciones.

Sin embargo, El **MSE** penaliza los errores grandes más que el **MAE** debido al cuadrado de las diferencias, lo que lo hace **más sensible a valores atípicos**, mientras que el **MAE** penaliza los errores de manera uniforme.  También El **MAE** es **más interpretable**, ya que se mide en las mismas unidades que el valor objetivo, mientras que el MSE se mide en unidades cuadradas.

> [!important]
>
> Normalmente, el **error cuadrático medio (MSE)** es la función de pérdida preferida en la optimización de modelos predictivos frente al error absoluto medio (MAE). La razón crucial es que el MSE es una **función de pérdida suave y continuamente diferenciable**. Esta propiedad es vital para los algoritmos de **optimización basados en el gradiente**, que son el motor del aprendizaje automático. La suavidad permite un cálculo del gradiente lineal, lo que garantiza un movimiento estable y eficiente hacia el mínimo. El MAE, en contraste, presenta un **punto no diferenciable en cero**, lo que introduce inestabilidad y complicaciones computacionales para los métodos de gradiente estándar.
>
> También es importante tener en cuenta que minimizar el MSE es **matemáticamente equivalente** a maximizar la verosimilitud de un modelo donde se asume que el error de observación o ruido sigue una **distribución gaussiana**. Esta equivalencia proporciona un marco robusto para la modelización probabilística.
>
> Adicionalmente, el MSE **penaliza cuadráticamente los errores grandes**, forzando al modelo a reducir la magnitud de los *outliers* de manera más agresiva que el MAE, un comportamiento a menudo deseable cuando los errores mayores conllevan un riesgo más significativo.

> **Ejemplo:** En un modelo de predicción de precios de acciones, donde los grandes errores podrían tener un impacto financiero significativo, podría preferirse usar el **MSE** para penalizar esos errores grandes. Sin embargo, en un modelo de predicción de la demanda de energía eléctrica diaria, el **MAE** se utiliza para interpretar fácilmente los errores y ser más resistente a valores atípicos ocasionales.

#### Métricas en modelos de clasificación: Matriz de confusión

La **matriz de confusión** es una herramienta fundamental en la evaluación de modelos de clasificación. Permite visualizar el rendimiento del modelo al mostrar las predicciones realizadas frente a los valores reales. Se organiza en una matriz de 2x2 (en el caso más simple de problemas binarios), con las siguientes categorías:

|                   | Predicción Positiva       | Predicción Negativa       |
| ----------------- | ------------------------- | ------------------------- |
| **Real Positivo** | Verdaderos Positivos (TP) | Falsos Negativos (FN)     |
| **Real Negativo** | Falsos Positivos (FP)     | Verdaderos Negativos (TN) |

- **Verdadero Positivo (TP)**: Predicciones correctas de la clase positiva (predicho positivo y es realmente positivo).
- **Falso Positivo (FP)**: Predicciones incorrectas de la clase positiva (predicho positivo pero es realmente negativo).
- **Verdadero Negativo (TN)**: Predicciones correctas de la clase negativa (predicho negativo y es realmente negativo).
- **Falso Negativo (FN)**: Predicciones incorrectas de la clase negativa (predicho negativo pero es realmente positivo).

Algunas de las métricas más habituales asociadas a la matriz de confusión son:

###### Exactitud (Accuracy)

Indica el porcentaje de predicciones correctas en relación con el total de predicciones realizadas. Es útil cuando el coste de las predicciones incorrectas es similar para todas las clases, pero puede ser engañosa en problemas con clases desbalanceadas.

$$
\text{Exactitud} = \dfrac{TP + TN}{TP + TN + FP + FN}
$$

> **Ejemplo:** Un modelo de reconocimiento facial identifica correctamente 98 de 100 rostros, logrando una exactitud del 98%.

###### Precisión (Precision)

La **precisión** mide el porcentaje de casos **predichos como positivos** que **realmente son positivos**. Es útil cuando nos importa conocer la proporción de verdaderos positivos entre todos los elementos que el modelo ha clasificado como positivos. Su fórmula es:

$$
\text{Precisión} = \dfrac{TP}{TP + FP}
$$

> **Ejemplo**: En un clasificador de detección de fraudes, una alta precisión significa que la mayoría de las transacciones etiquetadas como fraudulentas son efectivamente fraudes.

###### Sensibilidad (Sensivity o Recall)

Mide la capacidad del modelo para detectar correctamente las verdaderas instancias positivas. Es especialmente útil cuando es crítico identificar todos los positivos, como en la detección de enfermedades.

$$
\text{Recall} = \dfrac{TP}{TP + FN}
$$

> **Ejemplo**: En la detección de cáncer, es vital que el modelo detecte todos los casos de cáncer (recall alto), aunque ocasionalmente marque falsos positivos.

###### F1-Score

Es la media armónica entre precisión y recall. Es útil en situaciones con datos desbalanceados, donde es importante equilibrar ambas métricas.

$$
\text{F1} = 2 \times \dfrac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}
$$

> **Ejemplo**: En un sistema de clasificación de spam, donde es importante tanto detectar correctamente los correos no deseados como minimizar el número de correos válidos etiquetados como spam, el F1-Score sería una métrica clave.

> [!tip]
>
> Si tenemos en cuenta los conceptos de precisión y *recall* podemos encontrar cuatro escenarios posibles que es necesario manejar adecuadamente
>
> - **Alta precisión y alto recall**: El modelo maneja perfectamente esa clase
> - **Alta precisión y bajo recall**: El modelo no detecta la clase muy bien, pero cuando lo hace es altamente confiable.
> - **Baja precisión y alto recall**: La clase detecta bien la clase pero también incluye muestras de otras clases.
> - **Baja precisión y bajo recall:** El modelo no logra clasificar la clase correctamente.
>
> Cuando tenemos un dataset con clases no balanceadas, suele ocurrir que obtenemos un **alto valor de precisión en la clase mayoritaria y un bajo recall en la clase minoritaria**

##### Eleccion de la métrica en problemas de clasificación

Visto lo anterior surge la cuestión de cómo elegir la **métrica adecuada** en un problema de **machine learning**. Ello va a depender del contexto del problema y del objetivo del modelo. 

Si las clases están **desbalanceadas**, es decir, una clase tiene muchos más ejemplos que la otra (por ejemplo, en la detección de fraudes, donde los casos fraudulentos son mucho menos frecuentes), métricas como la **exactitud** pueden ser engañosas. En este caso, es mejor utilizar métricas que se centren en los casos positivos, como la **sensibilidad** o la **precisión**.

Tambien será importante considerar hasta qué punto los **falsos positivos** o los **falsos negativos** son más críticos para el problema. La **sensibilidad** es crítica cuando los **falsos negativos** tienen un costo elevado, como en el diagnóstico de enfermedades (donde es preferible detectar todos los casos). Sin embargo la **precisión** es más importante cuando los **falsos positivos** son costosos, como en la detección de fraudes (donde marcar transacciones legítimas como fraudulentas puede dañar la confianza del cliente).

Si queremos considerar en el mismo plano precisión y sensibilidad, es útil utilizar una métrica que equilibre a ambas. Para ello se dispone del **F1-Score**, que como hemos comentado es la media armónica entre precisión y sensibilidad y útil cuando se busca un balance equilibrados.

Finalmente, cuando las clases están bien representadas en los datos y los errores tienen un costo similar, la **exactitud** puede ser una métrica adecuada, ya que proporciona una visión global del rendimiento del modelo.

> **Ejemplos de elección de métrica:**
>
> **Detección de cáncer (problema crítico de salud)**: La **sensibilidad** es clave, ya que es preferible detectar todos los posibles casos de cáncer (minimizar los falsos negativos), aunque se generen más falsos positivos.
>
> **Sistema de recomendaciones**: La **precisión** es más importante, ya que recomendar productos irrelevantes puede reducir la satisfacción del usuario (minimizar los falsos positivos).
>
> **Spam en correos electrónicos**: El **F1-Score** puede ser útil si se busca un balance entre detectar spam (recall) y evitar que correos válidos se marquen como spam (precisión).

**Para reflexionar...**

> **¿Cuándo es más relevante el uso del F1-Score frente a otras métricas como la precisión?**
>
> Reflexiona sobre la importancia del equilibrio entre precisión y recall, especialmente en problemas con clases desbalanceadas.

> **¿Qué riesgos puede haber al usar solo la precisión como métrica en problemas desbalanceados?**
>
> Piensa en cómo un alto valor de precisión puede ocultar un rendimiento pobre en la identificación de la clase minoritaria.

Puedes ver estos vídeos. Explican bastante bien los conceptos vistos anteriormente:

- https://www.youtube.com/watch?v=haEWWO0b42Y - Concepto de matriz de confusión
- https://www.youtube.com/watch?v=H8FSfqxRWmA - Métricas asociadas a la matriz de confusión

## Algoritmos

Un **algoritmo de aprendizaje automático** es un conjunto formal de procedimientos matemáticos y reglas que permiten a un sistema aprender patrones a partir de los datos, con el fin de hacer predicciones, clasificaciones o tomar decisiones. Su base teórica hay que buscarla en la estadística, el álgebra lineal y la optimización matemática. El objetivo de cualquier algoritmo en aprendizaje automático es minimizar una **función de coste** (o también una **función de pérdida**) que representa el **error** del modelo al realizar sus objetivos marcados. La diferencia fundamental entre función de pérdida y función de coste estriba en que la primera mide el error en una sola predicción, mientras que la segunda se configura a partir de la agregación de los errores (generalmente el promedio) en todas las instancias de los datos de entrenamiento.

### Fundamento matemático
Los algoritmos de machine learning se fundamentan en principios matemáticos que optimizan las funciones objetivo, ajustando los parámetros del modelo para minimizar errores y maximizar el rendimiento. Como hemos comentado más arriba, la mayoría de estos algoritmos se basan en la **minimización de una función de coste** o **pérdida**.

> Ejemplo: En la regresión lineal, el objetivo es encontrar los coeficientes de una función que minimice la suma de los errores cuadráticos entre las predicciones y las observaciones reales.

El concepto de **derivada** es crucial en machine learning porque permite calcular los cambios en la función objetivo en respuesta a modificaciones en los parámetros del modelo. Esto es esencial para entrenar modelos y minimizar errores en predicciones. La derivada de una función mide la pendiente o el ritmo de cambio de la función en un punto. En machine learning, la función de pérdida (o en su caso de coste), que mide el error entre las predicciones del modelo y los valores reales se optimiza encontrando los parámetros del modelo que minimicen esta función. Las derivadas parciales respecto a cada parámetro indican cómo cambiar cada parámetro para reducir el error.

A partir del concepto de **derivada parcial** podemos obtener el de **gradiente**. El **gradiente** es una de las herramientas más utilizadas en optimización. Es un **vector** que contiene las derivadas parciales de la función objetivo con respecto a cada uno de los parámetros del modelo.

El concepto de gradiente puede aplicarse en el algoritmo denominado **gradiente descendente**. Este algoritmo es uno de los algoritmos de optimización más utilizados en entrenamiento de modelos de machine learning.

> **Ejemplo**: En un modelo de regresión lineal, la derivada de la función de pérdida con respecto a los coeficientes indica si se deben aumentar o disminuir los coeficientes para acercarse a la solución óptima.

> [!tip]
>
> Las derivadas también están presentes en el campo de las redes neuronales, ya que son fundamentales para el algoritmo de **retropropagación** (*backpropagation*). Este algoritmo calcula cómo los errores se propagan a través de las capas de la red. Usando derivadas, el algoritmo ajusta los pesos en cada capa para mejorar la precisión del modelo.

**Para reflexionar...**
> **¿Cómo afecta el cálculo incorrecto del gradiente a la convergencia de un algoritmo de optimización?**
>
> Un gradiente mal calculado podría dirigir el modelo en la dirección incorrecta, empeorando el rendimiento.

### Parámetros vs. datos disponibles

El número de parámetros de un modelo define su **complejidad**. Si el modelo tiene muchos parámetros, puede ajustar mejor los datos de entrenamiento, lo que aumenta su capacidad para capturar patrones. Sin embargo, si el número de datos disponibles no es suficiente en relación con la cantidad de parámetros del modelo, esto puede generar **sobreajuste** (**overfitting**).

El **sobreajuste** ocurre cuando un modelo se ajusta tan bien a los datos de entrenamiento que también aprende el ruido y las variaciones aleatorias en los datos. Esto significa que, aunque su desempeño en el conjunto de entrenamiento es excelente, no generaliza bien en datos nuevos, ya que está "demasiado especializado" en el conjunto de entrenamiento.

Por otro lado, un modelo con **pocos parámetros** puede ser demasiado simple para representar adecuadamente las relaciones presentes en los datos. En este caso, el modelo sufriría de **subajuste** (**underfitting**), lo que significa que no captura correctamente los patrones subyacentes de los datos. Este fenómeno lleva a un mal rendimiento tanto en el entrenamiento como en la validación. 

Una regla empírica comúnmente usada en entrenamiento de modelos sugiere que, para estos generalicen bien, se necesita al menos **10 veces más instancias de datos que parámetros**. Es decir, si un modelo tiene 1,000 parámetros, se recomienda tener al menos 10,000 instancias de datos. Si no se puede alcanzar esta relación óptima, se deberán utilizar técnicas de **regularización**, que reducirán la complejidad del modelo y así evitar el sobreajuste.

> **Ejemplo:** Imagina que entrenas un modelo para predecir el precio de casas. Un modelo simple (como una regresión lineal con un solo parámetro, basado solo en el tamaño de la casa) puede sufrir de **subajuste**, ya que no captura otros factores como la ubicación o el número de habitaciones. Por otro lado, un modelo muy complejo (una red neuronal profunda con muchos parámetros) puede **sobreajustar** los datos de entrenamiento, reconociendo detalles irrelevantes y no generalizando bien a datos de nuevas casas.

### Optimización y Ajuste
El ajuste de modelos en *machine learning* implica encontrar el conjunto óptimo de parámetros que minimice una función de pérdida o costo, lo que se conoce como **optimización**. Este proceso de optimización busca maximizar el rendimiento del modelo al aprender de los datos de entrenamiento y generalizar a datos nuevos. Existen distintos métodos de optimización utilizados para ajustar modelos, y entre los más comunes podemos encontrar **algoritmo de mínimos cuadrados ordinario** o el **gradiente descendente** entre otros

#### Algoritmo de mínimos cuadrados

El más simple de todos los algoritmos de optimización es sin duda el **algoritmo de mínimos cuadrados ordinarios (OLS)**. El algoritmo de mínimos cuadrados es una técnica utilizada para optimizar modelos en **problemas de regresión**. Su objetivo es encontrar la relación entre una o más variables independientes (también llamadas **características** o *features*) y una variable dependiente (la **salida** o resultado), y que minimice la suma de los errores entre las predicciones del modelo y los valores observados, lo que se conoce como error cuadrático (SSE)

En una **regresión lineal simple**, el modelo que se ajusta es de la forma:

$$
y = \theta_0 + \theta_1 x
$$

Donde:
- $y$ es el valor que queremos predecir.
- $x$ es la característica independiente.
- $\theta_0$ es el término independiente.
- $\theta_1$ es la pendiente o coeficiente de $y$.

El objetivo de OLS es encontrar los valores óptimos de $\theta_0$ y $\theta_1$ que minimicen la **suma de los errores cuadráticos** (SSE), donde el error es la diferencia entre los valores observados $y_i$ y las predicciones $\hat{y}_i$.

$$
SSE = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

El SSE es la suma de los cuadrados de los errores y se minimiza resolviendo el sistema de ecuaciones lineales que surge de las derivadas parciales con respecto a $\theta_0$ y $\theta_1$. Para minimizar la suma de errores, calculamos las derivadas parciales de la función de coste respecto a los parámetros $\theta_0$ y $\theta_1$ y las igualamos a cero. Esto da lugar a las **ecuaciones normales**:

$$
\theta = (X^T X)^{-1} X^T y
$$

Donde:
- $\theta$ es el vector de parámetros que incluye $\theta_0$ y $\theta_1$.
- $X$ es la matriz de las características (con una columna de 1's para el término independiente).
- $y$ es el vector de los valores observados.
- $X^T$ es la traspuesta de la matriz $X$.
- $(X^T X)^{-1}$ es la inversa de $X^T X$, siempre que exista.

Las ecuaciones de normalización son una forma algebraica de calcular los parámetros óptimos $\theta$ que minimizan el error cuadrático medio entre las predicciones del modelo y los valores reales. En lugar de usar un algoritmo **iterativo**, estas ecuaciones permiten calcular $\theta$ directamente en un solo paso. Es importante tener en cuenta que esta ecuación funciona únicamente cuando $X^T X$ es invertible.

En términos geométricos, OLS **ajusta la línea que pasa más cerca de todos los puntos en el espacio de características**, reduciendo al mínimo las distancias verticales al cuadrado desde cada punto a la línea.

> **Ejemplo**: Supongamos que estamos estudiando la relación entre el **tamaño de una casa (en metros cuadrados)** y su **precio (en miles de euros)**. Los datos de entrenamiento podrían ser los siguientes:
> $$
> \begin{array}{|c|c|}
> \hline
> \text{Tamaño (m²)} & \text{Precio (€)} \\
> \hline
> 50 & 200 \\
> 60 & 250 \\
> 70 & 300 \\
> \hline
> \end{array}
> $$
> Aplicando el algoritmo de OLS, primero formamos la matriz $X$ (incluyendo un término de 1 para el término independiente) y el vector $y$:
>
> $$
> X = \begin{bmatrix}
> 1 & 50 \\
> 1 & 60 \\
> 1 & 70
> \end{bmatrix}
> , \quad y = \begin{bmatrix}
> 200 \\
> 250 \\
> 300
> \end{bmatrix}
> $$
> 
>
> A continuación, resolvemos las ecuaciones normales para obtener los valores óptimos de $\theta_0$ (término independiente) y $\theta_1$ (pendiente). Al hacerlo, podemos generar la ecuación de la recta que mejor se ajusta a los datos, permitiendo hacer predicciones sobre los precios de las casas basados en su tamaño.
>

#### El gradiente descendente

El **gradiente descendente** es uno de los algoritmos de optimización más utilizados en *machine learning*, especialmente en problemas de aprendizaje supervisado. Este algoritmo ajusta los parámetros del modelo iterativamente en la dirección opuesta al gradiente de la función de costo, buscando minimizar dicha función. A cada paso, el algoritmo evalúa la derivada de la función de costo con respecto a los parámetros actuales del modelo. La clave del éxito del gradiente descendente radica en el **tamaño del paso** (también llamado tasa de aprendizaje): un paso pequeño puede hacer que el proceso sea lento, mientras que un paso demasiado grande puede hacer que el algoritmo no converja.

> [!tip]
>
> En este enlace puedes encontrar un vídeo muy interesante que explica este concepto de forma muy intuitiva:
> https://youtu.be/A6FiCDoz8_4?si=YyE-SEakC4Yqj4xj

