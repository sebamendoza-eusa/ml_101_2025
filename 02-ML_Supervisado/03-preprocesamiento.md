# 3. Preprocesamiento de datos en machine learning supervisado

## Introducción

> [!tip]
>
> En el aprendizaje supervisado, el objetivo del preprocesamiento es preparar los datos en un formato óptimo para ser utilizados por los modelos, aumentando la precisión, robustez y generalización de los mismos. Este proceso incluye la limpieza, transformación y reducción de datos, lo que permite maximizar el rendimiento de algoritmos y mitigar posibles fuentes de ruido o sesgo.

Vimos en el capítulo anterior que el análisis exploratorio de datos era un paso previo y complementario al preprocesamiento en el pipeline de machine learning supervisado. Durante el EDA, se exploraban patrones, distribuciones y relaciones en los datos, lo cual permitía identificar características que pueden influir en el rendimiento del modelo, como valores atípicos, datos faltantes y correlaciones entre variables. También vimos cómo estos hallazgos eran esenciales para orientar el preprocesamiento, ya que facilitaban la selección de técnicas de limpieza, transformación y reducción de dimensionalidad adecuadas. Es importante insistir en que análisis exploratorio y preprocesamiento van a ir de la mano: mientras el análisis exploratorio revela las necesidades específicas del conjunto de datos, el preprocesamiento las implementa, optimizando la estructura y calidad de los datos antes de alimentar el modelo supervisado.

Si hemos de caracterizar rápidamente la fase de preprocesamiento de datos se podría decir que, en cualquier proyecto de machine learning,  es una fase en la que el objetivo es transformar datos en bruto en un formato que maximice el rendimiento y la robustez del modelo. En el aprendizaje supervisado, este proceso adquiere una importancia aún mayor, pues la calidad y estructura de los datos afectan directamente a la precisión y generalización de los modelos. Sin un adecuado preprocesamiento, los algoritmos pueden sufrir de sobreajuste, subajuste o, en algunos casos, resultar ineficientes y poco precisos.

La fase de preprocesamiento no tiene un carácter uniforme y puede variar en función del tipo de datos y el modelo de machine learning a utilizar. Sin embargo, ciertos pasos son comunes y esenciales para casi todos los tipos de datos y algoritmos. En primer lugar, lo normal es abordar la **limpieza de datos**. La limpieza permite identificar y corregir inconsistencias como valores faltantes, atípicos o ruido, que pueden distorsionar los resultados del modelo. Los valores faltantes son particularmente problemáticos en el aprendizaje supervisado, ya que pueden llevar a un sesgo o a una pérdida de información significativa. Veremos como las técnicas para tratar estos valores incluyen la imputación mediante medias o medianas, la eliminación de observaciones o el uso de algoritmos avanzados de imputación.

Además de lo anterior,  hay que destacar que el preprocesamiento de datos en machine learning no solo incluye la limpieza y transformación de datos, sino que también abarca el tratamiento de un desafío habitual en muchos problemas prácticos que es el de los **datos desbalanceados**. Cuando se tiene una disparidad significativa entre la representación de las clases en un conjunto de datos, los modelos supervisados tienden a favorecer la clase mayoritaria, disminuyendo así su capacidad para identificar patrones en la clase minoritaria. Esta fase de ajuste en el preprocesamiento permite implementar técnicas como el sobremuestreo, el submuestreo o la generación de datos sintéticos para equilibrar el conjunto de datos, asegurando que todas las clases estén adecuadamente representadas. Además, el uso de pesos de clase en algunos modelos permite que las clases menos frecuentes tengan un impacto justo en el entrenamiento, logrando modelos más precisos y capaces de generalizar correctamente en escenarios de clasificación desbalanceada.

Una vez limpio, el siguiente paso habitual es realizar una **transformación adecuada de los datos** para que sean compatibles con los algoritmos de aprendizaje. La transformación abarca técnicas como el escalado, la normalización o la codificación de variables categóricas. El escalado y la normalización son necesarios en algoritmos que dependen de la magnitud de las variables, como aquellos basados en distancias. Por su parte, la codificación permite transformar variables categóricas en representaciones numéricas, facilitando su uso en modelos que solo operan con datos cuantitativos.

Además de la limpieza y transformación, en muchos casos es necesario aplicar un proceso de **reducción de dimensionalidad** con el objeto de simplificar el conjunto de datos sin perder información significativa. Este paso es crítico cuando se trabaja con grandes cantidades de características, ya que ello no solo reduce el tiempo de cómputo, sino que también ayuda a mejorar la capacidad de generalización del modelo al eliminar variables redundantes o poco relevantes.

Por último, aunque no forma parte estricta del preprocesamiento, la **división del conjunto de datos** es una fase esencial dentro del pipeline de machine learning supervisado, enfocada en la validación y evaluación del modelo. Esta fase implica segmentar los datos en subconjuntos de entrenamiento, validación y prueba, lo que asegura que el modelo aprenda en un conjunto de datos específico, ajuste sus hiperparámetros en un segundo subconjunto, y finalmente, sea evaluado en un conjunto independiente. Esta división es clave para estimar la capacidad de generalización del modelo en datos nuevos y prevenir el sobreajuste. La validación cruzada (cross-validation) es una técnica común que permite una evaluación más robusta, utilizando múltiples particiones para reducir la varianza en los resultados.

> [!tip]
>
> El preprocesamiento de datos es un proceso iterativo que depende de las características específicas del conjunto de datos y del problema a resolver. Esta flexibilidad permite aplicar y ajustar diversas técnicas para transformar los datos de manera óptima, adaptándolos a las necesidades del modelo supervisado. Una optimización adecuada del preprocesamiento mejora la calidad de las predicciones y refuerza la capacidad del modelo para generalizar en distintos contextos. En las secciones siguientes, profundizaremos en cada etapa del preprocesamiento, analizando tanto las técnicas más comunes como las prácticas recomendadas para evitar errores y maximizar la eficiencia del pipeline de aprendizaje supervisado.

**Para reflexionar...**
> **¿Por qué es necesario el preprocesamiento en machine learning?** 
> **Clave:** A menudo, los datos recolectados contienen valores faltantes, ruido o redundancias que afectan negativamente el aprendizaje del modelo. Reflexionar sobre cómo estas características afectan los resultados y qué técnicas de preprocesamiento pueden corregirlas.

## Limpieza de Datos

La limpieza de datos es una etapa esencial en el preprocesamiento. En ella se identifican y corrigen errores e inconsistencias en los datos, preparando un conjunto más robusto y representativo para que sea consumido por el modelo. Los datos en bruto suelen presentar problemas que, de no resolverse, pueden afectar la precisión y generalización del modelo, introduciendo sesgos o ruido que con seguridad van a comprometer su rendimiento. La limpieza se puede enfocar en varios subprocesos clave:

- **Imputación de valores faltantes**: reemplazo o ajuste de datos ausentes para preservar la consistencia.
- **Tratamiento de valores atípicos (outliers)**: identificación y ajuste o eliminación de observaciones extremas que pueden distorsionar el aprendizaje del modelo.
- **Corrección de errores y duplicados**: eliminación de registros repetidos y rectificación de errores de registro que puedan alterar los resultados.
- **Eliminación de ruido**: proceso de depuración de datos irrelevantes o que no contribuyen al aprendizaje, mejorando la claridad de los patrones.

### Imputación de valores faltantes

Los valores faltantes son comunes en los datos reales y representan un desafío que, si no se aborda adecuadamente, puede generar sesgos y limitar la calidad de las predicciones del modelo. El origen de los valores faltantes puede estar ocasionado por distintas causas, desde errores de registro hasta condiciones propias de la captura de datos, y las estrategias para imputarlos dependerán de su naturaleza y de la cantidad de valores ausentes.

El proceso de imputación de valores faltantes abarca varias técnicas, y entre las más comunes podemos enumerar las siguientes

#### **Imputación con medidas estadísticas de centralización (media, mediana o moda)**: 

Este método es una solución sencilla y rápida para conjuntos de datos en los que los valores faltantes son escasos. Para datos numéricos, se reemplaza el valor faltante con la media o mediana de la columna. En el caso de datos categóricos, se suele usar la moda (valor más frecuente) de la variable.

$$
X' = \frac{\sum_{i=1}^N X_i}{N} \quad \text{(para la imputación con la media)}
$$

> **Ejemplo**
> Supongamos un conjunto de datos de propiedades inmobiliarias donde el precio de algunas viviendas está ausente. Si el 5% de los precios están faltantes, podríamos imputar la media de la columna "precio" para conservar la coherencia en el dataset sin perder registros.

#### **Imputación mediante modelos predictivos**:

Aquí, los valores faltantes se estiman utilizando modelos como regresión lineal o algoritmos más complejos, como el algoritmo de K-Nearest Neighbors (KNN), el bosque aleatorio (Random Forest) e incluso redes neuronales. Estas técnicas aprovechan la información de otras variables en el conjunto de datos para predecir los valores ausentes en función de patrones y relaciones ya presentes en los datos disponibles.

De  los anteriores, la regresión lineal y KNN son dos métodos de aproximación ampliamente usados en estoss escenarios. La **regresión lineal** es una técnica común en conjuntos de datos numéricos, especialmente cuando se observan relaciones lineales entre las variables. En este enfoque, la variable con valores faltantes se considera como una variable dependiente y las variables restantes actúan como independientes, ayudando a estimar los valores ausentes en función de los patrones lineales detectados en el resto del conjunto de datos. El **K-Nearest Neighbors (KNN)** es otro método ampliamente utilizado para la imputación, ya que no asume ninguna relación lineal específica y es capaz de capturar relaciones no lineales en los datos. Para cada valor faltante, el KNN busca los $k$ vecinos más cercanos (es decir, registros completos con valores similares en otras características) y utiliza una medida agregada de estos vecinos, como la media o la mediana, para imputar el valor faltante. Este enfoque es especialmente útil cuando las variables presentan fuertes correlaciones no lineales o patrones complejos.

> **Ejemplo**
> En un conjunto de datos sobre pacientes con valores faltantes en el índice de masa corporal (IMC), el KNN imputaría estos valores basándose en la edad, el peso y la altura de pacientes similares, capturando relaciones complejas sin necesidad de asumir una relación lineal.

#### **Eliminación de registros o variables**:

Cuando el número de valores faltantes es considerable y afecta a un subconjunto específico, puede resultar más efectivo eliminar los registros o variables en lugar de imputarlos. Esta decisión depende de la importancia de la variable para el modelo y de la cantidad de información que se perdería.

> **Ejemplo**
> Supongamos que estamos trabajando con un conjunto de datos sobre clientes de una entidad financiera para predecir la probabilidad de que un cliente adquiera un producto bancario específico. En este conjunto de datos, la variable "historial crediticio" presenta un 40% de valores faltantes. Esta variable es relevante, ya que suele influir en el comportamiento de compra de productos financieros, pero su alto porcentaje de valores ausentes podría introducir ruido si intentamos imputarla.
>
> En este caso, si "historial crediticio" no es fundamental para el modelo o se dispone de otras variables predictoras con información similar, podría ser más eficiente eliminar esta columna en lugar de imputarla. Alternativamente, si los registros con valores faltantes se concentran en un segmento específico (por ejemplo, clientes recientes sin historial suficiente), podríamos decidir eliminar solo esos registros, preservando la mayor cantidad de información posible y reduciendo el riesgo de introducir estimaciones poco fiables.

---

Un elemento que ha de tenerse en cuenta a la hora de acometer el problema de los valores faltantes es el hecho de cómo se sitúan en el conjunto del dataset. Así, el tratamiento de valores faltantes podría diferir según pertenezcan fundamentalmente  a características (columnas) o a observaciones (filas). Las decisiones a tomar dependerán de varios factores, incluyendo la cantidad de datos faltantes, su distribución y la relevancia de las variables para el modelo.

Cuando los valores faltantes se concentran en **columnas** (es decir, una o varias variables específicas contienen datos ausentes), existen varias estrategias para abordarlos:

- **Eliminación de columnas**: Si la cantidad de valores faltantes es alta y la columna no es esencial para el modelo, puede eliminarse. Esto es especialmente útil cuando la columna presenta más de un 30% o 40% de valores ausentes y no existe un método de imputación confiable.
  
- **Imputación**: Cuando la columna es importante o tiene información única para el modelo, se pueden emplear técnicas de imputación, como el uso de la media, mediana, moda o métodos predictivos (por ejemplo, KNN o Random Forest).

- **Ingeniería de características**: En algunos casos, los valores faltantes en una columna pueden llevar a nuevas características. Por ejemplo, la ausencia de datos en una columna de historial crediticio podría indicar que un cliente es nuevo, lo que puede utilizarse para crear una variable que indique “cliente nuevo” frente a “cliente recurrente”.

Por otro lado, cuando los valores faltantes concentran en **filas** (es decir, en registros individuales que contienen algunos valores ausentes), el tratamiento dependerá de cuántas variables y qué porcentaje de datos están ausentes:

- **Eliminación de filas**: Si una fila tiene varios valores ausentes en distintas columnas, y la pérdida de información no afectará significativamente el análisis, la fila puede eliminarse. Esto suele aplicarse cuando los valores faltantes afectan a un bajo porcentaje del conjunto de datos total y la eliminación de las filas no introduce sesgos.

- **Imputación por fila**: Si la fila tiene un solo valor faltante o en variables particularmente importantes, es común aplicar técnicas de imputación para llenar esos valores faltantes sin perder el registro completo. Este enfoque es útil para preservar la mayor cantidad de datos posible sin comprometer la integridad de los registros.

- **Segmentación de datos**: Si las filas con valores faltantes corresponden a un grupo específico (por ejemplo, nuevos clientes o productos con menos registros), es posible tratarlas de manera diferente o separarlas del conjunto principal, ajustando el análisis o el modelo a las características de cada grupo.

> **Ejemplo**
> En un conjunto de datos médicos, la variable "nivel de glucosa" presenta valores faltantes en un 25% de los registros. Si el nivel de glucosa es fundamental para el modelo, podríamos imputarlo utilizando el promedio o la mediana de los valores conocidos. Sin embargo, si los valores faltantes se concentran en una serie de registros específicos (por ejemplo, pacientes nuevos sin historial previo), podríamos optar por eliminarlos o tratarlos de manera distinta, según el objetivo del análisis.

**Para reflexionar...**

> **¿En qué casos puede resultar contraproducente imputar valores faltantes en lugar de eliminarlos?** 
> **Clave**: Considera la proporción de valores ausentes en relación al tamaño del conjunto de datos y reflexiona sobre si la imputación podría introducir ruido o distorsionar patrones clave.

### Tratamiento de valores atípicos

Los valores atípicos, o *outliers*, son observaciones que se desvían considerablemente del patrón general de los datos. Estos valores pueden surgir por errores de registro, condiciones anómalas durante la recolección de datos o, en algunos casos, por características intrínsecas del fenómeno en estudio. El tratamiento adecuado de los valores atípicos es esencial, ya que, si no se abordan, pueden distorsionar las métricas y como en otros casos afectar al aprendizaje del modelo, conduciéndolo a predicciones sesgadas o ineficaces.

Existen diferentes métodos para identificar y tratar los valores atípicos, y la elección de la técnica dependerá de la naturaleza del dataset y de la relevancia de los valores extremos para el análisis. A continuación, se detallan los principales enfoques para el tratamiento de valores atípicos.

#### Métodos de detección de valores atípicos

##### **Método basado en desviación estándar**

Este método asume que los datos siguen una distribución aproximadamente normal y considera que cualquier observación que se encuentre a una distancia superior a un número específico de desviaciones estándar respecto a la media es un valor atípico. Un umbral común es **tres desviaciones estándar**, aunque puede ajustarse según el contexto.

$$
\text{Lim superior} = \mu + 3\sigma \quad \text{y} \quad \text{Lim inferior} = \mu - 3\sigma
$$

> **Ejemplo**
> En un conjunto de datos de altura de personas, donde la media es 1,70 m y la desviación estándar es 0,1 m, cualquier persona con una altura superior a 2,0 m o inferior a 1,4 m podría considerarse un valor atípico y, por tanto, candidato a ser tratado.

##### **Método del rango intercuartílico (IQR)**

Este método es **no paramétrico** y, por tanto, es útil cuando los datos no siguen una distribución normal. Calcula los cuartiles primero ($Q1$ y $Q3$) y define los valores atípicos como aquellos que están fuera de un rango específico alrededor del IQR (es decir, el rango entre el primer y tercer cuartil).

$$
\text{Lim superior} = Q3 + 1.5 \times \text{IQR} \quad \text{y} \quad \text{Lim inferior} = Q1 - 1.5 \times \text{IQR}
$$

> **Ejemplo**
> En un conjunto de datos de ingresos, donde el IQR es 20.000 y los cuartiles son $Q1 = 30.000$ y $Q3 = 70.000$, los valores fuera del rango $[-10.000, 110.000]$ serían considerados atípicos.

> [!tip]
>
> Un **método no paramétrico** es un enfoque estadístico o de machine learning que **no asume ninguna forma específica para la distribución subyacente de los datos**. Esto significa que no requiere que los datos se ajusten a una distribución predefinida, como la distribución normal, por lo que es más flexible cuando se trabaja con datos de diferentes formas o distribuciones desconocidas.
>
> En contraste, **los métodos paramétricos** dependen de ciertos parámetros definidos y de supuestos sobre la distribución de los datos. Por ejemplo, en un modelo de regresión lineal (paramétrico), se asume una relación lineal entre las variables, y el modelo se ajusta a través de parámetros como la pendiente y la intersección. 
>
> Los métodos no paramétricos, por su parte, no hacen tales suposiciones, lo que los hace más robustos en situaciones donde los datos son complejos o presentan distribuciones irregulares. Sin embargo, esta flexibilidad también puede requerir mayores cantidades de datos para ofrecer estimaciones precisas, ya que el modelo "aprende" la estructura de los datos sin partir de una suposición inicial sobre ellos. 
>
> Dos ejemplos clásicos de métodos no paramétricos son, por un lado, el **rango intercuartílico (IQR)** para detectar outliers, basado únicamente en los cuartiles. Por otro, el **K-Nearest Neighbors (KNN)**, que en machine learning permite clasificar un dato en función de la cercanía a otros puntos, sin asumir una distribución específica. 

##### **Métodos basados en la  densidad (DBSCAN)**

En conjuntos de datos de alta dimensión, los métodos basados en la densidad, como **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, pueden ser útiles para detectar outliers. Este modelo agrupa puntos basándose en la densidad de datos. A diferencia de otros métodos de clustering, como el k-means, DBSCAN es capaz de identificar grupos de forma arbitraria y detectar outliers o ruido en los datos.

DBSCAN se fundamenta en dos parámetros definitorios. Por un lado, **epsilon (ε)**, que es la distancia máxima que define el vecindario de un punto, es decir, qué tan cerca deben estar otros puntos para considerarse vecinos. Por otro lado, el **mínimo de puntos (MinPts)**, que es la cantidad mínima de puntos que deben existir en el vecindario (dentro de una distancia ε) para que un punto se considere "denso". Con estos parámetros el modelo clasificará los punttos en tres categorías: **Puntos centrales** (aquellos que tienen al menos el valor de **MinPts** en su vecindario), **puntos frontera** (puntos que están en el vecindario de un punto central, pero que no tienen suficientes puntos vecinos para ser considerados centrales por sí mismos y, por último, **ruido u outliers** (puntos que no cumplen las condiciones para ser ni puntos centrales ni de frontera)

> **Ejemplo**
> Supongamos que estamos trabajando con un dataset de localización de taxis en una ciudad. Cada punto en el dataset representa la posición GPS de un taxi en un momento dado. La mayoría de los puntos se agrupan en áreas densas de la ciudad, como el centro y las estaciones de transporte público, pero hay algunos puntos aislados en zonas inusuales que no corresponden a rutas típicas de los taxis. Estos puntos podrían ser errores de localización o lecturas espurias que introducen **ruido** en el dataset.
>
> Para eliminar este ruido, podemos utilizar **DBSCAN** como sigue:
>
> 1. **Definimos los parámetros de DBSCAN**, eligiendo valores para **ε** y **MinPts** que definan las áreas de densidad donde suelen encontrarse los taxis. Por ejemplo:
>    - Establecemos ε = 0.01 grados (aproximadamente 1 km), que define el radio máximo para considerar puntos como vecinos.
>    - MinPts = 5, para que un área se considere un cluster debe tener al menos cinco taxis en un radio de 1 km.
> 2. **Aplicamos DBSCAN**, recorriendo cada punto (posición GPS) en el dataset y clasificando los puntos como **puntos centrales** o **de frontera** si cumplen con los criterios de densidad. Los puntos aislados, que no cumplen con MinPts en un radio de 1 km, se etiquetan como **ruido**.
>
> Al final, los puntos clasificados como ruido (outliers) corresponderán a ubicaciones poco frecuentes o improbables para los taxis y, por tanto, pueden eliminarse del dataset. Nos quedaremos únicamente con los puntos que corresponden a rutas y ubicaciones comunes de taxis, eliminando el ruido causado por errores de localización. Con el ruido eliminado, el dataset ahora refleja mejor los patrones de movimiento real de los taxis en la ciudad y puede usarse para análisis de rutas, optimización de tiempos de espera, y predicción de demanda en zonas específicas sin la distorsión de los puntos aislados.

### Técnicas de tratamiento de valores atípicos

Una vez identificados, los valores atípicos pueden ser tratados de distintas maneras dependiendo del objetivo del análisis y del impacto que se espera de estos valores en el modelo:

**Eliminación de valores atípicos**:

Cuando los valores atípicos se consideran ruidosos o irrelevantes, y su eliminación no afecta la representatividad del conjunto de datos, puede optarse por excluirlos. Esta es una opción frecuente cuando se trabaja con valores claramente anómalos y se puede justificar su omisión.

> **Ejemplo**
> En un dataset de puntajes académicos, una puntuación de 150 en una escala de 0 a 10 sería un valor atípico que probablemente indica un error de registro. En este caso, eliminarlo puede ser la mejor opción.

**Transformación de valores atípicos**

En algunos casos, se puede ajustar el valor atípico para que esté más alineado con el rango general de los datos. Las transformaciones logarítmica o raíz cuadrada son útiles para reducir la influencia de valores extremadamente altos y mejorar la simetría de los datos.

> **Ejemplo**
> En un conjunto de datos sobre precios de viviendas, donde algunas propiedades tienen precios extremadamente altos, la transformación logarítmica puede reducir el impacto de estos valores y hacer que la distribución sea más manejable para ciertos modelos.

**Imputación de valores atípicos**

Si se considera que el valor atípico puede sustituirse sin perder información importante, se puede optar por imputarlo, ya sea mediante la media, la mediana o utilizando valores más representativos del conjunto de datos. Esta estrategia es útil cuando los valores atípicos están asociados a errores de registro o entradas accidentales.

**Ajuste de modelos robustos a outliers**

En lugar de eliminar o transformar valores atípicos, algunos modelos están diseñados para ser resistentes a estos valores extremos. Algoritmos como el de regresión robusta o el uso de árboles de decisión son naturalmente menos sensibles a outliers, lo que permite trabajar con datos sin necesidad de eliminarlos o transformarlos.

**Para reflexionar...**

> **¿Cómo afecta la eliminación de valores atípicos en la interpretación y generalización del modelo?** 
> **Clave**: Reflexiona sobre la representatividad del conjunto de datos y si la eliminación de valores atípicos podría distorsionar patrones reales, especialmente en conjuntos de datos donde los outliers reflejan casos reales importantes.

## Transformación de datos

Otra fase que no puede pasarse por alto en cualquier pipeline de preprocesamiento de machine learning es la transformación de datos. En efecto, ajustar la escala y estructura de las variables permite mejorar la eficiencia y precisión de los modelos, especialmente en algoritmos que son sensibles a la magnitud de las variables. En esta fase, el **escalado**, en forma de **normalización**  o **estandarización** juegan roles específicos para hacer los datos más manejables y comparables.

### Escalado de datos

El **escalado** es un proceso general en el preprocesamiento de datos que transforma las variables para ajustar su magnitud o amplitud, ayudando a que tengan un impacto más equilibrado en el modelo. Este ajuste es importante, ya que sin escalado, las variables con rangos muy diferentes pueden hacer que los modelos de machine learning interpreten desproporcionadamente la información, llevando a decisiones de aprendizaje que no representan adecuadamente el comportamiento del sistema. 

> **Ejemplo**
> En un conjunto de datos que contiene características como ingresos y años de experiencia, la diferencia de magnitud entre las unidades (dólares y años) puede hacer que los algoritmos perciban los ingresos como una característica más importante simplemente debido a sus valores más altos, sin que esto sea necesariamente cierto.

Existen distintos métodos de escalado que se eligen según el tipo de algoritmo y la naturaleza de los datos. Es importante tener claro que el escalado no modifica la estructura o relación entre los datos, sino que ajusta su amplitud para hacer más eficiente el aprendizaje y reducir el tiempo de convergencia en modelos iterativos. 

> [!tip]
>
> El escalado mejora la uniformidad en el tratamiento de las variables, incrementando la precisión, estabilidad y eficiencia del modelo, especialmente cuando se utilizan algoritmos que dependen de relaciones de magnitud entre los datos.

#### Normalización

La **normalización** es un tipo específico de escalado que ajusta los valores de las variables para que se encuentren dentro de un rango definido, generalmente entre 0 y 1. Este enfoque es útil cuando los datos presentan rangos de valores muy dispares, lo cual puede afectar algoritmos como redes neuronales y máquinas de vectores de soporte (SVM), donde las diferencias de magnitud entre variables pueden influir en los gradientes y desestabilizar el proceso de entrenamiento.

La fórmula para normalizar un valor $X$ en un rango $[0, 1]$ es:

$$
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

donde:
- $X_{\min}$ y $X_{\max}$ son el valor mínimo y máximo de la variable, respectivamente.

> **Ejemplo**
> En redes neuronales, la normalización permite que las variables se mantengan en un rango uniforme, mejorando la velocidad y estabilidad del entrenamiento al evitar que valores extremos dominen en el cálculo de los gradientes.

#### Estandarización

La **estandarización**, también conocida como ***Z-score scaling***, transforma los datos para que tengan una media de 0 y una desviación estándar de 1. A diferencia de la normalización, la estandarización no ajusta los datos a un rango específico, sino que los centra y escalan en función de su variabilidad. Este método es especialmente útil cuando los datos tienen una distribución cercana a la normal y en algoritmos que dependen de la varianza de las variables, como la regresión lineal o los modelos basados en distancia.

La fórmula para estandarizar un valor $X$ es:

$$
z = \frac{X - \mu}{\sigma}
$$

donde:
- $\mu$ representa la media de la variable,
- $\sigma$ es la desviación estándar de la variable.

> **Ejemplo**
> La estandarización es ideal para algoritmos sensibles a la dispersión de los datos, como K-Nearest Neighbors (KNN) y regresión logística, ya que permite comparar variables de diferente magnitud en una escala común sin modificar su distribución relativa.

---

**Para reflexionar...**

> **¿En qué situaciones sería más adecuado utilizar normalización en lugar de estandarización y viceversa?**  
> **Clave**: Reflexiona sobre cómo ciertos algoritmos, como las redes neuronales, se benefician de la normalización al requerir un rango uniforme de datos, mientras que otros, como los basados en distancia, se benefician de la estandarización para comparar variables con distintas unidades de medida.

### Codificación de variables categóricas

Ya sabemos que los **datos categóricos** son tipos de variables que representan categorías o grupos y no tienen un valor numérico intrínseco. Por  ello, al no ser interpretables numéricamente, deben convertirse en un formato que los modelos puedan procesar. Ya estudiamos que existen distintas técnicas de codificación, y la elección de una u otra dependerá del tipo de modelo y de la naturaleza de las categorías:

##### **One-hot encoding**

En esta técnica, cada categoría de una variable se convierte en una columna binaria (0 o 1). Esto es útil cuando se tienen pocas categorías y todas ellas son igualmente relevantes para el análisis. Este método evita que los modelos interpreten una jerarquía o relación ordinal entre las categorías.

> **Ejemplo**
> En un conjunto de datos con una columna "Color" que contiene valores "Rojo", "Verde" y "Azul", el one-hot encoding crea tres columnas binarias: "Color_Rojo", "Color_Verde" y "Color_Azul", donde el valor es 1 para indicar la presencia del color y 0 en caso contrario.

##### **Label encoding**

El label encoding asigna un valor numérico a cada categoría, lo que permite reducir el número de columnas en el dataset. Es una técnica útil en modelos basados en árboles de decisión, que pueden manejar directamente las relaciones ordinales o jerárquicas en categorías numéricas.

> **Ejemplo**
> En una variable "Tamaño" con categorías "Pequeño", "Medio" y "Grande", el label encoding podría asignar 1 a "Pequeño", 2 a "Medio" y 3 a "Grande".

##### **Codificación ordinal**

La codificación ordinal se utiliza cuando **las categorías tienen un orden inherente**. Este tipo de codificación asigna un número a cada categoría según su jerarquía, permitiendo que los modelos interpreten la relación entre estas categorías. Es útil en algoritmos sensibles al orden, como la regresión.

> **Ejemplo**
> En una variable "Calidad" con valores "Baja", "Media" y "Alta", la codificación ordinal asignaría 1 a "Baja", 2 a "Media" y 3 a "Alta".

##### **Codificación de frecuencia**

La codificación de frecuencia asigna a cada categoría un valor numérico basado en su frecuencia de aparición en el dataset. Es útil cuando las categorías más comunes tienen un impacto mayor en el modelo y permite simplificar el número de columnas.

> **Ejemplo**
> En un dataset de productos vendidos, si la categoría "Electrónica" aparece en el 60% de los registros, "Ropa" en el 30% y "Hogar" en el 10%, se asignan valores proporcionales a la frecuencia (por ejemplo, 0.6 para Electrónica, 0.3 para Ropa y 0.1 para Hogar).

##### **Codificación de Target (Target Encoding)**

Reemplaza cada categoría con la media (o mediana) de la variable objetivo para esa categoría. Es útil en problemas de clasificación cuando hay una fuerte correlación entre las categorías y el objetivo, aunque puede llevar al sobreajuste si no se maneja cuidadosamente.

> **Ejemplo**
> En un dataset de predicción de precios de vivienda con una columna "Barrio" y la variable objetivo "Precio", la codificación de target asigna a cada barrio la media de precios en ese barrio: "Centro" = 250.000 €, "Periferia" = 180.000 €, "Rural" = 120.000 €.

##### **Codificación Leave-One-Out (LOO)**:

Variante de Target Encoding donde cada valor se codifica excluyendo el valor de la variable objetivo de la fila actual, lo que ayuda a reducir el sobreajuste.

> **Ejemplo**
> En un dataset de seguros con una columna "Categoría de Vehículo" y una variable objetivo "Coste de Seguro", cada categoría se codifica con la media de costes de seguro, excluyendo la observación actual. Así, "SUV" podría tener una media de costo de 1.200 € excluyendo el valor actual, "Compacto" = 900 €, "Deportivo" = 1.500 €.

##### **Codificación Binaria**

Convierte cada categoría en una representación binaria y asigna cada dígito de la secuencia binaria a una columna nueva. Esto reduce la dimensionalidad en comparación con One-Hot Encoding y es útil para variables con muchas categorías.

> **Ejemplo**
> En un conjunto de datos de ventas con una columna "Producto" que contiene los valores "Producto A", "Producto B", "Producto C", la codificación binaria convierte cada producto en una secuencia binaria: "Producto A" = 00, "Producto B" = 01, "Producto C" = 10. Esta codificación permite reducir la dimensionalidad cuando estamos trabajando con muchas categorías.



**Para reflexionar...**

> **¿Cómo puede afectar la elección del método de codificación de variables categóricas al rendimiento y precisión del modelo?**  
> **Clave**: Reflexiona sobre cómo algunos métodos, como el one-hot encoding, pueden aumentar la dimensionalidad del dataset, mientras que otros, como la codificación ordinal o de target, pueden introducir información adicional que influye en los resultados. Considera cómo esta elección podría afectar a distintos tipos de algoritmos.

## Reducción de dimensionalidad

La reducción de dimensionalidad es una técnica fundamental en machine learning, utilizada para simplificar el espacio de características (también llamado espacio de variables) sin perder información relevante. Al disminuir el número de características, se facilita el procesamiento y almacenamiento de datos, además de mejorar la eficiencia y capacidad de generalización de los modelos, especialmente en conjuntos de datos de alta dimensionalidad. Esta técnica es especialmente útil en aplicaciones con gran volumen de datos, donde demasiadas variables pueden introducir ruido, aumentar la complejidad del modelo y ralentizar el aprendizaje.

> [!tip]
>
> El **espacio de características** es el conjunto de todas las variables o atributos (características) que describen las instancias en un conjunto de datos. Cada característica representa una dimensión en este espacio, y cada instancia o ejemplo del conjunto de datos se representa como un punto dentro de él, definido por los valores de sus características.
>
> En términos geométricos, si un dataset tiene $n$ características, el espacio de características será un espacio $n$-dimensional donde cada dimensión corresponde a una característica. Por ejemplo, en un dataset con solo dos características ("edad" y "ingresos"), el espacio de características sería bidimensional, donde cada punto (instancia) se posiciona en función de sus valores de edad e ingresos. En problemas con alta dimensionalidad, este espacio puede volverse muy grande, dificultando la visualización y el análisis, razón por la cual se aplican técnicas como la reducción de dimensionalidad.

Existen dos enfoques principales para la reducción de dimensionalidad. Por un lado, la **extracción de características**, que trata de transformar las variables originales en un conjunto de nuevas variables, manteniendo la información relevante en un espacio de menor dimensionalidad. Por otro lado, la **selección de características**, que intenta retener solo las variables más relevantes del conjunto original, descartando aquellas de menor importancia o redundantes. 

La elección de un enfoque dependerá de los objetivos del análisis y del tipo de datos.

### Métodos de extracción

#### **Análisis de Componentes Principales (PCA)**

El Análisis de Componentes Principales (PCA) es una técnica de extracción de características que transforma las variables originales en un conjunto de componentes ortogonales (independientes entre sí). Estos componentes capturan la mayor cantidad de varianza en los datos, con el objetivo de conservar la información más relevante y descartar redundancias.

> [!tip]
>
> Un **conjunto de componentes ortogonales** es un conjunto de variables derivadas (o componentes) que son mutuamente independientes entre sí, es decir, no presentan correlación. En el contexto de reducción de dimensionalidad, estos componentes ortogonales son el resultado de transformar las características originales de modo que cada componente capture una porción distinta de la variabilidad o información presente en los datos, **sin solapamiento entre componentes.**
>
> En técnicas como el **Análisis de Componentes Principales (PCA)**, los componentes ortogonales son calculados de manera que cada uno maximiza la varianza de los datos en una dirección específica, asegurando que la información capturada por cada componente es única respecto a los demás. Esta independencia facilita el análisis al evitar redundancia y permite representar el espacio de datos en una dimensión más baja sin pérdida de información redundante.

La transformación realizada con la técnica **PCA** (Análisis de Componentes Principales) reorganiza las características originales para que los nuevos componentes (variables derivadas) capturen la máxima cantidad de variación posible en los datos. Esto significa que estos componentes calculados **explican la mayor parte de la variabilidad en el dataset**. Si seleccionamos únicamente los componentes principales calculados (aquellos que contienen la mayor variabilidad), se logra reducir la dimensionalidad del conjunto de datos mientras se mantiene la mayor parte de la información relevante, evitando así redundancia y simplificando el análisis.

$$
\text{PCA}: \quad \text{Maximiza la varianza a lo largo de componentes ortogonales}
$$

> **Ejemplo:** Imaginemos un conjunto de datos muy sencillo que tiene solo dos variables, **Altura** y **Peso**, para tres personas. Los valores son los siguientes:
>
> | Persona | Altura (cm) | Peso (kg) |
> | ------- | ----------- | --------- |
> | A       | 160         | 55        |
> | B       | 170         | 65        |
> | C       | 180         | 75        |
>
> A partir de estos datos, observamos que **Altura** y **Peso** están correlacionados: a mayor altura, mayor peso. **PCA** busca transformar estos datos en dos nuevos componentes (nuevas variables) que capturen la máxima variabilidad de forma independiente (ortogonal).
>
> ###### **Ejecución de la técnica**
>
> Primero, para centrar los datos, restamos la media de cada variable:
>
> - Media de Altura = (160 + 170 + 180) / 3 = 170
> - Media de Peso = (55 + 65 + 75) / 3 = 65
>
> | Persona | Altura centrada | Peso centrado |
> | ------- | --------------- | ------------- |
> | A       | 160 - 170 = -10 | 55 - 65 = -10 |
> | B       | 170 - 170 = 0   | 65 - 65 = 0   |
> | C       | 180 - 170 = 10  | 75 - 65 = 10  |
>
> Ahora se trataría de calcular la covarianza entre Altura y Peso, para mostrar cómo varían juntas las dos variables:

> $$
> \text{Covarianza} = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
> $$
>
> Para nuestro ejemplo:
>
> - Cov(Altura, Altura) = $\frac{1}{2}((-10)^2 + 0^2 + 10^2) = 100$
> - Cov(Peso, Peso) = $\frac{1}{2}((-10)^2 + 0^2 + 10^2) = 100$
> - Cov(Altura, Peso) = $\frac{1}{2}((-10 \cdot -10) + (0 \cdot 0) + (10 \cdot 10)) = 100$
>
> La **matriz de covarianza** es entonces:
>
> $$
> \begin{bmatrix}
> 100 & 100 \\
> 100 & 100 \\
> \end{bmatrix}
> $$
>
> Ahora habría que encontrar los **valores propios** y **vectores propios** de la matriz de covarianza. Estos valores y vectores nos permitirán identificar los componentes principales, que representan las direcciones de máxima varianza en los datos. Para encontrar los valores propios tenemos que resolver la **ecuación característica**:
>
> $$
> \det(\text{Matriz de Covarianza} - \lambda I) = 0
> $$
>
> donde $\lambda$ representa los valores propios y $I$ es la matriz identidad.
>
> Para nuestra matriz:
>
> $$
> \det 
> \begin{bmatrix}
> 100 - \lambda & 100 \\
> 100 & 100 - \lambda \\
> \end{bmatrix}
> = 0
> $$
>
> Resolviendo esta ecuación, obtenemos dos valores propios:
>
> 1. $\lambda_1 = 200$
> 2. $\lambda_2 = 0$
>
> #### Interpretación de los Valores Propios
>
> - **$\lambda_1 = 200$**: Este es el valor propio mayor, y representa la **varianza máxima** capturada por el primer componente principal. Este valor indica que la mayor parte de la variabilidad en los datos se explica a lo largo de esta dirección.
> - **$\lambda_2 = 0$**: Este valor propio es cero, lo que significa que el segundo componente no aporta varianza adicional y, por lo tanto, no contiene información útil para explicar la variabilidad en los datos.
>
> A continuación, **calculamos los vectores propios asociados a cada valor propio**:
>
> 1. **Para $\lambda_1 = 200$**: Resolvemos el sistema lineal $(\text{Matriz de Covarianza} - 200I)v = 0$ y obtenemos el vector propio asociado:
>
> $$
>    v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
> $$
>
>    Este vector propio indica la dirección del primer componente principal, que representa una combinación lineal de **Altura** y **Peso**.
>
> 2. **Para $\lambda_2 = 0$**: Resolvemos el sistema para el segundo valor propio, obteniendo:
>
> $$
> v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
> $$
>
> Este vector propio representa una dirección perpendicular al primero, pero como su valor propio es cero, no aporta variabilidad útil a los datos.
>
> El siguiente paso conllevará proyectar los datos en el primer componente principal
>
> Con el primer componente principal $v_1 = [1, 1]$, proyectamos los datos sobre esta dirección para reducir la dimensionalidad:
>
> - Para la Persona A: $\text{Proyección} = (-10, -10) \cdot (1, 1) = -20$
> - Para la Persona B: $\text{Proyección} = (0, 0) \cdot (1, 1) = 0$
> - Para la Persona C: $\text{Proyección} = (10, 10) \cdot (1, 1) = 20$
>
> Finalmente vemos que la proyección en el primer componente principal (que captura la máxima varianza) nos da los valores $[-20, 0, 20]$ para las personas A, B y C. Al reducir la dimensionalidad a este único componente, hemos capturado la mayor parte de la variabilidad original de los datos en un solo eje, logrando una representación simplificada sin pérdida significativa de información.

> [!tip]
>
> Los **autovalores** y **autovectores** de una matriz son conceptos fundamentales en álgebra lineal y se utilizan en muchas aplicaciones de machine learning, entre las que se incluyen, como hemos visto, la reducción de dimensionalidad mediante PCA. 
>
> Un **autovalor** (o valor propio) de una matriz cuadrada $A$ es un número escalar $\lambda$ que cumple con la ecuación:
>
> $$
> A \mathbf{v} = \lambda \mathbf{v}
> $$
>
> donde $\mathbf{v}$ es un vector no nulo.
>
> Un **autovector** (o vector propio) de una matriz $A$ es un vector $\mathbf{v}$ que, cuando se multiplica por $A$, resulta en un vector que es un múltiplo escalar de $\mathbf{v}$ mismo, es decir, solo cambia de magnitud y no de dirección.
>
> Supongamos que la matriz $A$ es diagonalizable. Esto quiere decir que existen unas matrices $P$ y $D$ tal que 
>
> $$
> A = PDP^{-1}
> $$
>
> Donde $D$ es una matriz diagonal y $P$ es una  matriz formada por los autovectores de $A$ en columnas.
>
> En otras palabras, si $A$ es una transformación lineal representada por una matriz, **los autovectores son las "direcciones" que permanecen inalteradas bajo la acción de** $A$, **mientras que los autovalores indican cuánto se escalan (o comprimen) esos vectores en esas direcciones**.

> [!tip]
>
> ##### **Cálculo de los autovalores**: 
>
> Los autovalores $\lambda$ de una matriz $A$ se encuentran resolviendo la **ecuación característica**:
>
> $$
> \det(A - \lambda I) = 0
> $$
>
> Esta ecuación da un polinomio en $\lambda$, cuyas raíces son los autovalores de $A$. Este cálculo **no requiere que $A$ esté diagonalizada**.
>
> ##### **Cálculo de los autovectores**:  
>
> Con los autovalores obtenidos, se sustituyen cada uno en la ecuación $(A - \lambda I) \mathbf{v} = 0$ para encontrar los autovectores $\mathbf{v}$ asociados a cada autovalor. Estos autovectores representan las direcciones invariables bajo la transformación $A$.

### Análisis Discriminante Lineal (LDA)

El **Análisis Discriminante Lineal** es un método de reducción de dimensionalidad supervisado que se utiliza principalmente en problemas de clasificación. A diferencia del método PCA, que es no supervisado y busca maximizar la varianza general en el conjunto de datos, LDA aprovecha la información de las clases para encontrar las direcciones de proyección que maximizan la **separación entre clases**. Este método es especialmente útil cuando se desea reducir la dimensionalidad manteniendo la capacidad discriminativa de las características.

El objetivo de LDA es proyectar los datos en un nuevo espacio de menor dimensión en el que las observaciones de cada clase estén lo más agrupadas posible, a la vez que las **diferencias entre las clases sean máximas**. En otras palabras, **LDA busca maximizar la distancia entre las medias de las clases (varianza entre clases) y minimizar la dispersión dentro de cada clase (varianza intra-clase).**

La fundamentación matemática pasa por entender varias fases dentro de este método. En primer lugar es necesario calcular **las medias de cada clase** y la **media global**. La media de cada clase ($\mu_c$) es el promedio de los valores de cada característica dentro de una clase específica, mientras que la media global ($\mu$) es el promedio de todas las observaciones en el conjunto de datos, sin importar la clase.  Seguidamente hay que calcular las dispersiones **intra-clase** ($S_W$),  que mide qué tan dispersos están los datos dentro de cada clase, y **entre clases** ($S_B$), que mide la separación entre las nubes de cada clase en relación con la media global. Por último quedaría maximizar la razón entre la dispersión entre clases y la dispersión intra-clase, encontrando una manera de proyectar los datos en un nuevo eje donde las clases estén lo más separadas posible entre sí, mientras que la variabilidad dentro de cada clase se mantiene baja.

Una vez que se encuentran las direcciones de proyección óptimas, los datos originales se proyectan en este nuevo espacio de menor dimensión. En un problema con $C$ clases, LDA puede reducir los datos a un máximo de $C-1$ dimensiones, ya que la información discriminativa se limita a las diferencias entre clases.

**LDA es ideal para problemas donde la clasificación es el objetivo principal** y donde los datos presentan una estructura de clases bien definida. Algunas aplicaciones comunes de LDA incluyen el  **reconocimiento facial**, el **análisis de sentimientos** o el **diagnóstico médico**

Si bien es cierto que LDA tiene varias ventajas, como la capacidad de mejorar la interpretabilidad del modelo, la eficiencia computacional, además de ser particularmente efectivo en problemas de clasificación, también hay que tener en cuenta que presenta algunas limitaciones. Para empezar supone que las variables tienen **distribuciones normales y con la misma covarianza** en cada clase, lo cual puede no cumplirse en la práctica. Además también es menos efectivo en problemas donde las clases no son linealmente separables o en presencia de datos con alta variabilidad intra-clase.

### **Métodos de selección**

La selección de características, a diferencia de la extracción, consiste en identificar y retener solo las variables más relevantes del conjunto de datos original, **eliminando aquellas que aportan poco al modelo o son redundantes**. Esto ayuda a reducir el ruido y a mejorar la interpretabilidad del modelo, al trabajar con un número menor de variables significativas. En el caso de los métodos de selección podemos encontrar algunas técnicas que son las más habituales

#### Técnicas basadas en filtros

Estas técnicas son rápidas y sencillas de implementar, aunque podemos encontrarnos con el problema de que no son capaces de capturar relaciones entre características. Básicamente, se trata de evaluar cada característica de forma independiente de los modelos y seleccionar las que son más relevantes según alguna métrica estadística. Son útiles como paso inicial en la selección de variables.

Por un lado, se pueden mencionar los **análisis de correlación**, empleados para identificar y eliminar características que están altamente correlacionadas entre sí. Este análisis se utiliza para explorar y describir las relaciones entre pares de variables, identificando aquellas que presentan una correlación alta y pueden aportar información redundante. Es un procedimiento exploratorio en el que se calculan medidas de asociación (como el coeficiente de Pearson) sin evaluar la significancia estadística de esas relaciones.

También pueden usarse **pruebas de carácter estadístico**. Estas pruebas evalúan la relación entre características y la variable objetivo, ayudando a identificar las variables más relevantes para el modelo. Las más frecuentes son:

- **Chi-cuadrado**: Es una prueba utilizada con variables categóricas. Permite evaluar si existe una dependencia significativa entre dos variables categóricas, como por ejemplo, entre la variable de género y el resultado de una encuesta de preferencia.
- **Prueba de F**: Compara la varianza entre grupos de una variable categórica respecto a una variable continua. Es especialmente útil en problemas de clasificación, ya que ayuda a seleccionar las características que explican mejor la variabilidad entre grupos.
- **Significancia basada en el coeficiente de Pearson**: Estas pruebas utilizan el coeficiente de Pearson en una **prueba de hipótesis** que evalúa si la correlación observada en la muestra es significativa en la población. A través del cálculo de un valor $p$, determinan si la correlación es suficientemente fuerte como para ser considerada real y no producto del azar.

Por último, también se puede hacer uso de técnicas de **información Mutua**. En estos casos se trata de calcular la dependencia entre dos variables, sin limitarse a relaciones lineales. Es útil para seleccionar características con una dependencia alta respecto a la variable objetivo, ya que es capaz de medir hasta qué punto "conocer" una variable ayuda a predecir otra, independientemente de si la relación es lineal o no.

#### Selección basada en envolturas (*Wrapper*)

Los métodos de envoltura prueban varias combinaciones de características en el modelo y seleccionan las que maximizan su rendimiento. Estos métodos pueden ser más precisos porque son capaces de detectar interacciones entre variables. El problema es que requieren un alto costo computacional. Entre los más frecuentes se pueden enumerar los siguientes:

- **Método de Forward Selection** (Selección hacia adelante): Se trata de comenzar con un conjunto vacío e ir agregando características una por una, evaluando el rendimiento del modelo cada vez que se añade una nueva característica. Se seleccionan aquellas que mejoran el rendimiento.
- **Método de Backward Elimination** (Eliminación hacia atrás): En este caso se empieza con todas las características y se van eliminando aquellas que menos contribuyen al rendimiento del modelo en cada iteración hasta encontrar el conjunto óptimo.
- **Método de Recursión** (Recursive Feature Elimination, RFE): Entrena el modelo y elimina recursivamente las características menos importantes en cada iteración, recalculando la importancia en cada paso hasta quedarse con un subconjunto óptimo de variables.

#### Métodos Basados en Modelos Incorporados (*Embedded*)

Los métodos **incorporados** integran la selección de características dentro del entrenamiento del modelo. A medida que el modelo se entrena, asigna pesos o importancia a cada característica y selecciona las más relevantes. Los más habituales son:

- **Lasso (Least Absolute Shrinkage and Selection Operator)**. Este método aplica una penalización a los coeficientes del modelo, lo que fuerza a algunos coeficientes a cero, eliminando efectivamente características de menor relevancia. Este enfoque es útil en problemas de regresión.
- **Bosque Aleatorio y Árboles de Decisión**: Estos modelos calculan la importancia de cada característica según su impacto en la precisión del modelo. Las características que dividen mejor los datos (es decir, que más contribuyen a reducir la impureza de los nodos) reciben mayor importancia.
- **Máquina de Vectores de Soporte con Regularización (SVM con regularización L1)**: Asigna importancia a las características a través de sus pesos y selecciona solo las que tienen un impacto significativo en la clasificación.

> **Ejemplo práctico: Aplicación del método Lasso**
>
> El **método Lasso** es un ejemplo de técnica de selección de características **incorporada** (embedded). Realiza la selección de características dentro del proceso de entrenamiento del modelo. Este método es particularmente útil en problemas de **regresión** y es ampliamente utilizado en la selección de características cuando tenemos un gran número de variables. Veámoslo en mayor detalle:
>
> **Lasso** (Least Absolute Shrinkage and Selection Operator) es una variante de la **regresión lineal** que incluye una penalización basada en la norma L1 de los coeficientes. La fórmula de la regresión Lasso es:
>
> $$
> \underset{\beta}{\text{minimizar}} \quad \frac{1}{2n} \sum_{i=1}^n \left( y_i - \hat y_i  \right)^2 + \alpha \sum_{j=1}^p |\beta_j|
> $$
>
> donde:
>
> - $y_i$ son los valores observados,
> - $\hat{y}_i$ son los valores predichos,
> - $\beta_j$ son los coeficientes de las características,
> - $\alpha$ es el parámetro de regularización (también llamado parámetro de penalización)
>
> La penalización L1 (término $\alpha \sum_{j=1}^p | \beta_j |$) fuerza a algunos coeficientes $\beta_j$ a volverse exactamente **cero** cuando $\alpha$ es lo suficientemente grande. Esto elimina efectivamente aquellas características que no aportan valor al modelo, simplificando el conjunto de características a solo las más relevantes.
>
> 1. **Coeficientes Reducidos a Cero**: A medida que aumentamos el parámetro de penalización $\alpha$, Lasso disminuye los coeficientes de las características menos importantes hasta llevarlos a cero. Esto significa que la característica correspondiente es eliminada del modelo.
> 2. **Selección Automática**: Lasso selecciona automáticamente solo aquellas características que tienen un impacto significativo en la predicción, proporcionando un modelo más sencillo y eliminando el ruido de las variables irrelevantes.
> 3. **Balance entre Complejidad y Precisión**: El valor de $\alpha$ controla el balance entre mantener características y simplificar el modelo. Valores más altos de $\alpha$ tienden a eliminar más características, reduciendo la dimensionalidad pero pudiendo perder precisión. Valores bajos de $\alpha$ conservan más características pero pueden hacer que el modelo sea menos interpretable.
>
> Supongamos un modelo con cinco características, donde aplicamos Lasso y obtenemos los siguientes coeficientes después de la regularización:
>
> | Característica | Coeficiente sin Lasso | Coeficiente con Lasso |
> | -------------- | --------------------- | --------------------- |
> | $X_1$          | 2.5                   | 2.1                   |
> | $X_2$          | 1.2                   | 0.0                   |
> | $X_3$          | -0.5                  | -0.3                  |
> | $X_4$          | 0.3                   | 0.0                   |
> | $X_5$          | 0.7                   | 0.0                   |
>
> Aquí, las características $X_2$, $X_4$ y $X_5$ han sido eliminadas porque sus coeficientes fueron reducidos a cero por la penalización L1 de Lasso. Esto reduce la complejidad del modelo, manteniendo solo las variables con impacto real en la predicción.
>
> ##### Ventajas y Limitaciones de Lasso
>
> Lasso permite realizar la selección de características automáticamente dentro del proceso de entrenamiento, reduciendo la complejidad del modelo y el riesgo de sobreajuste. Al disminuir el número de variables, mejora la interpretabilidad, ya que conserva solo las características más relevantes. Sin embargo, Lasso puede no ser ideal cuando existe una gran cantidad de características correlacionadas, ya que en tales casos puede eliminar variables importantes. Además, la penalización L1 no siempre capta relaciones complejas entre variables, por lo que, en algunos casos, otros métodos de selección de características pueden ser más efectivos, como combinar Lasso con técnicas de filtrado inicial.

### Ventajas y limitaciones de la reducción de dimensionalidad

Evidentemente la reducción de dimensionalidad ofrece varias ventajas, como la mejora de la eficiencia computacional y la reducción del riesgo de sobreajuste, especialmente en conjuntos de datos pequeños o con alto número de características. Sin embargo, también es importante tener en  cuenta que al reducir la dimensionalidad, **también se puede perder interpretabilidad**, ya que las transformaciones pueden dificultar la relación directa entre los datos originales y las variables transformadas (como puede ocurrir en la técnica PCA, por ejemplo).

Es fundamental evaluar el balance entre eficiencia y pérdida de información al elegir los métodos y el grado de reducción.

**Para reflexionar...**

> **¿Cuándo es ventajoso reducir la dimensionalidad en un modelo supervisado?** 
> **Clave**: Reflexiona sobre cómo una alta dimensionalidad puede introducir ruido y dificultar el aprendizaje, mientras que una reducción moderada ayuda a mejorar el rendimiento del modelo y la generalización. Considera cómo, en problemas específicos, el exceso de variables puede dificultar la interpretación y aumentar el riesgo de sobreajuste.

## Tratamiento de datos desbalanceados

En problemas de clasificación supervisada, un conjunto de datos desbalanceado se caracteriza porque una o varias clases están mucho más representadas que otras. Esta disparidad en la cantidad de observaciones de cada clase puede afectar el desempeño del modelo, ya que los algoritmos tienden a aprender mejor los patrones de la clase mayoritaria. Ello puede conducir a una disminución de la precisión y de la capacidad de generalización en las clases minoritarias. Este fenómeno es común en aplicaciones como la detección de fraudes, diagnóstico de enfermedades raras o análisis de datos de clientes, donde una clase de interés suele estar subrepresentada.

El preprocesamiento de datos desbalanceados permite ajustar el conjunto de datos de modo que todas las clases tengan una representación adecuada. A continuación, revisamos las principales técnicas utilizadas para el tratamiento de datos desbalanceados, cada una con ventajas específicas dependiendo de la naturaleza del problema y del modelo.

Pero para empezar veamos cual es el alcance del problema de las clases desbalanceadas analizando las métricas de un problema de clasificación con la matriz de confusión. Vamos a suponer que estamos ante el problema clásico de detección de fraudes. Es habitual que en este tipo de problemas los datasets de entrenamiento estén desbalanceados a favor de la clase mayoritaria (transacciones no-fraudulentas). Imaginemos pues un dataset con 1.000 transacciones, donde:

- **950 transacciones** son normales (clase mayoritaria).
- **50 transacciones** son fraudes (clase minoritaria).

Supongamos que entrenamos un modelo de clasificación para detectar fraudes, pero no aplicamos ninguna técnica de balanceo de clases. El modelo entrenado podría generar la siguiente matriz de confusión:

|                     | Predicción: No Fraude | Predicción: Fraude |
| ------------------- | --------------------- | ------------------ |
| **Real: No Fraude** | 930                   | 20                 |
| **Real: Fraude**    | 45                    | 5                  |

En este caso:
- **930 verdaderos negativos**: transacciones normales que el modelo predijo correctamente como "No Fraude".
- **20 falsos positivos**: transacciones normales clasificadas incorrectamente como "Fraude".
- **45 falsos negativos**: fraudes clasificados incorrectamente como "No Fraude".
- **5 verdaderos positivos**: fraudes correctamente clasificados como "Fraude".

Ahora, calculamos las tres métricas comunes para este modelo:

1. **Exactitud (Accuracy)**: la proporción de predicciones correctas en el total de predicciones.

$$
\text{Exactitud} = \frac{\text{Verdaderos Positivos} + \text{Verdaderos Negativos}}{\text{Total de Predicciones}} = \frac{930 + 5}{1000} = 0.935 \text{ (93.5%)}
$$

2. **Precisión (Precision)** para la clase "Fraude": la proporción de predicciones correctas de "Fraude" entre todas las predicciones de "Fraude".

$$
\text{Precisión} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Positivos}} = \frac{5}{5 + 20} = 0.20 \text{ (20%)}
$$

3. **Sensibilidad (Recall)** para la clase "Fraude": la proporción de fraudes correctamente identificados.

$$
\text{Sensibilidad} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Negativos}} = \frac{5}{5 + 45} = 0.10 \text{ (10%)}
$$

A la luz de estos resultados podemos concluir que:

- Aunque el modelo tiene una **alta exactitud (93.5%)**, esto es engañoso porque se debe principalmente a la gran cantidad de transacciones normales (950). La mayoría de las predicciones correctas corresponden a la clase mayoritaria.

- La **precisión (20%)** y la **sensibilidad (10%)** en la clase "Fraude" son muy bajas, indicando que el modelo no está reconociendo correctamente los fraudes. De hecho podría haberse calculado el $F1-Score$ para incluir los efectos combinados de la precisión y la sensibilildad:

$$
F1 = 2 \times \frac{\text{Precisión} \times \text{Sensibilidad}}{\text{Precisión} + \text{Sensibilidad}} = 2 \times \frac{0.20 \times 0.10}{0.20 + 0.10} = 2 \times \frac{0.02}{0.30} = 2 \times 0.0667 = 0.1333 \text{ (13.3%)}
$$
  
  Este valor confirma el problema en la detección de la clase "Fraude"

- La matriz de confusión muestra que el modelo detecta solo 5 fraudes de los 50 existentes, mientras que clasifica erróneamente 45 fraudes como transacciones normales. Esto sugiere un problema de **sesgo hacia la clase mayoritaria** debido al desbalance.

Este ejemplo muestra cómo los datos desbalanceados pueden inflar métricas generales como la exactitud, mientras ocultan el pobre rendimiento del modelo en la clase minoritaria. Un análisis detallado de la matriz de confusión revela estas deficiencias, y la baja precisión y sensibilidad en la clase "Fraude" demuestran que el modelo no está capturando patrones de interés en la clase de menor representación.

### Principales técnicas para tratar datos desbalanceados

#### Submuestreo y sobremuestreo

El **submuestreo** es una técnica de balanceo de clases en la que se reduce el número de observaciones de la clase mayoritaria para igualar, o aproximar, el tamaño de la clase minoritaria. Esta técnica es útil cuando el conjunto de datos está muy desbalanceado y el exceso de datos de la clase mayoritaria podría conducir a un modelo sesgado. Sin embargo, al eliminar datos, existe el riesgo de perder información importante, por lo que se recomienda usarlo cuando **el tamaño de la clase mayoritaria es suficientemente grande para no comprometer la representatividad**.

> **Ejemplo 1. Detección de fraude en transacciones bancarias:** Imaginemos un dataset de 1.000 transacciones bancarias:
>
> - **950 transacciones** son normales.
> - **50 transacciones** son fraudes.
>
> Dado este desbalance, si entrenamos un modelo directamente, tenderá a clasificar las transacciones como normales. Con el submuestreo, seleccionamos aleatoriamente un subconjunto de las transacciones normales (por ejemplo, 50) y lo combinamos con las transacciones de fraude. Así, el nuevo dataset tendrá **50 transacciones normales y 50 fraudes**, lo que permite que ambas clases tengan igual representatividad y el modelo aprenda patrones de ambas sin sesgo hacia la clase mayoritaria.
>
> **Proceso de implementación del submuestreo**
>
> Para aplicar esta técnica, se siguen los pasos siguientes:
>
> 1. **Separar las clases**: Primero, separamos las observaciones de la clase mayoritaria y de la clase minoritaria.
> 2. **Seleccionar aleatoriamente** un número reducido de observaciones de la clase mayoritaria que iguale el tamaño de la clase minoritaria. Esto implica seleccionar una muestra de la clase mayoritaria sin reemplazo para no duplicar datos.
> 3. **Combinar** estas observaciones para formar un nuevo dataset balanceado, donde ambas clases tengan el mismo número de observaciones.
>
> Este proceso da como resultado un conjunto de datos equilibrado y optimizado para modelos de clasificación en problemas con alta desproporción entre las clases.
>

> **Ejemplo 2: Submuestreo en detección de correos spam**: Supongamos que estamos desarrollando un modelo para clasificar correos electrónicos en "spam" y "no spam". En un dataset de 10.000 correos:
>
> - **9.500 correos** son "no spam".
> - **500 correos** son "spam".
>
> Si entrenamos un modelo con estos datos sin balancear, tenderá a clasificar los correos como "no spam" debido a la alta proporción de esta clase. Para balancear el dataset, seleccionamos aleatoriamente 500 correos "no spam" y los combinamos con los 500 correos "spam". Esto resulta en un nuevo dataset con 1.000 observaciones (500 de cada clase), lo que permite al modelo aprender patrones en ambas clases de manera equitativa.
>

Visto lo anterior, se puede afirmar que el submuestreo presenta algunas ventajas, sobre todo la fundamental: evitar el sesgo hacia la clase mayoritaria y aumentar así su capacidad de generalización en situaciones de desbalanceo de clases. Pero también permite en determinados casos, al reducirse el tamaño del dataset, mejorar la velocidad de entrenamiento del modelo. Sin embargo, esta técnica también tiene limitaciones. Al eliminar observaciones, existe el riesgo de perder información importante si las muestras descartadas de la clase mayoritaria contienen patrones relevantes. Es por ello que, el submuestreo no es ideal cuando el tamaño total de datos es pequeño, ya que  se podría reducir la capacidad del modelo para generalizar y representar correctamente la variabilidad dentro de la clase mayoritaria.

> [!tip]
>
> El submuestreo es una técnica poderosa cuando se aplica en datasets grandes y desbalanceados, y es especialmente útil para asegurar que el modelo considere la clase minoritaria sin que el modelo tenga un sesgo inherente hacia la clase mayoritaria.

---

Al contrario del submuestreo, el **sobremuestreo** es una técnica que **incrementa la representación de la clase minoritaria en un conjunto de datos desbalanceado**. El sobremuestreo no elimina datos de la clase mayoritaria, sino que aumenta el tamaño de la clase minoritaria, ya sea duplicando ejemplos existentes o generando observaciones nuevas. Esta técnica es particularmente útil cuando se cuenta con pocos datos en la clase minoritaria y se desea evitar la pérdida de información. Sin embargo, el sobremuestreo **puede llevar a problemas de sobreajuste si no se aplica adecuadamente**.

Existen varios métodos de sobremuestreo, cada uno con diferentes enfoques y beneficios. A continuación, se describen las técnicas más comunes.

La **duplicación de observaciones** es el método de sobremuestreo más simple. Consiste en seleccionar y duplicar aleatoriamente instancias de la clase minoritaria hasta que ambas clases tengan un tamaño similar. Aunque esta técnica equilibra el dataset, también aumenta el riesgo de sobreajuste, ya que el modelo puede "memorizar" los ejemplos repetidos en lugar de generalizar.

> **Ejemplo**
> En un problema de detección de fraude en transacciones bancarias con 1.000 transacciones, de las cuales 950 son "no fraude" y 50 son "fraude", la duplicación consiste en seleccionar varias veces las 50 observaciones de "fraude" hasta llegar a 950, igualando el número de ambas clases. Este proceso permite al modelo tener una representación equilibrada de las clases, aunque aumenta el riesgo de sobreajuste.

En lugar de duplicar observaciones exactas, **el sobremuestreo aleatorio** selecciona instancias de la clase minoritaria con **reemplazo**. Esto significa que cada instancia de la clase minoritaria tiene la misma probabilidad de ser elegida varias veces, sin necesidad de que todas las observaciones sean idénticamente replicadas. Aunque es más flexible que la duplicación directa, el sobremuestreo aleatorio también puede llevar al sobreajuste, especialmente en datasets pequeños.

> **Ejemplo**
> Supongamos un dataset de clasificación de correos electrónicos donde 9.500 correos son "no spam" y 500 son "spam". El sobremuestreo aleatorio selecciona aleatoriamente correos "spam" y los incorpora en el dataset hasta alcanzar 9.500 instancias de "spam". Al hacerlo, el modelo puede ver una variedad de ejemplos de la clase minoritaria sin duplicarlos exactamente.

Por último, es interesante tener en cuenta las técnicas, **SMOTE (Synthetic Minority Over-sampling Technique)**. Se trata de un conjunto de técnicas avanzadas de sobremuestreo en las que el objetivo es crear nuevos ejemplos sintéticos de la clase minoritaria en lugar de duplicar o muestrear aleatoriamente observaciones existentes. SMOTE genera cada nuevo ejemplo interpolando entre un punto de la clase minoritaria y uno de sus vecinos más cercanos, lo cual introduce variabilidad y reduce el riesgo de sobreajuste. Al crear nuevos datos, SMOTE simula variaciones que podrían existir en la clase minoritaria.

> **Ejemplo**
> En un problema de diagnóstico médico en el que la clase "enfermedad rara" representa el 5% del total de observaciones, SMOTE crea nuevos ejemplos interpolando entre instancias de "enfermedad rara" cercanas en el espacio de características, de modo que el modelo recibe una muestra más variada de esta clase. Así, SMOTE ayuda al modelo a captar patrones útiles sin replicar instancias exactas, incrementando su capacidad de generalización.

Existen variantes avanzadas de SMOTE adaptadas a escenarios específicos. Se podrían citar:

- **Borderline-SMOTE**: Genera datos sintéticos cerca de las fronteras de decisión entre clases, lo cual es útil cuando el modelo tiene dificultad en distinguir las clases en esas áreas.
- **ADASYN (Adaptive Synthetic Sampling)**: Prioriza la generación de datos en áreas donde la clase minoritaria es más difícil de distinguir de la mayoritaria. Esto adapta el número de observaciones sintéticas según la dificultad del modelo para clasificar correctamente.

Estas variantes son útiles cuando el dataset presenta características particulares, como fronteras complejas entre las clases.

El principal beneficio del sobremuestreo es que permite que la clase minoritaria esté mejor representada sin reducir el tamaño de la clase mayoritaria, lo que es especialmente valioso en datasets pequeños. Al incrementar la representación de la clase minoritaria, el modelo tiene más oportunidades de aprender patrones específicos de esta clase, mejorando su capacidad de generalización. Sin embargo, el sobremuestreo también presenta algunas limitaciones. En métodos de duplicación y sobremuestreo aleatorio, el modelo puede sobreajustarse si la información de la clase minoritaria es limitada, captando ruido en lugar de patrones generales. Por otro lado, aunque SMOTE y sus variantes reducen este riesgo, la generación de datos sintéticos puede incrementar la complejidad del modelo y, en algunos casos, introducir variabilidad que no representa con precisión la clase minoritaria.

**Para reflexionar...**

> **¿Cuándo es preferible usar SMOTE en lugar de duplicación directa de observaciones en el tratamiento de datos desbalanceados?**  
> **Clave**: Reflexiona sobre el impacto del sobreajuste en datasets pequeños y sobre cómo SMOTE introduce variabilidad sin duplicar observaciones exactas, lo que puede hacer que el modelo generalice mejor en la clase minoritaria.

#### Pesos de clase

En algunos modelos de aprendizaje supervisado, como las máquinas de soporte vectorial o los modelos de redes neuronales, es posible asignar un **peso mayor a las clases minoritarias** en el proceso de entrenamiento. Esto permite que las clases subrepresentadas tengan un impacto proporcionalmente mayor en la función de pérdida, incentivando al modelo a considerarlas con la misma importancia que a la clase mayoritaria. La técnica es eficaz para conjuntos de datos donde la creación de datos sintéticos no es viable o donde el tamaño del dataset debe mantenerse fijo.

> **Ejemplo**
> En un problema de clasificación de correos electrónicos en donde la mayoría son legítimos y solo una pequeña porción son spam, podemos aplicar SMOTE para generar más ejemplos de correos clasificados como spam. Así, el modelo aprende a detectar patrones en los correos de spam en lugar de ser dominado por los ejemplos de correos legítimos, permitiéndole capturar mejor las características distintivas del spam.

**Para reflexionar...**

> **¿Qué técnica es más adecuada para un problema de detección de fraudes en el que la clase fraudulenta es extremadamente minoritaria?** 
> **Clave**: Considera que, en problemas de fraude, es esencial evitar perder información relevante de los casos existentes, lo que hace que las técnicas de generación sintética o el uso de pesos de clase puedan ser más eficaces que el submuestreo.
