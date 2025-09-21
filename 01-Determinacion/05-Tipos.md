# 5. Tipos de Machine Learning

## Aprendizaje supervisado

El **aprendizaje supervisado** es uno de los paradigmas fundamentales del aprendizaje automático. La idea es entrenar un modelo con un conjunto de datos etiquetados, donde cada entrada está asociada con una salida o etiqueta correcta. El objetivo principal es que el modelo aprenda patrones y relaciones en los datos de manera que pueda hacer predicciones precisas cuando se le presente información nueva. A lo largo de este proceso, el modelo ajusta sus parámetros internos para minimizar el error en las predicciones que realiza.

Este enfoque tiene aplicaciones muy amplias, como la clasificación de imágenes, la predicción de precios, la detección de fraudes, entre otros. En cada uno de estos casos, **se utilizan datos históricos donde las entradas y salidas correctas ya se conocen**. Por ejemplo, en la predicción de precios de vivienda, las entradas podrían ser las características de una casa (tamaño, ubicación, número de habitaciones), y la salida sería el precio. A partir de estas asociaciones, el modelo aprende a estimar el precio de nuevas propiedades.

El aprendizaje supervisado se puede aplicar tanto a problemas de **clasificación** como de **regresión**. En los problemas de clasificación, las salidas son categorías discretas (por ejemplo, clasificar un correo como spam o no spam). En los problemas de regresión, las salidas son valores continuos (por ejemplo, predecir el valor de una propiedad).

### Ventajas y desafíos del aprendizaje supervisado

Una de las principales ventajas del aprendizaje supervisado es su capacidad para obtener resultados precisos cuando se cuenta con un conjunto de datos etiquetado y representativo. Además, debido a que las etiquetas son conocidas de antemano, el proceso de evaluación es más sencillo. Los modelos de aprendizaje supervisado son capaces de aprovechar al máximo la información contenida en los datos etiquetados, permitiendo realizar predicciones robustas y bien fundamentadas.

Sin embargo, este enfoque también presenta desafíos. Uno de los principales es la necesidad de contar con un conjunto de datos etiquetado lo suficientemente grande y representativo del problema real, ya que en ocasiones, obtener un volumen significativo de datos etiquetados puede ser bastante costoso. Por otro lado, los modelos supervisados pueden ser susceptibles al **sobreajuste** cuando los datos de entrenamiento no son lo suficientemente variados o cuando el modelo es demasiado complejo en relación con la cantidad de datos.

Otro reto importante en este enfoque es la **calidad de las etiquetas**. Si los datos de entrenamiento contienen errores en las etiquetas, el modelo aprenderá patrones incorrectos, lo que afectará negativamente a su capacidad para generalizar. La supervisión humana en el proceso de etiquetado es costosa y, a veces, se pueden cometer errores, especialmente en problemas complejos donde las etiquetas son subjetivas o ambiguas.

> [!Note]
>
> Un ejemplo histórico significativo de esfuerzo en el etiquetado de datos para modelos de aprendizaje supervisado es el **Proyecto ImageNet**. Este proyecto fue iniciado en 2007 por la investigadora **Fei-Fei Li** y su equipo en la Universidad de Stanford. El objetivo de ImageNet era crear una base de datos masiva de imágenes etiquetadas que pudieran ser utilizadas para entrenar algoritmos de visión por computadora en tareas como el reconocimiento y clasificación de imágenes.
>
> El desafío de etiquetar este vasto conjunto de datos fue enorme, ya que se recopilaron más de **14 millones de imágenes** y se clasificaron en más de **22,000 categorías**. El etiquetado se realizó en gran medida utilizando **crowdsourcing** a través de la plataforma **Amazon Mechanical Turk**, donde miles de trabajadores humanos ayudaron a identificar y etiquetar manualmente las imágenes con descripciones adecuadas.
>
> El impacto de este esfuerzo fue monumental, ya que ImageNet se convirtió en un estándar en la investigación de visión por computadora, y dio lugar al **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, que ha sido crucial para el desarrollo y el éxito de los modelos de **deep learning**. De hecho, el modelo **AlexNet**, que ganó el desafío en 2012, marcó un punto de inflexión en el campo al demostrar que las redes neuronales profundas entrenadas en grandes conjuntos de datos etiquetados pueden superar a otros métodos en el reconocimiento de imágenes.

Por último, podemos citar un desafío adicional, y es el que radica en la capacidad del modelo para generalizar a nuevas situaciones. Aunque el modelo puede aprender muy bien a partir de los datos de entrenamiento, el verdadero reto está en su capacidad para hacer predicciones precisas cuando se enfrenta a datos nuevos que no ha visto antes. En este sentido, el equilibrio entre **sesgo** y **varianza** juega un papel crucial: un modelo con mucho sesgo puede no capturar adecuadamente los patrones en los datos, mientras que un modelo con alta varianza puede sobreajustarse a los datos de entrenamiento y fallar al generalizar.

> **Ejemplo**: Un sistema de detección de fraudes bancarios puede utilizar aprendizaje supervisado entrenando un modelo con transacciones etiquetadas como fraudulentas o no. El modelo aprenderá a identificar patrones sospechosos basados en las características de las transacciones, como la ubicación o el monto de la compra.

> **Ejemplo**: En el diagnóstico médico, los modelos de aprendizaje supervisado pueden ser entrenados con imágenes etiquetadas de tumores benignos y malignos para aprender a clasificar automáticamente nuevas imágenes y asistir a los médicos en la toma de decisiones.

> **Ejemplo**: Un modelo de predicción de la demanda de productos puede entrenarse con datos históricos de ventas, inventario y factores externos como el clima o la temporada. Así, puede predecir futuras demandas, optimizando la gestión del inventario.

**Para reflexionar...**

> **¿Cuáles son las principales limitaciones del aprendizaje supervisado en proyectos de machine learning?**
> Reflexiona sobre la necesidad de grandes cantidades de datos etiquetados y los costos asociados a obtenerlos. Piensa en problemas donde las etiquetas correctas no están fácilmente disponibles.

## Aprendizaje no supervisado

El **aprendizaje no supervisado** es un enfoque del machine learning en el que el modelo se entrena con datos que no tienen etiquetas asociadas, es decir, no se conoce a priori la salida correcta para cada entrada. El objetivo principal del aprendizaje no supervisado es descubrir patrones, estructuras ocultas o agrupaciones en los datos. Este enfoque es **fundamental en situaciones donde los datos etiquetados son difíciles de obtener, costosos de generar, o simplemente no están disponibles**.

### Fundamentos

A diferencia del **aprendizaje supervisado**, donde se cuenta con una salida o etiqueta que guía al modelo, en el aprendizaje no supervisado el modelo debe aprender de las relaciones intrínsecas en los datos. Estas relaciones se basan en **similitudes, distancias o correlaciones** que existen entre las características o variables de nuestro dataset de entrada. Los algoritmos intentan encontrar una estructura subyacente o representación útil de los datos sin requerir la intervención humana para etiquetar las muestras.

Los dos problemas más comunes en el aprendizaje no supervisado son el **clustering** (agrupamiento) y la **reducción de dimensionalidad**. 

El **clustering** agrupa datos similares en diferentes conjuntos (o *clusters*) basándose en sus características propias. Un ejemplo típico es el algoritmo **k-means**, que agrupa datos en $k$ clusters, donde $k$ es un parámetro definido por el usuario. Este tipo de algoritmos se utiliza en diversas aplicaciones, como segmentación de clientes, compresión de datos, y detección de anomalías.

> [!tip]
>
> El parámetro $k$ en el algoritmo **k-means** representa el número de clusters (grupos) que se desea encontrar en un conjunto de datos. Es un valor que debe definirse previamente y determina cuántos centros de clusters se inicializarán al inicio del algoritmo. El objetivo de k-means es asignar cada punto de datos al clúster más cercano, de forma que se minimice la distancia total entre los puntos y sus centros. La elección de $k$ es crítica, ya que un valor inadecuado puede llevar a agrupamientos no representativos o ineficientes.

En cuanto a la técnica de la **reducción de Dimensionalidad**, ésta tiene como objetivo simplificar los datos reduciendo el número de características o variables, a la vez que se mantiene la mayor parte de la información relevante. La **reducción de dimensionalidad** es considerada una técnica de **aprendizaje no supervisado** porque su objetivo principal también es descubrir patrones o estructuras ocultas en los datos **sin la necesidad de etiquetas o categorías predefinidas**. En lugar de intentar predecir una salida o clasificar instancias, esta técnica transforma los datos a un espacio de menor dimensionalidad, preservando la mayor cantidad posible de información. El objetivo de la reducción de dimensionalidad es principalmente eliminar redundancias y facilitar el procesamiento y la visualización de datos, lo que es útil en una amplia variedad en tareas como **clustering** o el análisis exploratorio (EDA).

> [!tip]
>
> Por ejemplo, en el caso del algoritmo **PCA (Análisis de Componentes Principales)**, se busca encontrar las **direcciones principales de variabilidad** en los datos, lo que permite proyectar los puntos en un espacio de menor dimensión sin perder demasiada información relevante. Esto se hace **sin usar etiquetas o supervisión directa**, por lo que es considerado aprendizaje no supervisado.

> [!important]
>
> El aprendizaje no supervisado es extremadamente útil en la exploración de grandes conjuntos de datos y en la extracción de características antes de aplicar otros métodos de machine learning. A menudo, estos métodos son la primera etapa en un flujo de trabajo más complejo, donde se utilizan para identificar patrones que luego se exploran en profundidad.

### Ventajas y Desafíos

El principal beneficio del aprendizaje no supervisado es que no requiere etiquetas para entrenar el modelo, lo que lo hace aplicable en muchas áreas donde los datos etiquetados son escasos o inexistentes. Este enfoque permite descubrir patrones ocultos en datos desestructurados, algo muy complicado en tareas como la segmentación de mercado, la biología computacional, o la minería de datos.

Sin embargo, también presenta varios desafíos. Uno de los más destacados quizás es la falta de una métrica clara para evaluar el rendimiento del modelo, ya que, al no haber etiquetas, no es posible calcular métricas tradicionales como la precisión o el error. Además, la interpretación de los resultados a menudo requiere la intervención de expertos en la materia para identificar si los patrones descubiertos son significativos y útiles.

> [!note]
>
> En la década de 2000, la industria de las telecomunicaciones comenzó a utilizar ampliamente el **clustering** para segmentar a sus clientes en grupos basados en sus patrones de uso de servicios. Esto permitió a las empresas identificar diferentes perfiles de usuarios, como aquellos que consumen muchos datos o los que realizan más llamadas. Este análisis les ayudó a ajustar sus tarifas y servicios de manera más personalizada.

> **Ejemplo:** En una empresa de comercio electrónico, se pueden aplicar algoritmos de clustering para agrupar a los clientes en función de sus patrones de compra, identificando aquellos que compran productos similares o que tienen comportamientos de compra repetitivos.

> **Ejemplo:** En biología computacional, los algoritmos de reducción de dimensionalidad como PCA se utilizan para analizar datos genómicos, reduciendo miles de variables (genes) a solo unas pocas que capturan la mayor variabilidad entre muestras biológicas.

> **Ejemplo:** En la detección de fraudes financieros, se pueden utilizar técnicas de clustering para identificar transacciones sospechosas agrupando aquellas que no siguen los patrones de comportamiento comunes.

**Para reflexionar...**

> **¿Cuáles son los principales desafíos en la interpretación de los resultados del aprendizaje no supervisado y cómo se pueden mitigar?**
>
> **Pistas**: Reflexiona sobre la necesidad de intervención humana en la interpretación de los clusters y los métodos para validar los resultados en ausencia de etiquetas.

## Aprendizaje por refuerzo

El **aprendizaje por refuerzo** (RL, por sus siglas en inglés) es un enfoque del aprendizaje automático que se basa en la interacción de un **agente** con un **entorno** para tomar decisiones y optimizar su comportamiento a lo largo del tiempo. En lugar de aprender de datos etiquetados o de patrones inherentes en los datos, como en el aprendizaje supervisado o no supervisado, el agente aprende a través de la **experiencia** y de la **retroalimentación** que recibe en forma de recompensas o penalizaciones.

El esquema básico de funcionamiento sería el siguiente: Primeramente el agente realiza una acción en un estado inicial dado; a continuación el entorno responde con una recompensa y una transición a un nuevo estado. El objetivo del agente es **maximizar la recompensa acumulada** a lo largo del tiempo, lo que significa que debe aprender a tomar decisiones secuenciales que le permitan lograr su objetivo de manera eficiente. Este tipo de aprendizaje se usa ampliamente en problemas de toma de decisiones en tiempo real y en entornos dinámicos, como la robótica, los videojuegos o los sistemas de recomendación.

### Componentes clave del aprendizaje por refuerzo

Los componente de cualquier sistema de aprendizaje por refuerzo son los siguientes:

- **Agente**: La entidad que toma las decisiones.
- **Entorno**: El universo en el que el agente interactúa y en el que las decisiones tienen consecuencias.
- **Acciones**: Las decisiones que puede tomar el agente.
- **Estados**: Las representaciones del entorno en un momento dado.
- **Recompensas**: Retroalimentación que el agente recibe tras tomar una acción; puede ser positiva (recompensa) o negativa (penalización).
- **Política**: La estrategia que el agente sigue para decidir qué acciones tomar en cada estado. Puede ser determinística o probabilística.
- **Valoración**: Una función que estima el valor futuro esperado de estar en un estado particular o tomar una acción particular.
- **Exploración y explotación**: El agente debe equilibrar la **exploración** de nuevas estrategias para descubrir cuál es la mejor, y la **explotación** de lo que ya ha aprendido para maximizar las recompensas.

Uno de los principales desafíos en el aprendizaje por refuerzo es encontrar el **equilibrio entre exploración y explotación**. Si el agente se enfoca demasiado en la explotación, puede no descubrir mejores políticas o estrategias; si se enfoca demasiado en la exploración, puede no maximizar adecuadamente la recompensa. Este equilibrio es clave para un buen rendimiento del modelo en entornos dinámicos y complejos.

> [!note]
>
> Imagina un dron autónomo que debe aprender a moverse dentro de un almacén cerrado, lleno de estanterías, pasillos estrechos y obstáculos, con el objetivo de entregar paquetes de manera eficiente. Este dron no cuenta con un mapa previo del lugar, por lo que debe tomar decisiones a medida que navega por el entorno. El dron se convierte así en el protagonista del proceso de **aprendizaje por refuerzo**, donde sus decisiones se basan en la experiencia adquirida con cada vuelo.
>
> A lo largo de su recorrido, el dron interactúa continuamente con el entorno, utilizando los sensores a bordo para detectar los obstáculos cercanos y determinar su posición dentro del almacén. A cada momento, tiene la opción de moverse en diversas direcciones: puede avanzar, retroceder, girar hacia la izquierda o la derecha, o cambiar su altura para evitar colisiones. Estos movimientos representan las **acciones** del dron, que afectan directamente su posición y estado en el espacio tridimensional del almacén.
>
> Al tomar decisiones, el dron recibe **retroalimentación** en forma de recompensas y penalizaciones. Cada vez que da un paso hacia su destino sin chocar con un obstáculo, se le otorga una pequeña recompensa. Pero si tropieza con una estantería o cualquier otra barrera, recibe una penalización, lo que lo motiva a evitar comportamientos peligrosos. La mayor recompensa se obtiene al completar su misión y llegar al punto de entrega del paquete. Las **recompensas positivas** se otorgan cuando el dron realiza acciones que lo acercan a su objetivo de forma efectiva. Por ejemplo, si el dron se desplaza sin chocar con ningún obstáculo, recibe una recompensa, como +1 o +5. A medida que completa con éxito etapas de la misión, como llegar a su destino o entregar el paquete en el lugar correcto, puede obtener recompensas mayores, como +100. Estas recompensas motivan al dron a tomar decisiones que minimicen el riesgo y maximicen la eficiencia. Por otro lado, las **penalizaciones** se aplican cuando el dron comete errores. Si el dron colisiona con un obstáculo, como una pared o una estantería, recibe una penalización de, por ejemplo, -10 o -50, dependiendo de la gravedad del error. Esto desalienta al dron de repetir dichas acciones, ayudándolo a mejorar su navegación y a evitar errores futuros.
>
> La función de **recompensa** es fundamental, ya que le permite al dron aprender a lo largo del tiempo cuáles son los comportamientos que optimizan la tarea (como evitar colisiones y tomar rutas eficientes) y cuáles debe evitar, logrando una navegación más segura y precisa.
>
> A medida que el dron acumula experiencia, comienza a **aprender** cuáles son las mejores decisiones para maximizar su eficiencia. Si bien al principio explora varias rutas y toma decisiones al azar, con el tiempo ajusta su **política** de navegación. Por ejemplo, aprende que ciertas áreas del almacén están más congestionadas y que debe evitarlas, mientras que otros pasillos permiten un tránsito más rápido. Esta política de decisiones optimizada se refina cada vez que el dron vuela por el almacén, permitiéndole mejorar gradualmente sus entregas.
>
> En este proceso, el dron también enfrenta el desafío de equilibrar **exploración** y **explotación**. Durante sus primeros vuelos, debe explorar nuevas rutas y posibilidades, incluso si algunas no parecen ser las mejores opciones. Sin embargo, a medida que gana experiencia, comienza a explotar las rutas que ya ha identificado como las más eficientes, minimizando el riesgo de accidentes y optimizando el tiempo de entrega.
>
> Así, el dron autónomo, mediante **aprendizaje por refuerzo**, aprende de sus errores, ajusta su comportamiento y se convierte en un mensajero cada vez más eficiente, capaz de navegar entornos complejos sin instrucciones específicas más allá de las recompensas y penalizaciones que recibe en cada vuelo.

### Tipos de algoritmos en aprendizaje por refuerzo

En el aprendizaje por refuerzo, los algoritmos se pueden clasificar en según tres enfoques principales: los que están **basados en valor**, los **basados en política**, y los basados en**métodos mixtos**. Cada uno de ellos aborda el problema de toma de decisiones de manera diferente, dependiendo de cómo el agente aprende y optimiza sus acciones para maximizar las recompensas en su entorno.

Los **Algoritmos basados en valor** están enfocados en aprender el valor de los estados o las acciones que el agente puede tomar. Un ejemplo clásico es el **Q-learning**, donde el agente construye una tabla conocida como **Q-table**. Esta tabla asigna a cada estado o acción un valor que representa el beneficio esperado a largo plazo de tomar esa acción en ese estado. A medida que el agente explora su entorno y recibe recompensas, actualiza la Q-table, lo que le permite tomar decisiones cada vez más acertadas. El objetivo es que el agente aprenda a identificar la mejor acción a tomar en cualquier estado dado, basándose en los valores almacenados en la tabla.

A diferencia de los métodos basados en valor, los **algoritmos basados en política** no intentan estimar explícitamente el valor de cada estado o acción. En su lugar, aprenden directamente una **política**, es decir, una función que mapea cada estado a una acción específica. Un ejemplo destacado es el **gradiente de políticas**, donde el agente ajusta su política a lo largo del tiempo mediante optimización, de manera que mejora gradualmente las decisiones que toma en cada situación. Este enfoque es particularmente útil en entornos con grandes espacios de estados, donde estimar valores para cada estado sería impracticable.

Por último, los **métodos mixtos** combinan elementos de los algoritmos basados en valor y los basados en política. Un ejemplo típico es el método **actor-crítico**, donde se entrenan dos componentes en paralelo: el **actor**, que es responsable de aprender la política (qué acciones tomar en cada estado), y el **crítico**, que evalúa la calidad de las decisiones del actor al estimar los valores de los estados. Esta combinación permite que el agente aproveche tanto la flexibilidad de las políticas aprendidas como la estabilidad de los valores estimados, logrando una mejora más robusta y efectiva en su rendimiento.

Estos enfoques permiten a los agentes de aprendizaje por refuerzo adaptarse a diferentes tipos de problemas, desde aquellos que requieren un análisis detallado de los valores de las acciones hasta otros que necesitan decisiones rápidas basadas en políticas dinámicas.

> **Ejemplo**: En los **videojuegos**, los personajes controlados por inteligencia artificial pueden usar aprendizaje por refuerzo para aprender a tomar decisiones estratégicas en función de su entorno y sus adversarios. A medida que juegan más partidas, mejoran sus habilidades al obtener recompensas por ganar o completar objetivos dentro del juego.

> **Ejemplo**: En **robótica**, un robot puede usar aprendizaje por refuerzo para aprender a navegar por una habitación llena de obstáculos. A través de prueba y error, el robot descubre la mejor ruta para evitar colisiones y alcanzar su objetivo, aprendiendo de la retroalimentación recibida al chocar o al alcanzar el destino.

> **Ejemplo**: En los sistemas de **negociación automatizada**, los agentes de software pueden usar RL para mejorar su capacidad de hacer ofertas y contraofertas basadas en la respuesta de sus contrapartes, maximizando así sus ganancias o acuerdos satisfactorios.

**Para reflexionar...**

> **¿Cómo puede afectar el proceso de aprendizaje por refuerzo en situaciones donde las consecuencias de una decisión incorrecta sean críticas, como en el ámbito médico o legal?**
>
> **Pistas**: Reflexiona sobre la importancia de tener sistemas de respaldo y la posibilidad de errores graves cuando el agente está en fase de exploración.

## Redes neuronales

Las **redes neuronales** son modelos inspirados en las neuronas del cerebro humano, formadas por nodos que procesan señales a través de capas. Su origen se puede hallar en el **perceptrón**, desarrollado por Frank Rosenblatt en 1957, un modelo computacional simple que intentaba emular una neurona biológica. El perceptrón realizaba una combinación lineal de las entradas y aplicaba una función de activación para tomar decisiones. A su vez, este modelo tenía una relación directa con la **regresión lineal**, donde las predicciones se basan en una combinación lineal de variables de entrada, si bien es cierto que el perceptrón añade la capacidad de clasificar mediante activaciones no lineales.

Así pues, las **redes neuronales artificiales** (ANN, por sus siglas en inglés) son modelos computacionales que se organizan en nodos y en capas. Cada capa está compuesta de nodos que se conectan con uno o más nodos de la siguiente capa. Una neurona artificial consta de una **capa de entrada**, una o más **capas ocultas**, y una **capa de salida**. Cada conexión entre nodos de distintas capas está caracterizada por un parámetro  denominado *peso*, que representa la fuerza de la conexión entre nodos.

En general, estos modelos son fundamentales en el campo del **aprendizaje profundo (deep learning)**, una subcategoría del aprendizaje automático diseñada para trabajar con grandes volúmenes de datos, tanto estructurados como no estructurados, y para resolver problemas altamente complejos.

### Estructura de las redes neuronales

Las redes neuronales suelen estar estructuradas en tres tipos de capas. En primer lugar una **capa de entrada**, que recibe los datos iniciales y los transmite a la siguiente capa. Cada nodo en esta capa **corresponde a una característica del conjunto de datos**.  En segundo lugar, las denominadas **capas ocultas**. Estas capas transforman las entradas utilizando funciones de activación y producen características más abstractas o complejas. La **profundidad** de una red (es decir, el número de capas ocultas) es lo que define un modelo como **red profunda** (deep neural network). Por último, la **capa de salida**, genera la predicción final o la clasificación, dependiendo de la naturaleza del problema. La salida puede ser un valor numérico, una categoría, o un conjunto de probabilidades.

Las redes neuronales tienen la capacidad de **aprender representaciones jerárquicas** de los datos. Esto significa que, a medida que la información avanza a través de las capas ocultas, el modelo extrae características cada vez más abstractas y complejas. Este proceso es crucial en tareas como el **reconocimiento de imágenes**, donde las primeras capas pueden detectar bordes y contornos simples, mientras que las últimas capas identifican formas y objetos más complejos.

### Funcionamiento básico

Cada nodo en una red neuronal realiza una operación matemática: toma una serie de entradas, las multiplica por los pesos correspondientes y les aplica una función de activación para producir una salida. Este proceso se repite en todas las capas hasta que se llega a la capa de salida.

Para hacer que el modelo aprenda, es necesario **entrenar** la red y optimizar los pesos de cada conexión. Durante esta fase, el modelo recibe datos etiquetados (en el caso del aprendizaje supervisado) y realiza predicciones. A continuación, compara las predicciones con los valores reales para calcular el **error** (o pérdida), que se utiliza para ajustar los pesos de la red. Este ajuste se lleva a cabo mediante una técnica llamada **retropropagación del error**.

### Retropropagación del error

La **retropropagación** es uno de los mecanismos clave en el entrenamiento de redes neuronales. Funciona ajustando los pesos de las conexiones neuronales en función del error cometido en la predicción. Este proceso se lleva a cabo en varias etapas. Primero, realizando un**cálculo del error**. Se compara la salida generada por la red con el valor real mediante una **función de pérdida** que depende del tipo de problema (regresión o clasificación). En segundo lugar, el error calculado se propaga desde la capa de salida hacia las capas anteriores, ajustando los pesos de cada conexión. Cuanto mayor sea el error en una capa, mayor será el ajuste de los pesos en esa capa. Por último se **actualizan los pesos**. Para se aplican algoritmos de optimización como el **gradiente descendente** para minimizar la función de pérdida y actualizar los pesos de manera iterativa. Este proceso de ajuste continuo de los pesos durante múltiples iteraciones, o **épocas**, permite a la red neuronal mejorar su capacidad para hacer predicciones precisas a partir de los datos.

### Funciones de activación

Las funciones de activación son otro componente esencial de las redes neuronales. Estas funciones definen la salida de una neurona, dado un conjunto de entradas ponderadas. Existen diferentes tipos de funciones de activación, cada una con sus propias características y aplicaciones. La más simple es la **función sigmoide**, la cual comprime los valores de salida entre 0 y 1 y la hace útil en problemas de clasificación binaria. Una variante de esta función es la función *tangente hiperbólica* (**Tanh**), que es similar a la sigmoide, pero comprimiendo los valores entre -1 y 1. Es habitual usarla en las capas ocultas de una red. Por último, podemos citar la función **ReLU (Rectified Linear Unit)**. Es la función de activación más utilizada en redes profundas. Introduce no linealidad al modelo y evita el problema del **desvanecimiento del gradiente**, lo que facilita el entrenamiento en este tipo de redes.

> [!tip]
>
> El **desvanecimiento del gradiente** es un problema que ocurre durante el entrenamiento de redes neuronales profundas, cuando los gradientes (derivadas parciales de la función de pérdida con respecto a los pesos) se vuelven extremadamente pequeños al propagarse hacia atrás a través de las capas de la red. Esto hace que las actualizaciones de los pesos sean insignificantes en las primeras capas, ralentizando el aprendizaje o impidiendo que ocurra. Este problema es especialmente común en las **redes neuronales profundas**  cuando se usan funciones de activación como la **sigmoide** o **tanh**. Es posible mitigar este efecto utilizando funciones de activación como **ReLU** o variantes.

### Desafíos de las redes neuronales

A pesar de su capacidad para aprender representaciones complejas, las redes neuronales también presentan varios desafíos. En primer lugar su **opacidad y falta de explicabilidad**. Es uno de los mayores retos de este tipo de modelos. A medida que la red se vuelve más compleja, resulta difícil entender cómo se toman las decisiones, lo que plantea problemas en áreas donde se requiere transparencia. En segundo lugar podríamos citar el problema del **sobreajuste**. Dado que las redes neuronales suelen tener un gran número de parámetros, corren el riesgo de memorizar los datos de entrenamiento en lugar de generalizar correctamente a nuevos datos. Para mitigar este problema, se emplean técnicas de regularización como el **dropout**, que desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento, o la **regularización L2**, que penaliza modelos con pesos demasiado grandes. Por último, no puede obviarse el **coste computacional**. Entrenar redes neuronales profundas requiere una gran cantidad de recursos computacionales, especialmente cuando se trabaja con grandes conjuntos de datos. Por esta razón, muchas veces es necesario utilizar **unidades de procesamiento gráfico (GPUs)** o **unidades de procesamiento tensorial (TPUs)** para acelerar el entrenamiento.

### Aplicaciones de las redes neuronales

Las redes neuronales han demostrado ser herramientas muy potentes en una amplia variedad de aplicaciones. Aquí enumeramos algunos ejemplos prácticos

> **Ejemplo:** En la detección de fraudes en transacciones financieras, se pueden usar redes neuronales para aprender patrones complejos en las transacciones. Estos modelos pueden detectar comportamientos anómalos que sugieren posibles fraudes, ofreciendo una precisión superior a otros enfoques.

> **Ejemplo:** En el procesamiento del lenguaje natural, las redes neuronales se usan para tareas como la traducción automática o la generación de texto. Las redes neuronales recurrentes (RNN) y las arquitecturas más avanzadas como **transformers** permiten analizar y generar secuencias de palabras, mejorando la comprensión y generación de lenguaje.

> **Ejemplo:** En el diagnóstico médico, las redes neuronales pueden analizar imágenes médicas para detectar enfermedades como el cáncer. Modelos entrenados con grandes cantidades de datos de imágenes pueden aprender a identificar tumores en etapas tempranas con una precisión comparable a la de los radiólogos.

**Para reflexionar...**

> **¿Qué técnicas adicionales podríamos utilizar para mejorar la interpretabilidad de las redes neuronales en aplicaciones críticas como la medicina o el derecho?**
>
> Considera técnicas como las redes neuronales explicables (XAI), el uso de saliency maps, y la implementación de modelos más simples en combinación con redes profundas para mejorar la transparencia en la toma de decisiones.
