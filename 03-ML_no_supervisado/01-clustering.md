# Sistemas de aprendizaje automático no-supervisado: Clustering

## Objetivos del módulo

> - Comprender el concepto de **clustering** y su aplicación en la segmentación de datos sin etiquetas.
> - Explorar los principales **algoritmos de clustering**, sus características y diferencias.
> - Aprender a seleccionar el **número óptimo de clusters** y evaluar la calidad del agrupamiento.
> - Implementar técnicas de clustering en Python utilizando la librería **scikit-learn**.
> - Aplicar el clustering a problemas prácticos en distintos dominios como marketing, biología o las finanzas.

---

## **Introducción**

El clustering es una de las técnicas más fascinantes dentro del **aprendizaje automático no supervisado**. A diferencia de los modelos supervisados, donde el objetivo es aprender una relación entre los datos de entrada y una salida conocida, el clustering permite descubrir **estructuras ocultas** en los datos sin necesidad de contar con etiquetas previas.

Imagina que tienes una gran cantidad de información sin clasificar, como registros de clientes, documentos de texto o imágenes. En lugar de etiquetar manualmente cada elemento, el clustering permite encontrar **grupos naturales** dentro de esos datos, agrupando elementos similares y separando aquellos que son diferentes.

El concepto clave detrás del clustering es la **similitud**, es decir, encontrar qué datos comparten características comunes y cómo pueden ser agrupados en conjuntos homogéneos. Cada uno de estos grupos se denomina **clúster**, y el objetivo del proceso de clustering es asegurarse de que los datos dentro de un mismo clúster sean lo más similares posible entre sí, mientras que los datos en clústeres diferentes deben ser lo más distintos posible.

### **El propósito del clustering**

Cuando se aplica clustering a un conjunto de datos, se persiguen dos propósitos esenciales. Por un lado, es importante lograr que los elementos que comparten características similares queden **agrupados dentro del mismo clúster**, garantizando que la información relevante se concentre en subconjuntos bien definidos. Por otro lado, también es crucial que los clústeres resultantes estén **claramente diferenciados entre sí**, de modo que se puedan identificar fácilmente distintos patrones de comportamiento.

Por ejemplo, si pensamos en un análisis de clientes de una tienda en línea, podríamos encontrar grupos de personas con hábitos de compra similares: aquellos que prefieren comprar tecnología, otros que se enfocan en productos de moda, o incluso clientes que realizan compras con poca frecuencia pero de alto valor. Identificar estos segmentos permite a la empresa diseñar estrategias personalizadas para cada tipo de cliente.

Este equilibrio entre la **homogeneidad dentro de los clústeres** y la **heterogeneidad entre ellos** es lo que hace que el clustering sea una herramienta tan poderosa para comprender datos no estructurados.

### **Clustering vs. clasificación supervisada**

A primera vista, el clustering puede parecer similar a la clasificación supervisada, ya que ambos buscan categorizar datos. Sin embargo, hay diferencias fundamentales que los distinguen. La clasificación supervisada se basa en un conjunto de datos previamente etiquetado, donde el modelo aprende a asignar etiquetas a nuevas observaciones basándose en experiencias pasadas. Por el contrario, en el clustering no existen etiquetas conocidas, por lo que el objetivo es **descubrir** estructuras dentro de los datos sin una orientación previa.

Otra diferencia clave radica en el uso que se hace de los modelos. Mientras que en clasificación supervisada se busca asignar correctamente las etiquetas en nuevas muestras, el clustering se utiliza principalmente para **exploración y descubrimiento de patrones**. Esto lo hace ideal en situaciones donde no se dispone de información previa sobre las categorías existentes o cuando se quiere obtener una comprensión más profunda de los datos antes de tomar decisiones.

Imaginemos, por ejemplo, que una empresa quiere analizar su base de clientes, pero no tiene etiquetas claras que definan distintos tipos de usuario. En este caso, aplicar un algoritmo de clustering puede revelar segmentos ocultos que posteriormente podrían etiquetarse y utilizarse en modelos supervisados.

### **Aplicaciones del clustering en el mundo real**

El clustering tiene una amplia variedad de aplicaciones en múltiples sectores, y su capacidad para descubrir patrones ocultos lo convierte en una herramienta fundamental en la toma de decisiones. Una de sus aplicaciones más comunes es en el ámbito del **marketing**, donde las empresas buscan segmentar a sus clientes según su comportamiento de compra, preferencias o nivel de gasto. Al identificar grupos de clientes con intereses similares, se pueden diseñar campañas de publicidad más efectivas y productos adaptados a cada segmento.

Otro ámbito donde el clustering resulta esencial es en la **detección de anomalías**, como en sistemas de seguridad informática o en el análisis de fraudes financieros. Al agrupar comportamientos típicos, se pueden identificar transacciones que se desvían de la norma, lo que permite detectar posibles actividades fraudulentas antes de que se conviertan en un problema.

En el campo de la **biología y la medicina**, el clustering se utiliza para identificar patrones en datos genómicos o agrupar pacientes según similitudes en sus condiciones de salud. Por ejemplo, al analizar datos de pacientes con enfermedades similares, los médicos pueden desarrollar tratamientos más personalizados y eficaces.

Además, en aplicaciones como la **segmentación de imágenes**, el clustering permite identificar regiones similares en una imagen, lo que se usa ampliamente en diagnóstico médico, análisis satelital y sistemas de visión por computadora.

Otro caso interesante se encuentra en la **personalización de contenido web**, donde plataformas como Netflix o Spotify agrupan a los usuarios según sus preferencias de consumo para recomendarles contenido relevante basado en lo que disfrutan otros usuarios con gustos similares.

> **Ejemplo:**
>
> Supongamos que una empresa de telecomunicaciones quiere comprender mejor a sus clientes a partir de datos de ingresos y gastos. Sin necesidad de conocer etiquetas previamente, pueden aplicar un algoritmo de clustering y descubrir cuatro grupos bien diferenciados:
>
> - **Clientes con bajos ingresos y pocos gastos**, que podrían estar interesados en planes económicos.
> - **Clientes con ingresos medios pero ahorradores**, a quienes podrían ofrecerles paquetes de ahorro o incentivos.
> - **Clientes con ingresos medios y gastos elevados**, que podrían ser un mercado potencial para productos de gama alta.
> - **Clientes con ingresos altos y gastos altos**, que podrían estar interesados en servicios premium y exclusivos.
>
> Con esta información, la empresa podría personalizar su oferta y mejorar la retención de clientes, optimizando sus estrategias de marketing en función de las características de cada grupo.

------

**Para reflexionar...**

> **¿Por qué el clustering es una herramienta tan valiosa cuando no se dispone de etiquetas en los datos?**
>
> **Clave:** Reflexiona sobre la capacidad del clustering para descubrir patrones ocultos que podrían pasar desapercibidos en un análisis superficial.

## **Conceptos clave en clustering**

El clustering es una técnica de aprendizaje no supervisado que permite encontrar estructuras ocultas en los datos a partir de su similitud. Para comprender su funcionamiento, es esencial conocer algunos conceptos fundamentales que determinan la calidad del agrupamiento y guían la selección de los algoritmos más adecuados para cada problema. Entre estos conceptos, la **similitud entre datos**, la representación en espacios de distintas dimensiones y la elección de métricas de distancia juegan un papel crucial. Asimismo, la manera en que se define un clúster, considerando elementos como su centroide, baricentro o densidad, son clave para la interpretación y aplicación práctica de los resultados. 

### **La noción de similitud entre datos**

El éxito del clustering depende de la capacidad de identificar **datos similares**, es decir, aquellos que comparten ciertas características. La similitud entre dos observaciones se evalúa a través de una medida cuantificable que permite determinar qué tan cerca o lejos están en el espacio de representación.

Esta similitud puede estar basada en distintas propiedades, como la proximidad en términos numéricos o la coincidencia de características en datos categóricos. Por ejemplo, en un conjunto de datos de clientes, la similitud podría evaluarse a partir de atributos como la edad, el nivel de ingresos o los hábitos de compra. Sin embargo, la elección de qué características considerar y cómo medir su similitud es un aspecto crítico que influye en la calidad de los grupos formados.

Es importante destacar que en algunos casos, los datos pueden ser similares bajo ciertas condiciones y diferentes bajo otras. Por ejemplo, dos clientes podrían ser similares en sus hábitos de gasto, pero completamente distintos en sus preferencias de productos. Esto resalta la importancia de comprender el contexto del problema antes de aplicar una técnica de clustering.

### **Espacios de representación y dimensionalidad**

Los datos con los que trabajamos en clustering se representan en espacios multidimensionales, donde cada dimensión corresponde a una característica o variable. En un espacio bidimensional, es fácil visualizar la relación entre dos variables, pero en problemas del mundo real, los datos suelen tener muchas más dimensiones, lo que dificulta su representación e interpretación.

La **alta dimensionalidad** plantea desafíos importantes. A medida que se incrementa el número de dimensiones, las observaciones tienden a dispersarse en el espacio, y la diferencia entre puntos cercanos y lejanos se vuelve menos significativa. Este fenómeno, conocido como la **maldición de la dimensionalidad**, puede afectar la capacidad del algoritmo para encontrar agrupaciones significativas, ya que las distancias entre puntos tienden a homogenizarse y los datos pierden estructura discernible.

Para hacer frente a estos problemas, a menudo se utilizan técnicas de **reducción de dimensionalidad**, como el Análisis de Componentes Principales (PCA) o t-SNE, que permiten representar los datos en un espacio de menor dimensión conservando la mayor cantidad posible de información relevante. Estas técnicas no solo facilitan la visualización, sino que también pueden mejorar el rendimiento de los algoritmos de clustering al eliminar redundancias y ruido en los datos.

### **Métricas de distancia**

La forma en que medimos la distancia entre puntos es crucial para definir la similitud y, por ende, la agrupación en clústeres. Diferentes algoritmos de clustering utilizan diversas métricas de distancia que influyen en la forma en que se construyen los grupos. Algunas de las más utilizadas se enumeran a continuación

#### **Distancia Euclidiana: "La distancia en línea recta"**

La **distancia Euclidiana** es la más intuitiva, ya que mide la separación en línea recta entre dos puntos en el espacio, como si midieras con una regla.Su expresión matemática es la siguiente

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

> **Ejemplo:**
>
> Imagina que dos tiendas están ubicadas en un plano cartesiano. La tienda A está en la coordenada $(2,3)$ y la tienda B en $(5,7)$. La distancia Euclidiana entre ellas sería:
>
> $$
> \sqrt{(5-2)^2 + (7-3)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
> $$

Si representamos este cálculo en un mapa, la distancia Euclidiana sería la línea recta más corta entre ambas tiendas. Esta métrica es ideal cuando las características tienen la misma escala y no están correlacionadas. Sin embargo, en datos de alta dimensionalidad, la distancia Euclidiana puede volverse poco discriminativa debido a la maldición de la dimensionalidad.

#### **Distancia Manhattan: "Moverse por una cuadrícula"**

A diferencia de la distancia Euclidiana, la **distancia Manhattan** mide la suma de las diferencias absolutas entre las coordenadas, desplazándose en líneas rectas paralelas a los ejes, como si te movieras por las calles de una ciudad en forma de cuadrícula. Su expresión es:

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

> **Ejemplo:**
>
> Supongamos que quieres caminar de un punto a otro en una ciudad con calles en cuadrícula. Si el punto A está en $(2,3)$ y el punto B en $(5,7)$, la distancia Manhattan sería:
>
> $$
> |5 - 2| + |7 - 3| = 3 + 4 = 7
> $$
> 
> Visualmente, en lugar de tomar la ruta más corta en línea recta (como en la distancia Euclidiana), te moverías primero hacia la derecha y luego hacia arriba siguiendo las calles de la ciudad.
>

La distancia Manhattan es útil en situaciones donde las variables representan dimensiones separadas, como en la logística y el procesamiento de imágenes. 

#### **Distancia de Minkowski: "Una métrica generalizada"**

La **distancia de Minkowski** es una generalización de las distancias Euclidiana y Manhattan. Introduce un parámetro $p$ que ajusta la fórmula para comportarse como una u otra. Su expresión matemática sería:

$$
d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

> **Ejemplo:**
>
> Si establecemos $p = 2$, la fórmula se convierte en la distancia Euclidiana, mientras que si usamos $p = 1$, obtenemos la distancia Manhattan.
>
> Imagina un punto A en $(2,3)$ y un punto B en $(5,7)$. Si usamos Minkowski con $p = 3$:
>
> $$
> \left( |5 - 2|^3 + |7 - 3|^3 \right)^{\frac{1}{3}} = (27 + 64)^{\frac{1}{3}} = 91^{\frac{1}{3}} \approx 4.49
> $$

Esta métrica permite ajustar el valor de $p$ según las características de los datos, proporcionando flexibilidad en problemas donde no está claro qué tipo de distancia es más adecuada.

#### **Distancia de Hamming: "Contando diferencias"**

Cuando se trabaja con datos categóricos, la **distancia de Hamming** resulta particularmente útil. Esta métrica cuenta la cantidad de posiciones en las que dos secuencias de caracteres difieren, lo que la hace ideal para comparar cadenas de texto, secuencias de ADN o códigos binarios.Su expresión es:

$$
d(x, y) = \sum_{i=1}^{n} \mathbb{1}(x_i \neq y_i)
$$

> **Ejemplo:**
>
> Supongamos que tenemos dos códigos de producto representados como cadenas binarias:
>
> - Producto A: `101110`
> - Producto B: `100100`
>
> Comparando posición por posición, observamos diferencias en las posiciones 2 y 5, lo que da una distancia de Hamming de 2.
>

La distancia de Hamming se emplea en aplicaciones como la detección de errores en transmisión de datos y en el análisis de similitud en secuencias biológicas.

#### **Comparación práctica de las métricas**

Para ilustrar la diferencia entre estas métricas, supongamos que queremos comparar tres puntos en un plano: 

- Punto A: $(1,1)$ 
- Punto B: $(4,5)$ 
- Punto C: $(1,5)$ 

Las distancias entre A y B serían: 

| Métrica | Distancia A-B|
| ----------------- | -------------------------------------- |
| Euclidiana| $\sqrt{(4-1)^2 + (5-1)^2} = 5$ |
| Manhattan | \|4-1\| + \|5-1\| = 7 |
| Minkowski ($p=3$) | $(27 + 64)^{\frac{1}{3}} \approx 4.64$ |

Las distancias entre A y C serían:

| Métrica | Distancia A-C|
| ----------------- | ---------------------------------- |
| Euclidiana| $\sqrt{(1-1)^2 + (5-1)^2} = 4$ |
| Manhattan | \|1-1\|+ \|5-1\|= 6 |
| Minkowski ($p=3$) | $(0 + 64)^{\frac{1}{3}} \approx 4$ |

Esto demuestra cómo cada métrica considera la relación entre los puntos de manera diferente, lo que puede influir en los resultados del clustering.

**Para reflexionar...** 

> **¿Cómo influiría el uso de la distancia Manhattan en un problema de clustering de clientes basado en sus hábitos de compra?** 
> **Clave:** Piensa en cómo la estructura del problema influye en la elección de la métrica adecuada.



> [!warning]
>
> Elegir la métrica de distancia correcta es crucial para el éxito del clustering. Mientras que la distancia Euclidiana es adecuada para datos numéricos homogéneos, la distancia Manhattan resulta útil cuando las variables representan dimensiones independientes. Por otro lado, la distancia de Hamming es imprescindible en problemas con datos categóricos, y la distancia de Minkowski ofrece flexibilidad para adaptar el análisis a las características de los datos.
>
> Cada métrica tiene sus ventajas y desventajas, y su elección debe basarse en una comprensión profunda del dominio del problema y la naturaleza de los datos.

### **Centroide, baricentro y densidad: elementos clave en la formación de clústeres**

Los algoritmos de clustering suelen basarse en la idea de que cada clúster puede representarse mediante un **punto central**, que resume la información del grupo. Este punto puede adoptar distintas definiciones según el método utilizado.

El **centroide** de un clúster es el punto promedio de todas las observaciones contenidas en él y se utiliza en algoritmos como K-Means para representar el centro geométrico del grupo. Se calcula como la media aritmética de las coordenadas de todos los puntos en el clúster, proporcionando una representación eficaz cuando la **distancia Euclidiana** es una métrica adecuada para medir similitudes. Matemáticamente, el centroide de un conjunto de $n$ puntos $x_1, x_2, \dots, x_n$ en un espacio de dimensión $d$ se define como:

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

donde:
- $\mu$ representa el centroide del clúster,
- $n$ es el número total de puntos en el clúster,
- $x_i$ es el vector de características de cada punto de datos.

Por otro lado, el **baricentro** es una extensión del concepto de centroide que incorpora ponderaciones asignadas a cada observación, reflejando con mayor precisión la estructura del grupo cuando ciertas observaciones tienen una mayor importancia relativa. En este caso, cada punto $x_i$ se multiplica por un peso asociado $w_i$, y el baricentro se obtiene calculando la media ponderada de los puntos del clúster:

$$
\mu_{\text{ponderado}} = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
$$

donde:
- $w_i$ representa el peso asignado a cada punto,
- $\sum_{i=1}^{n} w_i$ es la suma total de los pesos,
- $\mu_{\text{ponderado}}$ es el baricentro del clúster.

El uso del baricentro es especialmente útil en situaciones donde las observaciones tienen diferentes niveles de importancia, como en el análisis de clientes donde se asigna un peso mayor a aquellos con mayor valor de compra o en aplicaciones donde se desea minimizar la influencia de valores atípicos asignándoles un peso menor.

En conclusión, mientras que el centroide ofrece una visión equitativa de todos los puntos de un clúster, el baricentro permite una representación más ajustada cuando se requieren consideraciones ponderadas, lo que lo convierte en una herramienta poderosa para escenarios en los que no todos los datos tienen la misma relevancia.

Otro enfoque en la formación de clústeres se basa en la **densidad**, concepto utilizado por algoritmos como DBSCAN, que definen un clúster como una región del espacio donde los puntos están densamente agrupados. Esta técnica permite descubrir estructuras más complejas y detectar datos atípicos, pero puede ser más sensible a la elección de los parámetros de densidad.

### **Características de un buen clustering**

Para que el clustering proporcione resultados útiles, es importante que los grupos obtenidos cumplan con ciertas características deseables. Un clúster debe ser **compacto**, lo que significa que sus puntos deben estar lo más cerca posible del centroide o baricentro, asegurando la homogeneidad dentro del grupo. Además, los clústeres deben estar **bien separados**, evitando solapamientos que dificulten la interpretación y el uso práctico de los resultados.

El tamaño del clúster es otro aspecto fundamental; grupos demasiado pequeños pueden no representar patrones generales, mientras que grupos excesivamente grandes pueden carecer de especificidad. Finalmente, un buen clustering debe ser **relevante para el contexto del problema**, es decir, debe proporcionar información significativa que permita la toma de decisiones informadas en el ámbito de aplicación, ya sea en marketing, biología, finanzas o cualquier otro campo. 

**Para reflexionar...**

> **¿Cómo influye la elección de la métrica de distancia en la formación de clústeres?** 
> **Clave:** Piensa en cómo diferentes métricas afectan la percepción de la similitud entre los puntos.

## Estrategias de clustering en aprendizaje no supervisado

El clustering es una de las técnicas más utilizadas en el aprendizaje no supervisado para descubrir patrones ocultos en los datos, agrupando observaciones similares en conjuntos homogéneos. A lo largo de los años, se han desarrollado diversos enfoques para abordar el problema de la agrupación, cada uno con su propia perspectiva sobre cómo deben organizarse los datos y qué criterios deben emplearse para definir la similitud entre ellos.

La elección del enfoque de clustering adecuado depende en gran medida de la estructura subyacente de los datos, la naturaleza del problema a resolver y los objetivos específicos del análisis. Por ejemplo, algunos algoritmos asumen que los datos están distribuidos de manera uniforme en el espacio, mientras que otros pueden adaptarse mejor a conjuntos de datos con formas irregulares o distribuciones complejas.

A grandes rasgos, las estrategias de clustering se pueden agrupar en cuatro categorías principales, cada una con sus propios métodos, ventajas y desafíos. Estas categorías permiten abordar problemas de segmentación desde distintas perspectivas, dependiendo de las características de los datos y del tipo de información que se busca extraer.

### Clustering basado en particiones

Uno de los enfoques más utilizados en clustering es el basado en particiones, el cual busca dividir el conjunto de datos en un número predefinido de clústeres de forma tal que cada observación pertenezca a un único grupo. La idea central de estos métodos es optimizar un criterio de similitud o disimilitud, de modo que los puntos dentro de un mismo clúster sean lo más similares posible, mientras que los puntos de diferentes clústeres estén claramente separados.

El algoritmo más representativo de este enfoque es **K-Means**, que funciona asignando cada punto de datos al clúster cuyo centroide esté más cercano, recalculando iterativamente estos centroides hasta alcanzar una configuración óptima. Aunque K-Means es eficiente y fácil de interpretar, presenta ciertas limitaciones, como su sensibilidad a la inicialización de los centroides y su incapacidad para manejar clústeres de formas arbitrarias.

Otro aspecto a considerar en los algoritmos de partición es la necesidad de definir el número de clústeres de antemano, lo que puede resultar un desafío cuando no se tiene información previa sobre la estructura de los datos. Técnicas como el método del codo o el coeficiente de silueta se utilizan a menudo para ayudar a determinar el número óptimo de clústeres.

### Clustering jerárquico

El clustering jerárquico ofrece un enfoque diferente, basado en la construcción de una estructura en forma de árbol, conocida como **dendrograma**, que muestra las relaciones entre los datos en distintos niveles de granularidad. A diferencia de los métodos basados en particiones, el clustering jerárquico no requiere especificar el número de clústeres desde el principio, lo que permite una exploración más flexible de la estructura de los datos.

Existen dos estrategias principales en el clustering jerárquico: el **método aglomerativo**, que comienza con cada punto como un clúster individual y fusiona gradualmente los más similares hasta formar un único grupo; y el **método divisivo**, que parte de todos los puntos en un solo clúster y los divide recursivamente hasta llegar a clústeres individuales.

Una de las ventajas del clustering jerárquico es su capacidad para proporcionar una visión global de la organización de los datos, lo que facilita la interpretación y el análisis exploratorio. Sin embargo, su complejidad computacional puede ser una limitante, especialmente en conjuntos de datos de gran tamaño.

### Clustering basado en densidad

El clustering basado en densidad adopta una perspectiva distinta, identificando agrupaciones de puntos en función de su concentración en ciertas regiones del espacio. En lugar de basarse en distancias globales, estos métodos detectan regiones densas de puntos separadas por áreas de baja densidad, permitiendo descubrir estructuras de formas complejas sin la necesidad de definir el número de clústeres previamente.

El algoritmo más representativo de esta categoría es **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, el cual clasifica los puntos de datos en función de su vecindad, definida por un radio de búsqueda ($\varepsilon$) y un número mínimo de puntos dentro de esa vecindad ($MinPts$). DBSCAN permite identificar clústeres de formas arbitrarias y detectar valores atípicos, lo que lo hace especialmente útil en escenarios donde los datos contienen ruido o regiones de diferentes densidades.

A pesar de sus ventajas, la efectividad de los algoritmos basados en densidad depende en gran medida de la correcta elección de los parámetros, lo que puede ser un desafío en conjuntos de datos con densidades variables. Sin embargo, su capacidad para adaptarse a estructuras de datos complejas lo convierte en una herramienta poderosa en aplicaciones como el análisis geoespacial, la detección de fraudes y la segmentación de imágenes.

### Modelos probabilísticos

Los modelos probabilísticos de clustering adoptan un enfoque basado en estadísticas, donde se asume que los datos provienen de una combinación de distribuciones subyacentes. En lugar de asignar determinísticamente cada punto a un clúster, estos métodos calculan la **probabilidad de pertenencia** de cada observación a los distintos clústeres, proporcionando una clasificación más flexible y probabilística.

Uno de los enfoques más utilizados dentro de esta categoría es el **modelo de mezcla de gaussianas (GMM, Gaussian Mixture Models)**, que asume que los datos han sido generados por múltiples distribuciones gaussianas superpuestas. Este enfoque permite modelar datos más complejos, donde las agrupaciones pueden tener diferentes formas y tamaños, proporcionando una mayor flexibilidad que los métodos basados en particiones.

Una ventaja clave de los modelos probabilísticos es su capacidad para manejar incertidumbre, ya que permiten obtener probabilidades de pertenencia que pueden ser útiles en la toma de decisiones o en aplicaciones donde se desea medir la confianza en la clasificación. Sin embargo, estos modelos suelen requerir un ajuste más complejo y pueden ser sensibles a la inicialización de los parámetros.

> [!note]
>
> Cada uno de estos enfoques de clustering ofrece una perspectiva única para abordar el problema de la agrupación de datos. Mientras que los métodos basados en particiones son eficientes y fáciles de interpretar, los algoritmos jerárquicos proporcionan una visión detallada de las relaciones entre los datos. Por otro lado, las técnicas basadas en densidad destacan en la detección de estructuras complejas, y los modelos probabilísticos ofrecen una forma más flexible de modelar la incertidumbre en los datos.
>
> La elección del método más adecuado dependerá de las características específicas del conjunto de datos y de los objetivos del análisis. Factores como la forma y distribución de los datos, la necesidad de interpretar los resultados y las restricciones computacionales son aspectos clave a considerar al seleccionar la estrategia de clustering más apropiada.

**Para reflexionar...**

> **¿Cuál de los enfoques de clustering consideras más adecuado para un conjunto de datos geoespaciales con estructuras complejas y ruido?**
>**Clave:** Reflexiona sobre la capacidad de los métodos basados en densidad para identificar agrupaciones de forma arbitraria y detectar valores atípicos de manera natural.

## **Clustering basado en particiones: Algoritmo K-Means**

El **clustering basado en particiones** tiene como objetivo dividir un conjunto de datos en **grupos disjuntos**, de modo que cada observación pertenezca a un único clúster. La idea principal de estos algoritmos es iterativamente asignar y reajustar las observaciones en función de su similitud con un centroide representativo del clúster.

Este enfoque busca encontrar una **partición óptima**, minimizando la variabilidad interna dentro de los clústeres y maximizando la separación entre ellos. Para lograr esto, se definen criterios de optimización basados en métricas de distancia, como la **Euclidiana**, aunque pueden utilizarse otras según el tipo de datos.

Los algoritmos de clustering basados en particiones siguen un enfoque iterativo que busca dividir un conjunto de datos en grupos exclusivos, optimizando un criterio de similitud para garantizar que las observaciones dentro de un mismo clúster sean lo más homogéneas posible, mientras que las pertenecientes a diferentes clústeres sean lo más disímiles entre sí.

Este proceso comienza con una etapa de **inicialización**, en la cual se seleccionan puntos representativos del conjunto de datos que actuarán como los centroides iniciales de cada clúster. Estos centroides son puntos de referencia alrededor de los cuales se agruparán las observaciones. La selección inicial de estos puntos puede realizarse de forma aleatoria, aunque existen técnicas avanzadas, que buscan una distribución más estratégica para mejorar la convergencia del algoritmo y evitar resultados subóptimos. La calidad de esta fase es crucial, ya que una mala elección de los centroides iniciales puede afectar el rendimiento y la estabilidad del proceso.

Una vez que los centroides iniciales han sido definidos, se procede a la etapa de **asignación**, donde cada punto del conjunto de datos se asigna al clúster cuyo centroide esté más cercano. La proximidad entre un punto y un centroide se determina mediante una métrica de distancia, que puede variar según la naturaleza de los datos. La **distancia euclidiana** es una de las más utilizadas, especialmente en conjuntos de datos con características numéricas, ya que mide la separación directa entre los puntos en el espacio multidimensional. Sin embargo, otras métricas, como la distancia de Manhattan o la distancia de Coseno, pueden ser más apropiadas dependiendo de la estructura de los datos y su interpretación.

Una vez que todos los puntos han sido asignados a sus respectivos clústeres, el algoritmo entra en la fase de **reajuste**, en la cual se recalculan los centroides de cada clúster. El nuevo centroide se obtiene calculando el promedio de todas las observaciones asignadas al clúster correspondiente. Este paso permite que los centroides se desplacen dentro del espacio de datos, acercándose progresivamente a las verdaderas regiones centrales de cada grupo. Este proceso de recalibración es fundamental para mejorar la cohesión interna de los clústeres y reducir la variabilidad dentro de ellos.

El proceso de asignación y reajuste no se realiza una sola vez, sino que se repite de manera iterativa en la etapa de **convergencia**, hasta que los centroides alcanzan una posición estable. La convergencia se define cuando los centroides dejan de cambiar significativamente entre iteraciones sucesivas o cuando se cumple un criterio de optimización específico, como la minimización de la suma de las distancias cuadradas dentro de cada clúster. En algunos casos, se puede establecer un número máximo de iteraciones para evitar un sobreprocesamiento innecesario, especialmente en conjuntos de datos grandes donde la convergencia puede llevar tiempo.

<img src=".\assets\617px-K-means_convergence.gif" alt="img" />

> **Ejemplo:**
>Imaginemos un conjunto de datos que contiene información sobre pacientes de un hospital, incluyendo variables como la edad, el índice de masa corporal (IMC) y la presión arterial. Si aplicamos un algoritmo de clustering basado en particiones para agrupar a los pacientes en diferentes categorías de riesgo, primero se seleccionarán centroides iniciales que representen grupos hipotéticos de pacientes con distintos niveles de salud. Luego, cada paciente se asignará al grupo más cercano según sus características de salud, y en cada iteración los centroides se recalcularán hasta estabilizarse, ofreciendo una segmentación útil para los médicos.

Este enfoque de clustering presenta ventajas significativas, como su rapidez y eficiencia en la segmentación de grandes volúmenes de datos. Sin embargo, también presenta desafíos, como la necesidad de definir de antemano el número de clústeres, lo cual no siempre es evidente en contextos donde la estructura de los datos no es conocida de antemano.

<img src="https://miro.medium.com/v2/resize:fit:709/1*JsfEdbXKwJw_Euprvx17KA.png" alt="Fully Explained K-means" />

Matemáticamente, el objetivo de K-Means es minimizar la siguiente función de costo:

$$
J = \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
$$

Donde:
- $x_j$ representa un punto de datos,
- $\mu_i$ es el centroide del clúster $C_i$,
- $k$ es el número total de clústeres,
- La métrica $\| x_j - \mu_i \|^2$ representa la distancia Euclidiana entre cada punto y su centroide asignado.

### **Elección del número de clústeres en K-Means**

Uno de los aspectos más desafiantes en el clustering basado en particiones es la **elección del número adecuado de clústeres**, representado por el parámetro $k$ en algoritmos como K-Means. Determinar el valor óptimo de $k$ es crucial para obtener una segmentación significativa de los datos, ya que un número inadecuado de clústeres puede llevar a agrupaciones demasiado generales o excesivamente específicas, afectando la interpretabilidad y utilidad del modelo.

Si se elige un número de clústeres demasiado bajo, se corre el riesgo de agrupar puntos de datos muy disímiles dentro de la misma categoría, perdiendo información valiosa sobre patrones específicos en los datos. Por otro lado, un valor de $k$ demasiado alto puede generar clústeres artificialmente pequeños y fragmentados, lo que resulta en una segmentación excesivamente detallada y difícil de interpretar.

Para abordar este problema, se han desarrollado varias estrategias que permiten estimar el número óptimo de clústeres de manera sistemática. Entre las técnicas más utilizadas se encuentran el **método del codo (Elbow Method)** y el **coeficiente de silueta**, las cuales proporcionan información clave para tomar decisiones informadas sobre la cantidad de clústeres a utilizar.

#### Método del codo (Elbow Method)

El **método del codo** es una técnica visual ampliamente utilizada para determinar el número óptimo de clústeres en un conjunto de datos. Su principio se basa en analizar la **suma de los errores cuadrados internos** (WCSS, por sus siglas en inglés *Within-Cluster Sum of Squares*), que mide la compactación de los puntos dentro de cada clúster.

El WCSS se calcula como la suma de las distancias cuadradas entre cada punto de un clúster y su centroide:

$$
WCSS = \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
$$

Donde:
- $x_j$ representa cada punto de datos,
- $\mu_i$ es el centroide del clúster $C_i$,
- $k$ es el número total de clústeres.

El procedimiento para aplicar el método del codo consiste en entrenar el modelo de clustering con diferentes valores de $k$ y graficar el valor de WCSS en función de la cantidad de clústeres. Inicialmente, la reducción de WCSS es rápida, ya que agregar más clústeres ayuda a minimizar la distancia entre los puntos y sus centroides. Sin embargo, a partir de cierto punto, el beneficio marginal de agregar más clústeres disminuye y la curva comienza a aplanarse, formando una forma de "codo".

El punto donde se produce este cambio de pendiente se considera el valor óptimo de $k$, ya que representa un equilibrio entre la compactación de los clústeres y la simplicidad del modelo.

> **Ejemplo:**
>
> Imaginemos que una empresa quiere segmentar a sus clientes en función de sus patrones de compra. Al aplicar K-Means con valores de $k$ entre 1 y 10 y graficar el WCSS, observamos que la curva se aplana a partir de $k = 4$. Esto sugiere que cuatro clústeres representan un equilibrio razonable entre precisión y simplicidad en la segmentación de los clientes.
>

> [!warning]
>
> El método del codo es simple y efectivo, pero su interpretación puede ser subjetiva, ya que la "curvatura" no siempre es fácilmente identificable. Es recomendable complementar este método con otras métricas para garantizar una elección más robusta del número de clústeres.

#### Coeficiente de silueta

Por otro lado, el **coeficiente de silueta** es una métrica que evalúa la **calidad del clustering** considerando simultáneamente la **cohesión interna** (qué tan bien agrupados están los puntos dentro de un clúster) y la **separación externa** (qué tan alejados están los clústeres entre sí).

Para cada punto de datos $x_i$, el coeficiente de silueta se define como:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Donde:
- $a(i)$ es la distancia promedio de $x_i$ a los demás puntos dentro de su propio clúster.
- $b(i)$ es la distancia promedio de $x_i$ a los puntos del clúster más cercano (el clúster vecino).

El coeficiente de silueta toma valores entre -1 y 1:

- Valores cercanos a 1 indican que los puntos están bien agrupados dentro de su clúster y alejados de los demás clústeres.
- Valores cercanos a 0 sugieren que los puntos están cerca de la frontera entre clústeres, lo que indica posibles solapamientos.
- Valores negativos indican una asignación incorrecta del punto al clúster, sugiriendo que debería pertenecer a otro clúster.

El coeficiente de silueta promedio para todo el conjunto de datos se puede utilizar como un criterio para seleccionar el valor óptimo de $k$. Un valor más alto sugiere una mejor calidad de clustering.

> **Ejemplo:**
>
> Supongamos que en un análisis de segmentación de clientes probamos diferentes valores de $k$. Para $k=3$ obtenemos un coeficiente de silueta promedio de 0.62, mientras que para $k=4$ el coeficiente baja a 0.45. Esto sugiere que la solución con $k=3$ proporciona clústeres más compactos y mejor separados.
>

> [!warning]
>
> El coeficiente de silueta ofrece una forma cuantitativa de evaluar el clustering, eliminando parte de la subjetividad presente en el método del codo, sin embargo, su cálculo es más costoso computacionalmente, ya que requiere medir distancias entre todos los puntos.

> [!tip]
>
> La elección del número óptimo de clústeres es un paso crucial en el proceso de clustering, ya que impacta directamente en la calidad y la interpretabilidad de los resultados. Métodos como el codo y el coeficiente de silueta proporcionan herramientas útiles para guiar esta decisión, permitiendo encontrar un balance entre la compacidad de los clústeres y su separación. Para garantizar una segmentación eficaz, es importante considerar también el contexto del problema, la naturaleza de los datos y realizar pruebas adicionales para validar la solidez de la elección de $k$.

### **Ventajas y limitaciones del clustering basado en particiones**

El clustering basado en particiones, y en particular el algoritmo **K-Means**, ha logrado una gran aceptación en la comunidad de análisis de datos debido a su **eficiencia, simplicidad e interpretabilidad**. Su capacidad para agrupar grandes volúmenes de datos de manera rápida lo convierte en una opción atractiva para aplicaciones en diversas industrias, como el marketing, la salud, la biología y la segmentación de clientes. Sin embargo, a pesar de sus ventajas, este método no está exento de limitaciones que pueden comprometer la calidad de los resultados si no se toman las precauciones adecuadas.

Una de las **principales ventajas** de K-Means es su **rapidez y escalabilidad**, ya que su tiempo de ejecución crece linealmente con el número de muestras y el número de clústeres, lo que lo hace adecuado para conjuntos de datos extensos. Además, su funcionamiento se basa en conceptos intuitivos de proximidad y promedios, lo que facilita su comprensión e implementación incluso para personas con un conocimiento técnico moderado. Gracias a su estructura bien definida, K-Means tiende a generar clústeres de forma **esférica y homogénea**, lo que permite capturar de manera efectiva patrones en datos bien estructurados y con una distribución clara.

Otra ventaja importante radica en su **interpretabilidad**. Los centroides generados representan el punto medio de cada grupo, lo que permite obtener información útil sobre las características promedio de cada clúster. Esta propiedad es especialmente valiosa en entornos empresariales, donde se requiere que los modelos sean comprensibles para la toma de decisiones.

No obstante, K-Means también presenta **varias limitaciones** que deben considerarse antes de su aplicación. Una de las más significativas es su **sensibilidad a la inicialización de los centroides**, ya que una mala elección inicial puede llevar a soluciones subóptimas o incluso a resultados inconsistentes en diferentes ejecuciones del algoritmo. Para mitigar este problema, se recomienda el uso de técnicas como **K-Means++**, que busca mejorar la selección inicial mediante un proceso de dispersión óptima de los centroides antes de comenzar la iteración del algoritmo.

Otra limitación importante es que K-Means **supone que los clústeres tienen forma esférica**, lo que significa que no funciona bien con datos que presentan distribuciones complejas o no convexas. Por ejemplo, en conjuntos de datos donde los grupos tienen formas irregulares, como agrupaciones en forma de anillo o estructuras concéntricas, K-Means puede fallar en la identificación correcta de los patrones. Algoritmos más flexibles, como **DBSCAN** o **clustering jerárquico**, pueden ser más adecuados en estos escenarios.

Además, K-Means es altamente **sensible a la escala de los datos**, lo que significa que variables con diferentes rangos de valores pueden dominar el proceso de clustering, distorsionando los resultados. Por esta razón, es fundamental aplicar técnicas de **normalización o estandarización**, como la transformación min-max o la estandarización z-score, para garantizar que todas las variables contribuyan de manera equitativa al proceso de agrupación.

Otro aspecto a considerar es su **incapacidad para manejar automáticamente valores atípicos**, ya que estos pueden influir negativamente en la ubicación de los centroides, generando clústeres distorsionados. En la práctica, se recomienda realizar una limpieza previa de los datos para mitigar el impacto de estos valores extremos.

Por último, es importante recordar que K-Means requiere que el número de clústeres $k$ sea definido de antemano, lo que puede representar un desafío cuando no se tiene un conocimiento previo del conjunto de datos. La elección inadecuada de $k$ puede llevar a agrupaciones poco significativas o fragmentadas. Técnicas como el **método del codo** o el **coeficiente de silueta** pueden ser utilizadas para estimar un valor adecuado de $k$.

### **Aplicaciones prácticas del clustering basado en particiones**

El clustering basado en particiones, en particular el algoritmo **K-Means**, ha encontrado aplicaciones en una amplia variedad de dominios, donde la capacidad de descubrir patrones ocultos y agrupar elementos similares resulta clave para la toma de decisiones estratégicas. Su versatilidad y eficiencia permiten abordar problemas que van desde el análisis de clientes hasta la biología molecular, proporcionando soluciones prácticas en diferentes industrias.

Una de las aplicaciones más comunes de K-Means es la **segmentación de clientes**, un pilar fundamental en marketing y ventas. Las empresas utilizan clustering para identificar grupos de clientes con comportamientos de compra similares, permitiéndoles personalizar campañas publicitarias, optimizar recomendaciones de productos y mejorar la retención. Por ejemplo, mediante el análisis de datos de compras, se pueden distinguir segmentos como **clientes frecuentes**, **compradores ocasionales** o **clientes potenciales**, lo que permite ajustar la estrategia de negocio a cada perfil específico.

En el ámbito de la **visión por computadora**, K-Means se emplea para la **agrupación de imágenes**, clasificando píxeles en función de sus características de color e intensidad. Un caso de uso típico es la **segmentación de imágenes médicas**, donde se pueden identificar regiones de interés, como tejidos o tumores, facilitando el diagnóstico asistido por computadora. La simplicidad del algoritmo lo hace ideal para procesar grandes volúmenes de datos visuales, contribuyendo a una rápida interpretación por parte de los especialistas.

Otro campo donde el clustering ha demostrado su utilidad es el **análisis de redes sociales**, donde la agrupación de usuarios en función de sus interacciones, intereses o patrones de comportamiento permite detectar **comunidades y tendencias emergentes**. Plataformas como Facebook o Twitter pueden aplicar K-Means para identificar grupos de usuarios con intereses comunes, lo que permite la creación de contenido personalizado y el análisis de opiniones sobre productos o eventos específicos.

En el sector biomédico, el clustering basado en particiones se ha utilizado para la **agrupación de datos genéticos**, donde se busca encontrar similitudes entre genes o secuencias de ADN. Por ejemplo, en estudios de expresión génica, K-Means ayuda a clasificar genes con patrones de activación similares, lo que puede proporcionar información clave para la identificación de enfermedades o el desarrollo de tratamientos personalizados. Gracias a esta técnica, los investigadores pueden reducir la complejidad de enormes volúmenes de datos genómicos y enfocarse en las agrupaciones más relevantes.

Además de estas aplicaciones principales, K-Means ha sido implementado con éxito en áreas como:

- **Segmentación geográfica**, para analizar el comportamiento de las poblaciones en diferentes regiones y optimizar recursos urbanos como el transporte público.
- **Análisis de fraude financiero**, agrupando transacciones sospechosas con patrones anómalos para detectar posibles actividades fraudulentas.
- **Optimización de inventarios**, ayudando a agrupar productos con demanda similar para mejorar la gestión del almacenamiento y la logística.

## Clustering jerárquico

### Introducción

En nuestra vida cotidiana, estamos constantemente agrupando cosas sin darnos cuenta. Por ejemplo, al organizar libros en una biblioteca, podríamos agruparlos por género, luego por autor y finalmente por año de publicación. Este proceso natural de clasificación nos ayuda a entender mejor la relación entre los elementos y facilita su búsqueda. De manera similar, en el mundo del análisis de datos, el **clustering jerárquico** es una técnica que permite organizar elementos en **niveles de agrupación**. Para ello hace uso estructuras en forma de árbol donde cada nivel representa una posible segmentación de los datos.

El clustering jerárquico es una herramienta poderosa dentro del **aprendizaje no supervisado**, ya que, a diferencia de otros métodos de clustering, como los basados en particiones, no requiere la especificación del número de grupos de antemano. Así, el clustering jerárquico proporciona una visión más flexible y profunda de la estructura subyacente.

Uno de los aspectos más distintivos del clustering jerárquico es su representación gráfica mediante un **dendrograma**, una especie de “árbol genealógico” de los datos. En un dendrograma, cada elemento comienza como un grupo separado y, a medida que avanzamos en la estructura, las observaciones se van fusionando en grupos más grandes hasta formar un solo conjunto. Este gráfico es especialmente útil para identificar relaciones entre los datos y decidir cómo dividirlos en grupos significativos.

En el fondo, el clustering jerárquico se basa en la idea de que los datos pueden agruparse según su **similitud**, utilizando métricas como la **distancia euclidiana**, que mide la proximidad entre puntos en un espacio multidimensional. Dependiendo de la naturaleza de los datos y del objetivo del análisis, se pueden aplicar diferentes estrategias para construir la jerarquía.

Este enfoque es especialmente útil cuando se busca analizar estructuras jerárquicas naturales, como taxonomías biológicas o segmentaciones de clientes basadas en múltiples niveles de similitud. Además, al ser un método **determinista**, garantiza que los resultados sean consistentes en diferentes ejecuciones.

Para construir una jerarquía de clústeres, el algoritmo de clustering jerárquico sigue un proceso que comienza con la medición de similitudes entre las observaciones. Posteriormente, se agrupan o dividen iterativamente los elementos hasta formar una estructura que revela patrones de agrupamiento en diferentes escalas.

El procedimiento general del clustering jerárquico involucra cuatro etapas clave: Primero, se calcula una **matriz de distancias**, que mide la similitud entre los puntos utilizando métricas como la distancia euclidiana, Manhattan o medidas de correlación. Luego, se lleva a cabo la **fusión o división de clústeres**, dependiendo de si se utiliza un enfoque aglomerativo o divisivo. Una vez formados los clústeres, se construye el dendrograma, una representación visual que muestra cómo se organizan los datos en diferentes niveles jerárquicos. Finalmente, se selecciona el número óptimo de clústeres mediante la identificación de un punto de corte en el dendrograma.

### Métodos de clustering jerárquico

Existen dos enfoques fundamentales para la construcción de clústeres jerárquicos: el **método aglomerativo** y el **método divisivo**.

El método aglomerativo es el más utilizado debido a su simplicidad y claridad visual. En este enfoque, se comienza con cada observación como un clúster independiente y, en cada iteración, los clústeres más cercanos se combinan hasta formar una única agrupación global. Este procedimiento bottom-up es útil cuando se desea identificar estructuras de agrupamiento compactas y bien definidas.

Por otro lado, el método divisivo adopta un enfoque top-down, en el que se parte de un único clúster que agrupa a todas las observaciones. A partir de ahí, el clúster se divide sucesivamente en subgrupos hasta que cada elemento queda asignado a su propio clúster. Aunque este enfoque puede proporcionar una visión más general de la estructura de los datos, **su implementación suele ser más costosa computacionalmente.**

El resto de la sección se centrará en el **método aglomerativo.**

### Criterios de enlace en clustering jerárquico

Como se ha comentado anteriormente, el primer paso en cualquier proceso de clustering jerárquico es calcular la matriz de distancias de las observaciones. Una vez que las distancias entre observaciones están definidas, el siguiente paso es fusionar los clústeres de acuerdo con lo que se denomina un **criterio de enlace (linkage)**. Este criterio determina cómo se mide la distancia entre grupos de observaciones a medida que se van uniendo nuevos puntos.

Existen varios métodos de enlace disponibles. Uno de los métodos más sencillos de aplicar es el **enlace simple**, que genera los clústeres en función de la menor distancia entre sus elementos. Si bien este método permite detectar patrones encadenados, tiende a generar agrupaciones de forma irregular.

Por su parte, el **enlace completo**, en contraste, mide la distancia máxima entre los puntos de los clústeres a generar, lo que genera agrupaciones más compactas y de tamaño uniforme. Esta técnica es útil cuando se desea evitar la formación de clústeres alargados.

Otra alternativa es el **enlace promedio**, que calcula la distancia media entre todos los puntos de los clústeres, proporcionando un equilibrio entre la compacidad y la cohesión de los grupos resultantes.

Finalmente, el **enlace de Ward** es uno de los más utilizados en la práctica, ya que minimiza la varianza intra-clúster, generando agrupaciones homogéneas y bien separadas. Este criterio es particularmente útil en situaciones donde se desea obtener clústeres de tamaño similar y estructuras compactas.

<img src=".\assets\15-Hierarchical-Clustering-Linkages.png" alt="How the Hierarchical Clustering Algorithm Works" />

### Interpretación del dendrograma

Uno de los aspectos más valiosos del clustering jerárquico es la posibilidad de visualizar los resultados mediante un **dendrograma**, que representa gráficamente el proceso de agrupamiento de los datos. Cada nodo del dendrograma indica una fusión entre clústeres, y la altura a la que ocurre refleja la distancia o disimilitud entre ellos.

Interpretar un dendrograma implica identificar un punto de corte adecuado para seleccionar el número óptimo de clústeres. Este punto de corte se determina visualmente trazando una línea horizontal en el nivel donde las uniones entre clústeres son más grandes, lo que sugiere que los grupos son significativamente distintos entre sí.

<img src=".\assets\1VvOVxdBb74IOxxF2RmthCQ.png" alt="Hierarchical clustering explained | by Prasad Pai | Towards Data Science" />

### Ventajas y limitaciones del clustering jerárquico

El clustering jerárquico presenta varias ventajas que lo hacen especialmente atractivo en el análisis exploratorio de datos. Su capacidad para representar la estructura de los datos en múltiples niveles permite explorar diferentes granularidades sin necesidad de realizar múltiples ejecuciones con diferentes parámetros. Además, su carácter determinista asegura la reproducibilidad de los resultados, lo que facilita la interpretación y validación de los grupos obtenidos.

Sin embargo, este enfoque también tiene limitaciones importantes. Su principal desventaja radica en su **complejidad computacional**, que crece cuadráticamente con el número de observaciones, lo que puede dificultar su aplicación en conjuntos de datos grandes. Además, la técnica es **sensible al ruido y a los valores atípicos**, ya que estos pueden distorsionar la estructura del dendrograma, generando divisiones incorrectas.

Otra limitación es su **falta de flexibilidad** para detectar clústeres de formas complejas o de densidades variables, lo que puede ser un inconveniente en escenarios donde la distribución de los datos no es uniforme.

### Aplicaciones prácticas del clustering jerárquico

El clustering jerárquico se ha implementado con éxito en una amplia variedad de dominios donde la interpretación jerárquica de los datos es relevante. En **biología y genética**, por ejemplo, se utiliza para clasificar organismos en taxonomías evolutivas o agrupar genes con funciones similares.

En el ámbito del **marketing**, esta técnica se emplea para segmentar clientes en función de sus patrones de comportamiento, permitiendo la creación de estrategias de personalización basadas en niveles de segmentación cada vez más específicos.

Otra aplicación importante se encuentra en el **procesamiento de textos**, donde el clustering jerárquico se utiliza para organizar documentos o términos según su similitud semántica, facilitando la gestión de grandes volúmenes de información textual.

> **Ejemplo:**
>En un análisis de mercado, una empresa desea segmentar a sus clientes en función de su historial de compras. Utilizando clustering jerárquico, se obtiene un dendrograma que muestra cómo los clientes se agrupan en función de su gasto promedio y la frecuencia de compra. Al aplicar un punto de corte adecuado, se identifican tres segmentos de clientes claramente diferenciados: compradores frecuentes, compradores esporádicos y clientes potenciales.

**Para reflexionar...**

> **¿Por qué la elección del criterio de enlace es crucial en el clustering jerárquico?**
>**Clave:** Considera cómo diferentes criterios de enlace afectan la forma y cohesión de los clústeres, y piensa en escenarios donde un criterio podría ser más adecuado que otro.

## Clustering basado en densidad: DBSCAN

### Introducción

El clustering basado en densidad es un enfoque poderoso dentro del aprendizaje no supervisado que se fundamenta en la agrupación de puntos de datos en función de su concentración en el espacio. A diferencia de otros métodos, como K-Means o el clustering jerárquico, que dependen de supuestos sobre la forma o el número de clústeres, el algoritmo **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** identifica agrupaciones en regiones densamente pobladas, permitiendo la detección automática de estructuras de formas arbitrarias y la identificación de valores atípicos.

Este enfoque es especialmente útil en escenarios donde los datos contienen regiones de diferente densidad o están afectados por ruido. En lugar de asignar cada punto a un clúster específico, DBSCAN diferencia entre puntos **centrales**, **de borde** y **ruidosos**, proporcionando una mayor flexibilidad en comparación con métodos más tradicionales. 

![DBSCAN Clustering in Machine Learning](./assets/db6-e1584577503359.png)

### Funcionamiento del algoritmo DBSCAN

El algoritmo DBSCAN se basa en dos conceptos fundamentales: la **densidad** de los puntos de datos y la **conectividad** entre ellos. Su funcionamiento se estructura en torno a los siguientes principios:

#### Definición de la vecindad

El primer paso en el funcionamiento de DBSCAN consiste en analizar la vecindad de cada punto en el conjunto de datos. Para ello, se define un radio denominado $\varepsilon$, el cual establece el alcance dentro del cual se considerarán otros puntos como vecinos. Una vez determinado este radio, se procede a contar cuántos puntos se encuentran dentro de esa distancia. Si el número de puntos vecinos dentro de la región definida por $\varepsilon$ alcanza o supera un umbral mínimo, conocido como **mínimo de puntos** o $MinPts$, se considera que el punto analizado pertenece a una región densa. Esta idea de densidad es fundamental para el algoritmo, ya que permite detectar áreas donde los puntos están más concentrados, formando posibles agrupaciones naturales dentro de los datos.

#### Clasificación de puntos

Después de determinar la vecindad de cada punto, el algoritmo clasifica las observaciones en función de su relación con otros puntos dentro del radio $\varepsilon$. En este proceso, surgen tres categorías fundamentales de puntos que definirán la estructura de los clústeres.

En primer lugar, se identifican los **puntos centrales**, que son aquellos que cumplen con la condición de tener al menos $MinPts$ puntos dentro de su vecindad. Estos puntos son considerados el núcleo de un clúster y representan regiones de alta densidad en los datos.

Por otro lado, se encuentran los **puntos de borde**, que son aquellos que se encuentran dentro de la vecindad de un punto central pero que, por sí mismos, no cumplen con el umbral mínimo de densidad. A pesar de no ser puntos centrales, los puntos de borde contribuyen a la expansión del clúster y ayudan a definir sus límites.

Finalmente, se identifican los **puntos ruido**, que no cumplen con la condición de estar suficientemente cerca de otros puntos como para formar parte de un clúster. Estos puntos, al no estar conectados a ningún punto central, se consideran valores atípicos o anomalías dentro del conjunto de datos.

![Visualization of DBSCAN clustering algorithm. | Download Scientific Diagram](./assets/images.png)

#### Expansión de clústeres

Una vez que se ha identificado un punto central, el algoritmo inicia el proceso de expansión del clúster, explorando los puntos de borde adyacentes y aquellos que se encuentran dentro de su vecindad inmediata. Si alguno de estos puntos de borde cumple con la condición de convertirse en un nuevo punto central, se incorpora al clúster y su vecindad se explora de la misma manera, generando una expansión progresiva del grupo de puntos conectados.

Este proceso de expansión continúa iterativamente, recorriendo la estructura de los datos hasta que no se detecten más puntos que puedan ser incorporados al clúster. Como resultado, DBSCAN forma agrupaciones de puntos densamente conectados, sin necesidad de establecer de antemano el número de clústeres, como ocurre en otros algoritmos de agrupación.

#### Identificación de ruido

Durante la ejecución del algoritmo, algunos puntos no logran ser incorporados en ninguna agrupación densa. Estos puntos aislados no tienen suficientes vecinos en su entorno inmediato y tampoco están conectados a otros puntos centrales. DBSCAN los clasifica como **ruido**, etiquetándolos como valores atípicos.

La capacidad de DBSCAN para identificar estos puntos ruidosos es una de sus principales fortalezas, ya que permite distinguir entre patrones significativos y observaciones que no siguen ninguna estructura clara. En aplicaciones como la detección de anomalías o el análisis de datos geoespaciales, la capacidad de DBSCAN para detectar ruido es fundamental para obtener una visión más precisa de la estructura de los datos.

### Parámetros clave de DBSCAN

El rendimiento de DBSCAN está estrechamente ligado a la elección adecuada de sus hiperparámetros principales, los cuales definen la forma en que el algoritmo identifica regiones densas y, en consecuencia, los clústeres dentro de los datos. La correcta configuración de estos parámetros es esencial para obtener resultados significativos y evitar problemas como la sobre-segmentación o la agrupación inadecuada de puntos.

Uno de los hiperparámetros más importantes es $\varepsilon$, también conocido como **epsilon**, el cual determina el radio máximo dentro del cual un punto es considerado vecino de otro. Este valor define la proximidad entre puntos y, por lo tanto, influye directamente en la formación de clústeres. Si $\varepsilon$ se establece con un valor demasiado pequeño, el algoritmo puede fragmentar clústeres densos en múltiples subgrupos, perdiendo la estructura global de los datos. Por otro lado, si $\varepsilon$ es demasiado grande, los clústeres pueden fusionarse de manera artificial, agrupando puntos que en realidad no deberían estar conectados.

La elección adecuada de $\varepsilon$ depende de la escala y distribución de los datos, por lo que se recomienda un análisis previo mediante herramientas como la curva *k-distance*, que ayuda a identificar un valor óptimo al observar el punto donde la curva presenta un "codo", indicando un cambio en la densidad de los datos.

Otro hiperparámetro crucial es $MinPts$, que define el **número mínimo de puntos** necesarios dentro de la distancia $\varepsilon$ para que una región sea considerada densa. Este valor determina si un punto es clasificado como central y, en consecuencia, si puede iniciar la formación de un clúster.

La configuración de $MinPts$ suele depender de la dimensionalidad del conjunto de datos, siguiendo una regla empírica común que sugiere establecerlo como el doble de la dimensión del espacio de características. Por ejemplo, para un conjunto de datos con 5 características, una elección inicial razonable sería $MinPts=10$. Sin embargo, este valor puede ajustarse dependiendo de la cantidad de ruido presente en los datos y del nivel de granularidad deseado en la detección de clústeres.

> **Ejemplo:**
>Supongamos que estamos analizando datos de sensores ambientales en una ciudad, donde cada punto representa una estación de monitoreo con variables como temperatura, humedad y calidad del aire. Si configuramos un valor de $\varepsilon$ demasiado bajo, los sensores que registran patrones similares podrían no ser agrupados, dificultando la identificación de zonas con condiciones atmosféricas homogéneas. Por el contrario, un valor de $MinPts$ demasiado alto podría omitir estaciones aisladas que detectan eventos climáticos relevantes.

La combinación adecuada de $\varepsilon$ y $MinPts$ es fundamental para garantizar que DBSCAN detecte clústeres significativos, evitando tanto la sobresegmentación como la inclusión de puntos no relevantes en los grupos.

### Ventajas y limitaciones de DBSCAN

DBSCAN es un algoritmo de clustering altamente versátil que ofrece múltiples ventajas, haciéndolo especialmente útil en situaciones donde los datos presentan estructuras complejas o cuando se requiere la detección automática de anomalías. Su capacidad para adaptarse a la forma natural de los datos sin imponer suposiciones rígidas lo convierte en una opción atractiva para una amplia variedad de aplicaciones.

Una de sus principales fortalezas radica en que **no requiere especificar de antemano el número de clústeres**, a diferencia de otros algoritmos como K-Means, que dependen de una elección manual del número de grupos. DBSCAN, en cambio, identifica automáticamente las regiones densas, generando clústeres en función de la distribución de los puntos en el espacio. Esto permite que el algoritmo se adapte mejor a los patrones inherentes de los datos sin necesidad de intervención manual.

Otra ventaja importante es su capacidad para **detectar clústeres de formas arbitrarias**, lo que lo diferencia de algoritmos que solo identifican grupos de estructura esférica. Esto significa que DBSCAN puede agrupar datos en configuraciones complejas, como agrupaciones alargadas o no convexas, lo que resulta especialmente útil en aplicaciones geoespaciales o en problemas donde los datos presentan distribuciones irregulares.

Además, DBSCAN incorpora de manera natural un mecanismo de **detección de valores atípicos**, etiquetando como ruido aquellos puntos que no pertenecen a ninguna región densa. Esto lo hace ideal para tareas de detección de anomalías en datos financieros, industriales o de ciberseguridad, donde es importante identificar puntos de datos que no se ajustan a los patrones generales.

Otro aspecto a destacar es su **robustez frente a datos ruidosos**, ya que las áreas de baja densidad no afectan significativamente la formación de clústeres. Esto lo hace particularmente útil en entornos donde los datos pueden contener ruido o estar distribuidos de manera desigual, como en estudios de población o análisis de tráfico urbano.

A pesar de estas ventajas, DBSCAN presenta **algunas limitaciones** que deben ser tenidas en cuenta al momento de aplicarlo.

Uno de los principales desafíos es su **desempeño en conjuntos de datos de alta dimensionalidad**. A medida que la cantidad de dimensiones aumenta, la noción de densidad pierde significado debido al fenómeno conocido como la "maldición de la dimensionalidad". En estas situaciones, las distancias entre los puntos tienden a volverse más uniformes, lo que dificulta la identificación de regiones densas y hace que la elección de los parámetros sea más compleja.

La **selección de los hiperparámetros $\varepsilon$ y $MinPts$** es otro aspecto crítico en la aplicación de DBSCAN. Una mala elección de estos valores puede generar clústeres que no reflejan la estructura real de los datos, lo que lleva a una segmentación deficiente. Por esta razón, suele ser necesario realizar pruebas iterativas o utilizar herramientas como la curva *k-distance* para determinar valores óptimos, lo que puede requerir un esfuerzo adicional de ajuste.

Además, DBSCAN **no se adapta bien a conjuntos de datos con densidades muy variables**, ya que un único valor de $\varepsilon$ puede no ser suficiente para capturar agrupaciones de diferentes escalas. En casos donde los datos presentan regiones con densidades significativamente distintas, el algoritmo puede agrupar incorrectamente puntos pertenecientes a diferentes clústeres o ignorar algunos clústeres más dispersos.

> **Ejemplo:**
>Supongamos que se está utilizando DBSCAN para analizar datos de transacciones bancarias. Si los parámetros $\varepsilon$ y $MinPts$ no se eligen correctamente, el algoritmo podría etiquetar transacciones legítimas como fraude debido a una mala configuración de los límites de densidad, afectando la precisión del sistema de detección de anomalías.

> [!warning]
>
> En resumen, DBSCAN es una herramienta poderosa para la agrupación de datos en entornos complejos y con ruido, pero su aplicación efectiva requiere un conocimiento detallado de los datos y un ajuste cuidadoso de los parámetros para obtener los mejores resultados.

### Aplicaciones prácticas de DBSCAN

DBSCAN se ha convertido en una herramienta clave en diversos ámbitos donde la estructura de los datos es compleja y no se ajusta fácilmente a formas geométricas predefinidas. Su capacidad para detectar patrones en datos densamente agrupados, junto con su habilidad para identificar valores atípicos, lo hace especialmente útil en escenarios donde la flexibilidad y la adaptabilidad son fundamentales.

Uno de los campos donde DBSCAN ha demostrado su eficacia es el **análisis de redes sociales**, donde se utiliza para identificar grupos de usuarios con comportamientos similares en función de su actividad, interacciones y patrones de conexión. Por ejemplo, en plataformas como redes de mensajería o redes profesionales, DBSCAN permite detectar comunidades de usuarios que interactúan con mayor frecuencia, sin necesidad de especificar de antemano cuántos grupos existen. Esto facilita el análisis de tendencias y la personalización de contenido.

En el ámbito financiero, DBSCAN es una herramienta eficaz para la **detección de fraudes**, ya que permite identificar transacciones sospechosas que se desvían de los patrones normales de comportamiento. A diferencia de otros métodos de clustering que podrían pasar por alto actividades anómalas debido a su dependencia de la forma de los clústeres, DBSCAN tiene la capacidad de señalar puntos aislados que representan posibles fraudes, como transacciones inusuales en horarios atípicos o con montos elevados fuera de lo común.

Otro campo donde DBSCAN es ampliamente utilizado es el **procesamiento de imágenes**, particularmente en la segmentación de regiones densas en imágenes médicas o satelitales. En el análisis de imágenes médicas, por ejemplo, DBSCAN puede ser aplicado para identificar áreas de alta densidad de células tumorales en escaneos de tejidos, diferenciándolas de las áreas sanas. De manera similar, en imágenes satelitales, permite detectar regiones urbanas densamente pobladas o áreas de deforestación, proporcionando información crucial para la planificación urbana o la gestión medioambiental.

El **análisis geoespacial** es otra de las áreas donde DBSCAN resulta particularmente útil. Al trabajar con datos de ubicación geográfica, el algoritmo permite identificar patrones de concentración de recursos o actividad humana. Por ejemplo, en el sector del transporte y la logística, DBSCAN se emplea para detectar zonas de alta demanda de transporte en una ciudad, ayudando a optimizar rutas de autobuses o servicios de transporte compartido. De igual manera, en el ámbito de la seguridad pública, puede ser utilizado para identificar zonas con alta concentración de incidentes delictivos y mejorar la distribución de recursos policiales.

> **Ejemplo:**
>Imagina un proyecto de movilidad urbana que analiza datos de geolocalización de teléfonos móviles en una gran ciudad. Utilizando DBSCAN, es posible detectar zonas de alto tráfico en horarios pico, agrupando puntos donde la densidad de desplazamientos es elevada y clasificando áreas con menor actividad como ruido. Esta información permite a los planificadores urbanos optimizar la infraestructura vial y los servicios de transporte público.

Gracias a su capacidad para adaptarse a una variedad de estructuras de datos y su flexibilidad en la detección de patrones complejos, DBSCAN se ha convertido en una herramienta valiosa en múltiples disciplinas. Sin embargo, su aplicación efectiva requiere una comprensión adecuada de los datos y una selección cuidadosa de los parámetros para garantizar resultados precisos y útiles.

### Evaluación de la calidad del clustering con DBSCAN

Dado que DBSCAN no genera una partición rígida de los datos, su evaluación requiere un enfoque diferente al de otros algoritmos de clustering más tradicionales como K-Means. En métodos como K-Means, donde cada punto se asigna a un único clúster, es común utilizar métricas como la precisión o la exactitud cuando se dispone de etiquetas de referencia. Sin embargo, en DBSCAN, donde algunos puntos se etiquetan como ruido y los clústeres pueden tener formas irregulares, se necesitan métricas especializadas que reflejen la calidad de las agrupaciones detectadas.

Para evaluar adecuadamente el rendimiento de DBSCAN, es importante considerar tanto la calidad de los clústeres generados como la cantidad de puntos clasificados como ruido. Existen diversas métricas que permiten analizar la efectividad del algoritmo en la identificación de agrupaciones significativas dentro de los datos.

Una de las métricas más utilizadas es el **coeficiente de silueta**. Ya hemos visto que este coeficiente mide qué tan bien agrupados están los puntos dentro de un clúster en comparación con los puntos de otros clústeres. Esta métrica calcula, para cada punto, la diferencia entre la distancia promedio a los puntos de su propio clúster y la distancia promedio a los puntos del clúster más cercano. El coeficiente de silueta toma valores entre -1 y 1, donde valores cercanos a 1 indican que los clústeres están bien separados y son compactos, mientras que valores negativos sugieren que los puntos han sido asignados incorrectamente.

Otra métrica relevante es el **índice de Davies-Bouldin**, que mide la compacidad y separación de los clústeres generados. Este índice se basa en la relación entre la dispersión dentro de los clústeres y la distancia entre ellos. Un valor más bajo del índice de Davies-Bouldin indica clústeres más compactos y mejor separados, lo que sugiere una segmentación más precisa de los datos.

> **Ejemplo:**
>Supongamos que se aplica DBSCAN para segmentar clientes de un supermercado en función de su comportamiento de compra. Si el coeficiente de silueta es alto y el índice de Davies-Bouldin es bajo, significa que los clientes con hábitos de compra similares han sido agrupados de manera efectiva, mientras que aquellos con patrones diferentes han sido separados adecuadamente.

Además de estas métricas tradicionales, una consideración importante en DBSCAN es el **número de puntos ruidosos** detectados por el algoritmo. Una alta cantidad de puntos clasificados como ruido podría indicar que los parámetros $\varepsilon$ y $MinPts$ no se han ajustado correctamente. Si se elige un valor de $\varepsilon$ demasiado pequeño, es posible que muchos puntos no sean considerados parte de ningún clúster y se clasifiquen erróneamente como ruido. Por el contrario, un valor de $\varepsilon$ demasiado grande podría hacer que los clústeres se expandan en exceso, reduciendo la capacidad del algoritmo para detectar patrones significativos en los datos.

En algunos casos, es útil analizar la **distribución de los tamaños de clústeres**, ya que la presencia de clústeres muy pequeños o excesivamente grandes puede indicar problemas en la configuración de los parámetros. Para entender mejor la estructura de los datos, se pueden utilizar técnicas de visualización, como gráficos de dispersión coloreados por clústeres y mapas de calor de densidad, que permiten evaluar visualmente la efectividad de la agrupación.

En conclusión, la evaluación del rendimiento de DBSCAN requiere un enfoque multifacético que combine métricas cuantitativas, como el coeficiente de silueta y el índice de Davies-Bouldin, con análisis cualitativos, como la interpretación del número de puntos ruido y la visualización de los clústeres. Esta combinación permite ajustar los parámetros del algoritmo de manera efectiva y obtener agrupaciones significativas que reflejen la estructura real de los datos.

## Clustering basado en probabilidad: Modelos de mezcla de gaussianas (GMM)

### Introducción

Imagina que estás en una fiesta donde los invitados tienen diferentes intereses: algunos prefieren hablar de tecnología, otros de deportes y algunos más de arte. A simple vista, podrías agrupar a las personas según la conversación en la que participan. Sin embargo, algunas personas pueden estar interesadas en varios temas y moverse entre diferentes grupos, lo que hace difícil clasificarlas en una sola categoría de manera rígida.

El **clustering basado en probabilidad**, a diferencia de otros métodos como K-Means, que asignan a cada persona a un único grupo de conversación, permite reconocer que alguien puede pertenecer, en distintos grados, a más de un grupo. Por ejemplo, alguien podría tener un 70% de afinidad con las conversaciones sobre tecnología y un 30% con las de deportes. Esto proporciona una visión más realista y flexible, reflejando mejor la naturaleza compleja de los datos.

Este enfoque funciona bajo la idea de que los datos provienen de una combinación de patrones subyacentes, cada uno con su propia influencia. En lugar de dividirlos de manera tajante, los modelos probabilísticos evalúan qué tan probable es que cada observación pertenezca a un clúster determinado. Esto resulta particularmente útil en situaciones donde los grupos no están claramente definidos o cuando los datos se superponen en varias categorías.

El algoritmo más representativo de este enfoque es el **Modelo de Mezcla de Gaussianas (GMM, Gaussian Mixture Model)**, que asume que los datos están formados por una combinación de varias distribuciones gaussianas. Gracias a este modelo, es posible identificar patrones en datos complejos, como la segmentación de clientes, la detección de fraudes o la agrupación de imágenes, proporcionando una manera más detallada de comprender la estructura subyacente de los datos.

GMM es ampliamente utilizado en aplicaciones donde los datos no presentan límites bien definidos entre los grupos, como en la segmentación de clientes, la detección de anomalías y el procesamiento de señales biomédicas, donde es importante tener en cuenta la incertidumbre en la clasificación de las observaciones.

### Funcionamiento del algoritmo GMM

El modelo de mezcla de gaussianas entiende cada clúster como una distribución estadística en el espacio de características. Cada uno de estos clústeres se describe mediante una distribución gaussiana, que se caracteriza por tres elementos fundamentales: la **media**, la **covarianza** y el **peso**.

La **media** de cada componente define el punto central alrededor del cual se agrupan los datos, proporcionando una referencia para la posición del clúster en el espacio. Por ejemplo, si estuviéramos analizando el peso y la altura de diferentes poblaciones, la media indicaría la altura y el peso promedio de cada grupo identificado.

La **matriz de covarianza**, por su parte, describe la forma del clúster y su orientación en el espacio. No todos los grupos de datos tienen una distribución uniforme; algunos pueden estar más extendidos en una dirección específica o presentar correlaciones entre sus variables. La covarianza permite modelar estas relaciones, capturando la dispersión de los datos y proporcionando una representación más precisa de la estructura subyacente.

Por último, el **peso** de cada componente indica la importancia relativa de ese clúster dentro del conjunto total de datos. En términos simples, refleja la proporción de datos que se espera que pertenezcan a cada grupo.

En conjunto, la combinación de estas distribuciones gaussianas define la probabilidad total de cualquier punto de datos. Matemáticamente, el modelo se expresa como una suma ponderada de todas las distribuciones gaussianas individuales, donde cada una contribuye en función de su peso, media y covarianza:

$$
p(x) = \sum_{i=1}^{k} \pi_i \mathcal{N}(x \mid \mu_i, \Sigma_i)
$$

Donde:

- $k$ es el número de clústeres.
- $\mathcal{N}(x \mid \mu_i, \Sigma_i)$ representa la función de densidad de probabilidad de una distribución gaussiana multivariante.
- $\pi_i$ es el peso correspondiente a la $i$-ésima distribución gaussiana.

En esta ecuación, cada distribución gaussiana contribuye de manera diferente a la probabilidad total de un punto de datos, reflejando la naturaleza probabilística del modelo.

<img src=".\assets\1lTv7e4Cdlp738X_WFZyZHA.png" alt="Gaussian Mixture Models Explained | by Oscar Contreras Carrasco | Towards  Data Science" />

#### Método de **Expectación-Maximización (EM)**

El proceso de modelado en GMM consiste en encontrar la mejor combinación de estas distribuciones gaussianas para describir los datos de la manera más precisa posible. Para lograrlo, el algoritmo emplea un enfoque iterativo basado en el método de **Expectación-Maximización (EM)**, diseñado para estimar los parámetros desconocidos del modelo de manera eficiente.

El procedimiento comienza con una fase de **expectación**, en la cual el modelo, utilizando los parámetros actuales de cada distribución gaussiana, calcula la probabilidad de que cada punto de datos pertenezca a cada clúster. En esta etapa, no se realiza una asignación absoluta de pertenencia, sino que se calculan valores de **responsabilidad**, que indican el grado de pertenencia de cada punto a los distintos clústeres. Así, un punto podría tener, por ejemplo, un 60% de probabilidad de pertenecer a un clúster y un 40% a otro, lo que permite representar mejor la realidad en la que las categorías no siempre son claramente distinguibles.

Una vez calculadas estas responsabilidades, se procede a la fase de **maximización**, donde se utilizan los valores obtenidos en la fase anterior para actualizar los parámetros del modelo. En este paso, la media de cada clúster se recalcula como un promedio ponderado de los puntos según sus probabilidades de pertenencia, la matriz de covarianza se ajusta para reflejar la dispersión real de los datos, y los pesos de las distribuciones se actualizan para reflejar la proporción ajustada de cada clúster en la muestra.

Estas dos fases, de expectación y maximización, **se repiten de forma alternada en un ciclo iterativo**. En cada iteración, el modelo mejora su capacidad de representación de los datos hasta que los cambios en los parámetros sean mínimos o se alcance un criterio de convergencia predefinido. Al finalizar este proceso, GMM es capaz de ofrecer una segmentación que no solo clasifica cada punto en un clúster, sino que también proporciona una medida de certeza sobre esa clasificación, lo que lo convierte en una herramienta valiosa en aplicaciones donde la incertidumbre es un factor clave.

### Parámetros clave de GMM

La efectividad del modelo GMM depende de la correcta configuración de varios parámetros clave, que influyen en la precisión y la estabilidad del algoritmo.

Uno de los aspectos más importantes es la **elección del número de componentes gaussianas $k$**. A diferencia de DBSCAN, donde los clústeres se detectan en función de la densidad, en GMM el número de clústeres debe definirse previamente o seleccionarse mediante criterios como el **criterio de información bayesiano (BIC)** o el **criterio de información de Akaike (AIC)**, que permiten evaluar la calidad del modelo en función del equilibrio entre ajuste y complejidad.

Otro aspecto crítico es la **estructura de la matriz de covarianza**, que determina la forma de los clústeres que el modelo puede identificar. GMM permite diferentes configuraciones de covarianza, tales como:

- **Esférica:** Los clústeres tienen forma circular y el mismo tamaño en todas las dimensiones.
- **Diagonal:** Los clústeres tienen diferentes varianzas en cada dimensión, pero sin correlación entre ellas.
- **Completa:** Se permite cualquier forma y orientación, capturando relaciones complejas entre dimensiones.

![GMM classification — scikit-learn 0.15-git documentation](./assets/plot_gmm_classifier_0011.png)

> **Ejemplo:**
>Supongamos que se desea segmentar a los clientes de una tienda en función de su comportamiento de compra. Aplicando GMM con una estructura de covarianza completa, el modelo podría identificar clústeres con patrones de compra correlacionados, como clientes que gastan más en productos electrónicos también tienden a comprar artículos de hogar de alta gama.

### Ventajas y limitaciones de GMM

GMM ofrece varias ventajas que lo hacen atractivo para una amplia gama de aplicaciones en clustering.

Una de sus principales fortalezas es su **capacidad para modelar clústeres con formas elípticas**, lo que le otorga una mayor flexibilidad en comparación con algoritmos como K-Means, que asume agrupaciones esféricas. Además, al proporcionar probabilidades de pertenencia, GMM permite una clasificación más matizada, útil en situaciones donde los límites entre los clústeres no son claros.

Otra ventaja importante es la posibilidad de analizar la **incertidumbre en la clasificación**, ya que el modelo asigna probabilidades a cada punto de datos, lo que resulta útil en entornos donde los datos pueden tener una naturaleza ambigua o ruidosa.

Sin embargo, GMM también presenta ciertas limitaciones. Su rendimiento es sensible a la **inicialización de los parámetros**, lo que puede llevar a la convergencia en óptimos locales si los valores iniciales no están bien elegidos. Además, su **complejidad computacional** es mayor en comparación con algoritmos más simples, especialmente en conjuntos de datos grandes.

### Aplicaciones prácticas de GMM

El modelo de mezcla de gaussianas (GMM) ha demostrado ser una herramienta versátil en diversos campos donde la naturaleza de los datos es compleja y las fronteras entre grupos no están claramente definidas. Su capacidad para modelar distribuciones de datos de manera flexible lo hace especialmente útil en escenarios donde los clústeres pueden solaparse y donde se requiere un análisis probabilístico que permita manejar la incertidumbre en la clasificación.

En el campo del **procesamiento de imágenes**, GMM se utiliza ampliamente para la segmentación de regiones con características similares, como color, textura o brillo. Por ejemplo, en aplicaciones médicas, se pueden analizar imágenes de resonancia magnética o tomografías computarizadas, donde diferentes tejidos del cuerpo presentan niveles de intensidad que pueden modelarse mediante distribuciones gaussianas. A través de GMM, es posible identificar y separar estructuras anatómicas relevantes con una precisión superior a la que ofrecen métodos más rígidos como K-Means, ya que permite la superposición entre regiones con características similares.

Otro ámbito en el que GMM ha encontrado una aplicación clave es la **biología molecular**, donde el análisis de grandes volúmenes de datos genómicos requiere técnicas que puedan detectar patrones subyacentes de manera flexible. En estudios de expresión génica, por ejemplo, se pueden agrupar genes con perfiles de expresión similares a lo largo de diferentes condiciones experimentales, lo que ayuda a identificar posibles funciones compartidas o interacciones biológicas. La capacidad de GMM para modelar la variabilidad inherente a estos datos permite a los investigadores comprender mejor las relaciones entre los genes y sus roles en procesos celulares.

En el área de **reconocimiento de voz**, GMM es un pilar fundamental para la identificación de patrones acústicos, ya que permite modelar las características de los sonidos producidos por diferentes hablantes o fonemas. Durante el procesamiento de señales de audio, las características espectrales extraídas de la voz, como la frecuencia fundamental y la energía de las diferentes bandas, se pueden modelar mediante distribuciones gaussianas para capturar la variabilidad intrínseca de la pronunciación y los estilos de habla. Esta técnica ha sido ampliamente utilizada en sistemas de reconocimiento automático del habla y en aplicaciones de biometría, como la autenticación de usuarios basada en la voz.

En el ámbito financiero, GMM es una herramienta poderosa para el **análisis de comportamiento de clientes**, donde se busca agrupar a los consumidores según sus patrones de compra, frecuencia de uso de servicios o niveles de riesgo crediticio. Dado que los hábitos de los clientes no siempre se dividen en categorías bien definidas, GMM permite una segmentación más matizada, asignando probabilidades de pertenencia a diferentes perfiles de usuario. Esto permite a las instituciones financieras desarrollar estrategias más personalizadas, como ofertas de productos adaptadas a cada segmento o la detección temprana de posibles impagos. Además, en la **detección de fraudes**, GMM es capaz de identificar patrones de comportamiento anómalos al modelar el comportamiento típico de los usuarios y detectar transacciones que se desvían significativamente de las distribuciones normales esperadas.

> **Ejemplo:**
>Imagina una empresa de comercio electrónico que utiliza GMM para analizar los hábitos de compra de sus clientes. A través del modelo, la empresa identifica que algunos clientes tienden a comprar productos de lujo ocasionalmente, mientras que otros prefieren compras frecuentes de bajo costo. GMM permite asignar probabilidades a cada cliente, reflejando su pertenencia a múltiples segmentos a la vez y permitiendo diseñar estrategias de marketing más efectivas.

> [!tip]
>
> En resumen, GMM es una herramienta poderosa en entornos donde los datos presentan complejidad y donde la asignación probabilística de los puntos de datos permite obtener información más rica y detallada sobre los patrones ocultos en los datos. Desde la biología molecular hasta la seguridad financiera, su capacidad de modelado flexible continúa expandiendo sus aplicaciones en diversas disciplinas.

**Para reflexionar...**

> **¿En qué situaciones consideras que GMM sería más adecuado que K-Means para un problema de clustering?**
>**Clave:** Piensa en la capacidad de GMM para capturar distribuciones no esféricas y la importancia de modelar la incertidumbre en la clasificación.

## Implementación práctica de clustering con python

### Clustering K-Means

La implementación del algoritmo **K-Means** en Python es accesible gracias a potentes bibliotecas de aprendizaje automático y análisis de datos que facilitan su uso. Entre las más utilizadas se encuentra la ya habitual **Scikit-learn**, que proporciona herramientas eficientes para la construcción de modelos de clustering, clasificación y regresión, junto con funcionalidades de preprocesamiento y evaluación de modelos.

K-Means se implementa a través del módulo `sklearn.cluster`, que ofrece la clase `KMeans`, la cual permite ajustar el modelo a los datos de manera sencilla y flexible. Esta implementación incluye parámetros clave que controlan el comportamiento del algoritmo, como el número de clústeres a formar, el criterio de inicialización de los centroides y el número máximo de iteraciones para alcanzar la convergencia.

#### Claves para entender la clase `KMeans` en Scikit-learn

Al inicializar un modelo de clustering con la clase `KMeans`, es posible configurar distintos parámetros que afectan el proceso de agrupación. Algunos de los más relevantes incluyen:

- **`n_clusters`** *(int, por defecto=8)*
 Define el número de clústeres en los que se dividirán los datos. Es un parámetro obligatorio que debe seleccionarse cuidadosamente, ya que influye directamente en la calidad de la segmentación.
- *Claves:* Si se elige un número demasiado bajo, se perderá información relevante; si es demasiado alto, se pueden generar clústeres redundantes.
- *Ejemplo:* `KMeans(n_clusters=3)` agrupará los datos en tres clústeres.
- **`init`** *(‘k-means++’ por defecto, 'random' o matriz personalizada)*
 Determina cómo se inicializan los centroides antes de iniciar el proceso iterativo de clustering. La opción recomendada es `'k-means++'`, que selecciona los centroides de manera inteligente para acelerar la convergencia y mejorar los resultados. La opción `'random'` elige centroides de forma aleatoria, lo cual puede llevar a resultados inconsistentes.
- *Claves:* La inicialización adecuada puede reducir la cantidad de iteraciones necesarias para la convergencia y mejorar la estabilidad del resultado.
- *Ejemplo:* `KMeans(init='random')` seleccionará los centroides de forma aleatoria.
- **`n_init`** *(int, por defecto=1)*
 Especifica el número de veces que se ejecutará el algoritmo K-Means con diferentes inicializaciones de centroides. El mejor resultado, en términos de inercia (suma de distancias dentro de los clústeres), se conservará como solución final.
- *Claves:* Un valor mayor mejora la probabilidad de encontrar una mejor configuración, pero aumenta el tiempo de cómputo.
- *Ejemplo:* `KMeans(n_init=20)` ejecutará el algoritmo 20 veces con diferentes inicializaciones.
- **`max_iter`** *(int, por defecto=300)*
 Define el número máximo de iteraciones que el algoritmo realizará para intentar encontrar la solución óptima. Si los centroides dejan de cambiar antes de alcanzar este número, el algoritmo finalizará antes.
- *Claves:* Un número bajo puede llevar a una convergencia incompleta; un número demasiado alto puede hacer que el algoritmo consuma más recursos de los necesarios.
- *Ejemplo:* `KMeans(max_iter=500)` permite realizar hasta 500 iteraciones antes de detenerse.
- **`tol`** *(float, por defecto=1e-4)*
 Representa el criterio de tolerancia para la convergencia. Si el cambio en la suma de las distancias cuadradas dentro de los clústeres entre iteraciones consecutivas es menor que este valor, el algoritmo detendrá su ejecución.
- *Claves:* Un valor más pequeño hará que el algoritmo continúe buscando una mejor solución, mientras que un valor más grande puede detenerlo antes de tiempo.
- *Ejemplo:* `KMeans(tol=1e-3)` hará que el algoritmo termine si la mejora es menor a 0.001.

Una vez que el modelo ha sido ajustado a los datos, se pueden consultar **diversos atributos** que proporcionan información sobre el resultado del clustering. Los más importantes son:

- **`labels_`**
 Contiene las etiquetas asignadas a cada punto de datos, indicando a qué clúster pertenece cada observación.
- *Ejemplo:* `kmeans.labels_` devuelve un array con valores `[0, 1, 2, ...]`, donde cada número representa un clúster asignado.
- **`cluster_centers_`**
 Proporciona las coordenadas de los centroides finales de cada clúster, lo que permite analizar las características centrales de cada grupo.
- *Ejemplo:* `kmeans.cluster_centers_` devuelve una matriz con las posiciones de los centroides en el espacio de características.
- **`inertia_`**
 Muestra la suma de las distancias cuadradas de cada punto a su centroide más cercano, lo que sirve como una medida de la compacidad de los clústeres formados.
- *Claves:* Un valor bajo de inercia indica que los clústeres están bien definidos y los puntos están cerca de sus centroides.
- *Ejemplo:* `kmeans.inertia_` devuelve un valor numérico que representa la calidad del clustering.
- **`n_iter_`**
 Indica el número de iteraciones que el algoritmo realizó hasta alcanzar la convergencia.
- *Ejemplo:* `kmeans.n_iter_` devuelve un entero indicando cuántas iteraciones se realizaron antes de la convergencia.

La clase `KMeans` también proporciona **métodos esenciales** que permiten entrenar el modelo, predecir nuevas agrupaciones y optimizar el proceso de clustering:

- **`fit(X)`**
 Ajusta el modelo K-Means a los datos proporcionados. Durante este proceso, el algoritmo encuentra los centroides óptimos para los clústeres.
- *Ejemplo:* `kmeans.fit(X)` entrenará el modelo con los datos `X`.
- **`predict(X)`**
 Asigna etiquetas de clúster a nuevas observaciones basadas en la ubicación de los centroides aprendidos previamente.
- *Ejemplo:* `kmeans.predict(nuevos_datos)` asignará clústeres a nuevos puntos.
- **`fit_predict(X)`**
 Combina el ajuste del modelo y la predicción de etiquetas en una sola llamada, útil cuando se desea obtener rápidamente la clasificación de los datos.
- *Ejemplo:* `kmeans.fit_predict(X)` entrenará el modelo y devolverá las etiquetas de los clústeres para los datos `X`.

#### **Implementación paso a paso de K-Means**

A continuación, se muestra un ejemplo práctico de implementación del algoritmo K-Means en Python utilizando el conjunto de datos de ejemplo *Iris*, que contiene información sobre distintas especies de flores basada en características como la longitud y el ancho de los pétalos y sépalos.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Carga del conjunto de datos Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Preprocesamiento: estandarización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación del algoritmo K-Means con 3 clústeres
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
X['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualización de los resultados
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=X['Cluster'], palette='viridis')
plt.title('Clustering de Iris con K-Means')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Ancho del sépalo (cm)')
plt.legend(title='Clúster')
plt.show()

# Análisis de los centroides
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("Centroides de los clústeres:")
print(pd.DataFrame(centroids, columns=iris.feature_names))
```

La implementación comienza con la carga del conjunto de datos Iris, una colección de datos ampliamente utilizada para pruebas y aprendizaje en machine learning. El conjunto contiene cuatro características que describen las dimensiones de los sépalos y pétalos de tres especies diferentes de flores.

Dado que K-Means es un algoritmo basado en distancias, es crucial que las características estén en una escala similar para evitar que variables con rangos mayores dominen el proceso de agrupación. Para ello, se aplica la **normalización o estandarización** de los datos utilizando `StandardScaler` de Scikit-learn, que transforma cada característica para que tenga una media de 0 y una desviación estándar de 1.

Luego, se crea un modelo de K-Means especificando que se formarán tres clústeres, en concordancia con el número conocido de especies en el conjunto de datos. Se utilizan parámetros adicionales como:

- **`init='k-means++'`**, que selecciona de manera inteligente los centroides iniciales para mejorar la velocidad de convergencia.
- **`n_init=10`**, que establece cuántas veces se reinicializará el algoritmo para encontrar la mejor solución.
- **`max_iter=300`**, que define el número máximo de iteraciones para la convergencia.
- **`random_state=42`**, que fija la semilla aleatoria para obtener resultados reproducibles.

Una vez ajustado el modelo a los datos, se predicen las etiquetas de los clústeres, que posteriormente se agregan como una nueva columna en el dataframe original para facilitar el análisis.

Para interpretar visualmente los resultados, se genera un gráfico de dispersión utilizando las dos primeras características del conjunto de datos. Se colorean los puntos según los clústeres asignados por el algoritmo, lo que permite evaluar visualmente la segmentación lograda.

Finalmente, se imprimen los centroides de los clústeres en su escala original, utilizando la transformación inversa del escalado aplicado inicialmente. Estos centroides representan las posiciones promedio de cada grupo dentro del espacio de características, lo que proporciona información útil sobre las características distintivas de cada clúster.

Al aplicar K-Means, es importante considerar ciertos aspectos que pueden influir en la calidad del clustering:

- **Determinación del número óptimo de clústeres**: Seleccionar el valor adecuado de kk es fundamental. Se pueden utilizar técnicas como el método del codo (*elbow method*) o el coeficiente de silueta para evaluar la calidad de las agrupaciones y determinar el número óptimo de clústeres.
- **Sensibilidad a la inicialización**: Aunque el algoritmo K-Means++ mejora la elección de los centroides iniciales, sigue siendo importante ejecutar el algoritmo varias veces para obtener la mejor solución posible.
- **Formas de los clústeres**: K-Means tiende a formar clústeres esféricos, lo que puede no ser adecuado para datos con agrupaciones de forma más compleja.

Para evaluar el desempeño del clustering realizado con K-Means, se pueden utilizar métricas como:

- **Inercia (Suma de los errores cuadrados intra-clúster)**: Mide la compacidad de los clústeres; cuanto menor sea la inercia, mejor será la cohesión de los clústeres.
- **Coeficiente de silueta**: Evalúa qué tan bien separado está cada punto de su clúster en comparación con otros clústeres, proporcionando una medida entre -1 y 1.

Ejemplo de cálculo de estas métricas:

```python
from sklearn.metrics import silhouette_score

# Evaluación con la inercia
print(f"Inercia del modelo: {kmeans.inertia_}")

# Evaluación con el coeficiente de silueta
silhouette_avg = silhouette_score(X_scaled, X['Cluster'])
print(f"Coeficiente de silueta: {silhouette_avg:.2f}")
```

### Clustering jerárquico

Python ofrece una implementación eficiente del clustering jerárquico a través de la biblioteca **Scikit-learn**, que proporciona herramientas flexibles para aplicar y analizar este método. En particular, el módulo `sklearn.cluster` contiene la clase `AgglomerativeClustering`, que implementa el enfoque aglomerativo del clustering jerárquico, mientras que el módulo `scipy.cluster.hierarchy` permite visualizar el dendrograma y explorar las relaciones entre los datos.

#### Claves para entender la clase `AgglomerativeClustering` en Scikit-learn

El algoritmo de clustering jerárquico aglomerativo, implementado en la clase `AgglomerativeClustering`, permite construir clústeres fusionando observaciones de manera iterativa hasta formar una estructura jerárquica completa. Algunos de sus parámetros clave son:

- **`n_clusters`** *(int, por defecto=2)*
 Especifica el número de clústeres finales en los que se desea dividir los datos. A diferencia de otros métodos, en el clustering jerárquico es posible analizar la estructura completa antes de tomar una decisión sobre este valor.
- *Claves:* Si no se especifica, el modelo genera una estructura jerárquica sin un número fijo de clústeres.
- *Ejemplo:* `AgglomerativeClustering(n_clusters=3)` agrupará los datos en tres clústeres.
- **`affinity`** *(str, por defecto='euclidean')*
 Define la métrica de distancia utilizada para calcular la similitud entre las observaciones. Las opciones más comunes incluyen:
- `'euclidean'` (distancia euclidiana)
- `'manhattan'` (distancia de Manhattan)
- `'cosine'` (similitud de coseno)
- *Ejemplo:* `AgglomerativeClustering(affinity='manhattan')` utilizará la distancia de Manhattan para medir la similitud.
- **`linkage`** *(str, por defecto='ward')*
 Especifica el criterio de enlace, es decir, la forma en que se calcula la distancia entre clústeres durante la fusión. Las opciones disponibles son:
- `'ward'` (minimiza la varianza dentro de los clústeres)
- `'complete'` (distancia máxima entre puntos de los clústeres)
- `'average'` (promedio de las distancias entre todos los puntos)
- `'single'` (mínima distancia entre puntos de los clústeres)
- *Ejemplo:* `AgglomerativeClustering(linkage='complete')` usará la distancia máxima entre puntos al fusionar clústeres.
- **`distance_threshold`** *(float, por defecto=None)*
 Si se especifica, el proceso de agrupación continúa hasta alcanzar este umbral de distancia, permitiendo crear una estructura jerárquica sin necesidad de definir el número de clústeres de antemano.
- **`compute_full_tree`** *(bool, por defecto='auto')*
 Indica si se debe construir todo el árbol jerárquico o solo una parte relevante para formar los clústeres solicitados.

Una vez entrenado el modelo, la clase proporciona varios **atributos** útiles para analizar los resultados del clustering:

- **`labels_`**
 Contiene las etiquetas asignadas a cada punto, indicando el clúster al que pertenece cada observación.
- *Ejemplo:* `clustering.labels_` devolverá un array con los números de clústeres asignados a cada observación.
- **`n_clusters_`**
 Muestra el número de clústeres formados en la ejecución del modelo.

El clustering jerárquico en Scikit-learn se centra principalmente en el **método** `fit()`, que ajusta el modelo a los datos y permite analizar la estructura jerárquica.

- `fit(X)`

Ajusta el modelo a los datos de entrada y genera la jerarquía de clústeres.

- *Ejemplo:* `clustering.fit(X)` entrenará el modelo con los datos `X`.

#### Implementación paso a paso de clustering jerárquico

A continuación, se presenta un ejemplo práctico utilizando el conjunto de datos Iris, aplicando clustering jerárquico y visualizando el dendrograma resultante.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Carga del conjunto de datos Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Preprocesamiento: estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación del clustering jerárquico
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
X['Cluster'] = clustering.fit_predict(X_scaled)

# Visualización de resultados mediante un dendrograma
plt.figure(figsize=(10, 6))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='level', p=3, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrograma del Clustering Jerárquico')
plt.xlabel('Índice de muestras')
plt.ylabel('Distancia')
plt.show()

# Visualización de los clústeres
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=X['Cluster'], palette='viridis')
plt.title('Clustering Jerárquico de Iris')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Ancho del sépalo (cm)')
plt.legend(title='Clúster')
plt.show()
```

El proceso comienza cargando el conjunto de datos Iris, seguido de la estandarización de las características para garantizar que todas tengan la misma escala y evitar que una variable domine sobre las demás. Luego, se aplica el algoritmo de clustering jerárquico utilizando el criterio de enlace de **Ward**, que minimiza la varianza dentro de los clústeres y tiende a producir grupos más compactos.

Después de entrenar el modelo, se visualiza el dendrograma, que muestra la estructura jerárquica de los datos. En este gráfico, la altura de cada fusión indica la distancia entre los clústeres combinados, proporcionando información útil para decidir el número óptimo de clústeres mediante la identificación de "saltos" significativos en la distancia de enlace.

Finalmente, se realiza una visualización de los clústeres generados utilizando un gráfico de dispersión, lo que permite examinar cómo se agrupan las observaciones en función de sus características.

#### Consideraciones clave

Al aplicar el clustering jerárquico, es importante considerar los siguientes aspectos:

- **Elección del criterio de enlace**: Diferentes criterios pueden producir resultados significativamente distintos; por ejemplo, el enlace completo tiende a formar clústeres más equilibrados, mientras que el enlace simple puede generar cadenas de puntos.
- **Interpretación del dendrograma**: La selección del número de clústeres se realiza cortando el dendrograma en un nivel adecuado.
- **Escalabilidad**: El clustering jerárquico no es adecuado para grandes volúmenes de datos debido a su alta complejidad computacional.

### Implementación práctica del clustering DBSCAN

La clase `DBSCAN` en Scikit-learn proporciona una implementación flexible del algoritmo de clustering basado en densidad. Su funcionamiento se basa en la identificación de regiones densas en el espacio de características, lo que permite detectar agrupaciones de formas arbitrarias y detectar puntos de ruido.

Al configurar un modelo de clustering con DBSCAN, es fundamental comprender sus parámetros clave, así como los atributos y métodos que permiten analizar los resultados obtenidos.

Al crear una instancia de `DBSCAN`, se pueden ajustar varios **parámetros** que afectan el proceso de agrupación:

- **`eps`** *(float, por defecto=0.5)*
 Define el radio de vecindad dentro del cual un punto es considerado vecino de otro.
- *Claves:* Un valor muy pequeño puede fragmentar los clústeres, mientras que uno demasiado grande puede agrupar puntos no relacionados.
- *Ejemplo:* `DBSCAN(eps=0.7)` establecerá un radio de vecindad más amplio, lo que permitirá la formación de clústeres más grandes.
- **`min_samples`** *(int, por defecto=5)*
 Especifica el número mínimo de puntos que deben estar dentro del radio `eps` para considerar que un punto es un **núcleo** de un clúster.
- *Claves:* Valores pequeños pueden generar clústeres espurios, mientras que valores muy grandes pueden no identificar clústeres pequeños.
- *Ejemplo:* `DBSCAN(min_samples=10)` exigirá al menos 10 vecinos para formar un clúster.
- **`metric`** *(str o callable, por defecto='euclidean')*
 Determina la métrica de distancia utilizada para calcular la proximidad entre puntos. Se pueden usar métricas como `'euclidean'`, `'manhattan'`, `'cosine'` o definir una métrica personalizada.
- *Ejemplo:* `DBSCAN(metric='manhattan')` usará la distancia de Manhattan en lugar de la euclidiana.
- **`algorithm`** *(str, por defecto='auto')*
 Define el método utilizado para calcular las vecindades, con opciones como `'auto'`, `'ball_tree'`, `'kd_tree'` o `'brute'`.
- *Claves:* La elección del algoritmo afecta la velocidad de ejecución en grandes conjuntos de datos.
- *Ejemplo:* `DBSCAN(algorithm='ball_tree')` utilizará estructuras de árbol para una búsqueda más eficiente.
- **`leaf_size`** *(int, por defecto=30)*
 Influye en el rendimiento del cálculo de vecindades al usar estructuras de árbol. Un valor más pequeño puede mejorar la precisión, pero aumentar el tiempo de cómputo.
- *Ejemplo:* `DBSCAN(leaf_size=50)` ajustará el tamaño de los nodos de los árboles de búsqueda.
- **`n_jobs`** *(int, por defecto=None)*
 Controla la cantidad de núcleos de CPU utilizados para el cálculo. Un valor de `-1` utilizará todos los núcleos disponibles.
- *Ejemplo:* `DBSCAN(n_jobs=-1)` utilizará todos los núcleos disponibles para acelerar el proceso.

Una vez entrenado el modelo, la clase proporciona diversos **atributos** que permiten analizar los resultados del clustering:

- **`labels_`**
 Contiene las etiquetas de clúster asignadas a cada punto de datos. Los puntos considerados ruido son etiquetados con `-1`.
- *Ejemplo:* `dbscan.labels_` devolverá un array como `[0, 1, 1, -1, 2, 0]`, indicando clústeres y puntos de ruido.
- **`core_sample_indices_`**
 Indica los índices de las muestras que fueron identificadas como **puntos centrales**, es decir, aquellos que cumplen con la densidad mínima requerida.
- *Ejemplo:* `dbscan.core_sample_indices_` devolverá los índices de los puntos centrales en el conjunto de datos.
- **`components_`**
 Contiene las coordenadas de los puntos centrales, representando el núcleo de cada clúster identificado.
- *Ejemplo:* `dbscan.components_` devuelve un array con las coordenadas de los puntos centrales.
- **`eps`**
 Muestra el valor del radio de vecindad utilizado en el modelo.
- *Ejemplo:* `dbscan.eps` devolverá el valor configurado para `eps`.

La clase `DBSCAN` proporciona **métodos esenciales** para entrenar y analizar los resultados del modelo de clustering:

- **`fit(X)`**
 Ajusta el modelo a los datos de entrada, detectando los clústeres en función de la densidad especificada.
- *Ejemplo:* `dbscan.fit(X)` entrenará el modelo con los datos de entrada `X`.
- **`fit_predict(X)`**
 Ajusta el modelo y devuelve las etiquetas de clúster asignadas a cada observación. Es útil para obtener las agrupaciones de manera directa.
- *Ejemplo:* `labels = dbscan.fit_predict(X)` devolverá las etiquetas de clúster y ruido.

#### Implementación paso a paso de DBSCAN

Exploraremos la implementación de DBSCAN utilizando el conjunto de datos **"Mall Customers"**, el cual contiene información sobre clientes de un centro comercial, incluyendo su edad, género, ingresos anuales y puntajes de gasto. Este conjunto de datos es ideal para identificar patrones de comportamiento de clientes basados en su perfil de compra.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Carga del conjunto de datos
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
df = pd.read_csv(url)

# Selección de características relevantes (Ingresos anuales y Puntaje de gasto)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Preprocesamiento: estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación del algoritmo DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = dbscan.fit_predict(X_scaled)

# Visualización de los resultados
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], 
hue=df['Cluster'], palette='viridis', legend='full')
plt.title('Clustering de clientes con DBSCAN')
plt.xlabel('Ingresos anuales (k$)')
plt.ylabel('Puntaje de gasto (1-100)')
plt.legend(title='Clúster')
plt.show()

# Evaluación del clustering (ignorando puntos de ruido, etiqueta -1)
valid_clusters = df[df['Cluster'] != -1]['Cluster']
silhouette_avg = silhouette_score(X_scaled[df['Cluster'] != -1], valid_clusters)
print(f"Coeficiente de silueta para clústeres válidos: {silhouette_avg:.2f}")
```

La implementación comienza cargando el conjunto de datos *Mall Customers*, que contiene información sobre clientes de un centro comercial, incluyendo variables como ingresos anuales y puntaje de gasto. Para el clustering, se seleccionan estas dos variables, ya que proporcionan información clave sobre el perfil financiero y los hábitos de consumo de los clientes.

Dado que DBSCAN es un algoritmo basado en distancias, se aplica la **normalización de los datos** mediante `StandardScaler`, lo que garantiza que ambas características tengan la misma escala y contribuyan equitativamente al cálculo de distancias.

A continuación, se aplica el algoritmo **DBSCAN**, utilizando dos parámetros clave:

- **`eps=0.5`**: Define el radio máximo de vecindad dentro del cual se considera que un punto forma parte de un clúster.
- **`min_samples=5`**: Especifica el número mínimo de puntos necesarios dentro de la vecindad para que una región sea considerada densa.

Una vez ejecutado el modelo, se asignan etiquetas de clúster a cada punto de datos. DBSCAN asigna la etiqueta `-1` a los puntos considerados **ruido**, es decir, aquellos que no pertenecen a ninguna región densa.

Para visualizar los resultados, se genera un gráfico de dispersión que muestra los clústeres identificados en función de los ingresos anuales y el puntaje de gasto. Los puntos de ruido, si los hay, suelen aparecer como una categoría separada en la visualización.

Finalmente, se evalúa la calidad del clustering utilizando el **coeficiente de silueta**, una métrica que mide qué tan bien agrupados están los puntos dentro de sus clústeres. Dado que DBSCAN clasifica algunos puntos como ruido, estos se excluyen del cálculo para obtener una evaluación más precisa de los clústeres válidos.

#### Consideraciones clave al aplicar DBSCAN

Al trabajar con DBSCAN, es importante tener en cuenta ciertos aspectos que influyen en su desempeño:

- **Selección de los parámetros $\varepsilon$ y `min_samples`**: La elección de estos valores es crucial para obtener buenos resultados. Un valor de $\varepsilon$ demasiado pequeño puede fragmentar los clústeres, mientras que un valor demasiado grande puede agrupar puntos que no deberían estar juntos.
- **Sensibilidad a la densidad de los datos**: Si los clústeres tienen densidades muy variables, DBSCAN puede no identificar correctamente algunas agrupaciones.
- **Capacidad para detectar valores atípicos**: Una de las principales ventajas de DBSCAN es su habilidad para detectar puntos de datos que no pertenecen a ninguna agrupación, lo que puede ser útil en tareas de detección de anomalías.

### Implementación práctica del clustering GMM en Python

El algoritmo **GMM (Gaussian Mixture Model)** permite modelar los datos como una combinación de múltiples distribuciones gaussianas, proporcionando una manera flexible de realizar clustering en conjuntos de datos donde los clústeres pueden solaparse o presentar formas complejas. Su implementación en Python, como en los oros casos, es accesible gracias a la biblioteca **Scikit-learn**, que proporciona herramientas para ajustar y evaluar modelos de mezcla de gaussianas de manera eficiente.

La implementación de GMM se realiza a través del módulo `sklearn.mixture`, que contiene la clase principal `GaussianMixture`. Esta clase permite entrenar el modelo, realizar predicciones y evaluar la probabilidad de pertenencia de cada punto de datos a los distintos clústeres identificados.

La clase `GaussianMixture` ofrece una serie de parámetros clave que permiten controlar el comportamiento del modelo, así como atributos que proporcionan información útil sobre los resultados obtenidos.

#### **Parámetros clave de `GaussianMixture`**

- **`n_components`** *(int, por defecto=1)*
 Especifica el número de distribuciones gaussianas que compondrán el modelo, es decir, el número de clústeres a detectar.
- *Ejemplo:* `GaussianMixture(n_components=3)` agrupará los datos en tres clústeres.
- **`covariance_type`** *(str, por defecto='full')*
 Define la estructura de la matriz de covarianza para cada componente. Las opciones incluyen:
- `'full'`: Permite matrices de covarianza completas (formas elípticas).
- `'tied'`: Una única matriz de covarianza compartida por todos los clústeres.
- `'diag'`: Covarianzas diagonales (independencia entre características).
- `'spherical'`: Covarianza isotrópica en todas las direcciones.
- *Ejemplo:* `GaussianMixture(covariance_type='diag')` utilizará covarianzas diagonales.
- **`max_iter`** *(int, por defecto=100)*
 Especifica el número máximo de iteraciones que se realizarán durante el ajuste del modelo.
- *Ejemplo:* `GaussianMixture(max_iter=200)` permite realizar hasta 200 iteraciones.
- **`init_params`** *(str, por defecto='kmeans')*
 Método de inicialización de los parámetros del modelo, con opciones como:
- `'kmeans'`: Inicializa los centroides usando K-Means.
- `'random'`: Inicializa los parámetros aleatoriamente.
- *Ejemplo:* `GaussianMixture(init_params='random')` usará inicialización aleatoria.
- **`random_state`** *(int, por defecto=None)*
 Controla la reproducibilidad de los resultados al fijar una semilla aleatoria.
- *Ejemplo:* `GaussianMixture(random_state=42)` asegurará la reproducibilidad de los resultados.

#### **Atributos clave de `GaussianMixture`**

Una vez entrenado el modelo, se pueden consultar diversos atributos para interpretar los resultados obtenidos:

- **`weights_`**
 Muestra los pesos de cada componente gaussiana, reflejando la proporción de datos que pertenecen a cada clúster.
- *Ejemplo:* `gmm.weights_` devolverá un array con valores como `[0.3, 0.5, 0.2]`.
- **`means_`**
 Contiene las medias de cada componente gaussiana, indicando la posición de los centroides en el espacio de características.
- *Ejemplo:* `gmm.means_` devolverá un array con las coordenadas de los centroides.
- **`covariances_`**
 Muestra las matrices de covarianza asociadas a cada componente, describiendo la forma y orientación de los clústeres.
- *Ejemplo:* `gmm.covariances_` devolverá una lista de matrices de covarianza.
- **`converged_`**
 Indica si el algoritmo ha convergido dentro del número máximo de iteraciones.
- *Ejemplo:* `gmm.converged_` devolverá `True` si el modelo ha alcanzado la convergencia.

#### **Métodos principales de `GaussianMixture`**

- **`fit(X)`**
 Ajusta el modelo a los datos de entrada.
- *Ejemplo:* `gmm.fit(X)` entrenará el modelo con los datos `X`.
- **`predict(X)`**
 Asigna cada punto de datos a la distribución gaussiana más probable.
- *Ejemplo:* `gmm.predict(X)` devolverá las etiquetas de clústeres para cada muestra en `X`.
- **`predict_proba(X)`**
 Devuelve las probabilidades de pertenencia de cada punto a cada clúster.
- *Ejemplo:* `gmm.predict_proba(X)` mostrará la probabilidad de cada muestra para cada componente.

#### Implementación paso a paso de GMM

A continuación, se presenta un ejemplo práctico utilizando el conjunto de datos *Wine*, que contiene información química sobre distintas variedades de vino, permitiendo segmentarlas en función de sus características.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Carga del conjunto de datos Wine
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)

# Preprocesamiento: estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación del algoritmo GMM con 3 componentes
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
X['Cluster'] = gmm.fit_predict(X_scaled)

# Visualización de los resultados
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=X['Cluster'], palette='viridis')
plt.title('Clustering de vinos con GMM')
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.legend(title='Clúster')
plt.show()

# Análisis de los parámetros del modelo
print("Pesos de los clústeres:", gmm.weights_)
print("Medias de los clústeres:", gmm.means_)
```

El proceso de implementación del modelo de mezcla de gaussianas (GMM) comienza con la **carga y preprocesamiento de los datos**, utilizando el conjunto de datos *Wine*, el cual contiene diversas características químicas que describen diferentes variedades de vino. Dado que estas características tienen escalas distintas, se aplica una estandarización previa mediante la técnica de normalización estándar. Esta transformación garantiza que todas las variables tengan una media de 0 y una desviación estándar de 1, lo que es fundamental para mejorar la eficiencia y precisión del modelo, ya que GMM es un algoritmo sensible a la escala de los datos.

Una vez que los datos han sido preparados adecuadamente, se procede a la **configuración y entrenamiento del modelo**. En este caso, se especifica que el modelo debe identificar tres clústeres, reflejando el número esperado de variedades de vino en el conjunto de datos. Se opta por utilizar una matriz de covarianza completa, lo que permite que las distribuciones gaussianas adopten formas elípticas, capturando relaciones complejas entre las variables y proporcionando una mayor flexibilidad para la identificación de agrupaciones con estructuras variadas.

Con el modelo entrenado, se realiza la **predicción de los clústeres**, asignando a cada observación una etiqueta que indica su pertenencia a uno de los grupos detectados. Posteriormente, se lleva a cabo la **visualización** de los resultados mediante un gráfico de dispersión, donde se representan las muestras utilizando dos características seleccionadas del conjunto de datos. Los puntos se colorean en función del clúster al que pertenecen, lo que permite una interpretación intuitiva de la segmentación lograda y proporciona una idea visual de cómo las diferentes muestras de vino se agrupan en el espacio de características.

Finalmente, se realiza un **análisis detallado de los resultados** para comprender mejor la estructura de los clústeres identificados. Se extraen e imprimen los **pesos de los clústeres**, que reflejan la proporción relativa de cada grupo dentro del conjunto de datos, así como las **medias**, que indican las características promedio de cada clúster. Estos valores proporcionan información clave sobre las propiedades distintivas de cada agrupación y permiten interpretar el comportamiento general de los datos segmentados.

Este enfoque estructurado facilita la aplicación práctica del modelo GMM y proporciona una visión clara de su funcionamiento, desde la preparación de los datos hasta la interpretación final de los resultados obtenidos.

#### Claves a tener en cuenta al aplicar GMM

- La selección del número de componentes $k$ puede realizarse utilizando métricas como el **BIC** (Criterio de Información Bayesiano).
- GMM es sensible a la inicialización, por lo que se recomienda probar diferentes estrategias como `kmeans` o `random`.
- La interpretación probabilística permite analizar la incertidumbre en la clasificación, útil en aplicaciones donde los datos no están claramente separados.

