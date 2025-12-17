# Tema 5. Aprendizaje profundo y redes neuronales

## Redes convolucionales y visión por computadora

### Objetivos del módulo

> - **Entender cómo el cerebro procesa imágenes** y cómo esto inspiró el diseño de las redes convolucionales.
> - **Conocer la evolución de las CNN**, desde sus inicios hasta las arquitecturas modernas.
> - **Comprender por qué las MLP no son adecuadas para imágenes** y qué ventajas ofrecen las convoluciones.
> - **Aprender los conceptos clave de las CNN**, como filtros, convoluciones, stride, padding y pooling.
> - **Comparar la arquitectura de las CNN con las MLP** para entender sus diferencias fundamentales.
> - **Explorar las arquitecturas más influyentes**, como LeNet, AlexNet, VGG, GoogleNet y ResNet.



---

### **Fundamentos teóricos y evolución histórica**

Las redes neuronales convolucionales (**CNN, por sus siglas en inglés**) han transformado el campo del procesamiento de imágenes, permitiendo avances significativos en tareas como reconocimiento facial, diagnóstico médico por imágenes y conducción autónoma. A diferencia de las redes neuronales multicapa (**MLP**), las CNN están diseñadas para explotar la estructura espacial de los datos, capturando características jerárquicas de las imágenes mediante el uso de filtros y operaciones de convolución.

Para comprender la necesidad de las CNN, es fundamental revisar cómo los sistemas artificiales han abordado históricamente el reconocimiento de patrones en imágenes. Antes de la introducción de estas redes, los métodos tradicionales se basaban en ingeniería manual de características, utilizando transformaciones matemáticas y filtros diseñados para extraer bordes, texturas y formas. Sin embargo, estos enfoques presentaban limitaciones al enfrentarse a variaciones en la escala, iluminación y rotación de los objetos.

El desarrollo de las CNN estuvo inspirado en el funcionamiento del sistema visual biológico, específicamente en los estudios pioneros de **David Hubel y Torsten Wiesel** sobre la corteza visual del cerebro. Estos descubrimientos proporcionaron la base teórica para la creación de modelos computacionales que imitan la forma en que los organismos interpretan las imágenes. A partir de este punto, se han desarrollado diversas arquitecturas que han evolucionado para mejorar la eficiencia y precisión en el reconocimiento de imágenes.

A continuación, exploraremos la motivación biológica detrás de las CNN y cómo este conocimiento ha influido en su desarrollo.

#### **Motivación biológica**

El diseño de las redes neuronales convolucionales (**CNN**) está inspirado en la forma en que el cerebro procesa la información visual. En la década de 1960, los neurocientíficos **David Hubel y Torsten Wiesel** llevaron a cabo experimentos con gatos para estudiar cómo la corteza visual responde a estímulos externos. Sus hallazgos revelaron una organización jerárquica en el procesamiento de la información visual, lo que sentó las bases para el desarrollo de modelos computacionales de visión artificial.

Uno de los descubrimientos clave fue la existencia de **neuronas especializadas en la detección de patrones específicos**, como bordes con determinadas orientaciones. Hubel y Wiesel identificaron dos tipos principales de neuronas en la corteza visual:

- **Neuronas simples**, que responden a bordes o líneas en una posición y orientación específicas dentro de un área pequeña del campo visual.
- **Neuronas complejas**, que también detectan bordes y líneas, pero con mayor tolerancia a variaciones en la posición del estímulo dentro del campo visual.

Este modelo de procesamiento jerárquico sugiere que el cerebro descompone las imágenes en características elementales (bordes, texturas y formas básicas) antes de combinarlas en representaciones más abstractas. De manera similar, las **CNN aprenden a identificar características en distintos niveles de abstracción**, utilizando filtros convolucionales que detectan patrones desde bordes simples en capas iniciales hasta estructuras complejas en capas más profundas.

Esta capacidad de extraer características relevantes sin depender de ingeniería manual es una de las razones por las que las CNN han superado a los métodos tradicionales de procesamiento de imágenes. La conexión entre la visión biológica y la computación ha sido fundamental para el desarrollo de modelos más eficientes en tareas como reconocimiento facial, clasificación de imágenes y segmentación semántica.

A continuación, exploraremos la evolución de las redes neuronales desde sus primeras implementaciones hasta la llegada de las CNN.



#### **Evolución histórica de las redes convolucionales**

El desarrollo de las redes neuronales convolucionales no ocurrió de manera inmediata. Antes de que estas redes se convirtieran en el estándar para el procesamiento de imágenes, hubo una serie de avances que marcaron su evolución. Sus orígenes pueden rastrearse hasta los primeros modelos de redes neuronales, pero fue con la aparición de arquitecturas específicas para el reconocimiento de patrones cuando comenzaron a mostrar su verdadero potencial.

Uno de los primeros intentos en esta dirección fue el **Neocognitron**, desarrollado por **Kunihiko Fukushima** en 1981. Inspirado en la organización jerárquica de la corteza visual descubierta por **Hubel y Wiesel**, este modelo introdujo la idea de procesamiento en capas, donde cada nivel extraía características más abstractas de la imagen. A diferencia de las redes neuronales tradicionales de la época, el Neocognitron incorporaba **campos receptivos jerárquicos**, lo que permitía detectar patrones visuales sin depender de una ubicación específica dentro de la imagen. Gracias a este mecanismo, el modelo mostraba una mayor **invarianza a traslaciones**, lo que lo hacía más robusto al reconocimiento de formas desplazadas en el espacio. Sin embargo, aunque representó un avance significativo, carecía de un mecanismo eficiente de aprendizaje como la retropropagación, lo que limitó su adopción práctica.

> [!note]
>
> El concepto de **campo receptivo jerárquico** proviene del estudio de la **corteza visual** en neurociencia y ha sido adoptado en el diseño de las redes convolucionales. Se refiere a la región del espacio visual a la que responde una neurona en particular, es decir, el área de la imagen que influye en su activación. En el contexto de las redes convolucionales, cada neurona en una capa convolucional recibe información de una porción específica de la imagen en la capa anterior, lo que le permite detectar **patrones locales** como bordes, texturas o formas.
>
> Este mecanismo se organiza de manera **jerárquica**, de modo que las primeras capas de la red detectan características básicas, mientras que las capas más profundas combinan estas características para identificar estructuras más complejas. Por ejemplo, en una imagen de un rostro, las capas iniciales pueden identificar bordes y curvas, las intermedias pueden detectar partes como ojos o boca, y las últimas capas pueden reconocer rostros completos.
>
> Esta organización permite que las redes convolucionales **aprendan representaciones progresivamente más abstractas** a medida que la información fluye a través de sus capas, facilitando la tarea de clasificación y reconocimiento de patrones en imágenes.

El siguiente gran hito en la evolución de estas redes se produjo con el desarrollo de **LeNet-5**, una arquitectura creada por **Yann LeCun y colaboradores** en la década de los 80 y consolidada en 1998. Esta red fue la primera en aplicar con éxito la **retropropagación del error** dentro de una arquitectura convolucional, lo que permitió optimizar sus parámetros mediante el aprendizaje automático en grandes volúmenes de datos. Su estructura incorporaba **capas convolucionales y de pooling**, lo que le otorgaba la capacidad de extraer características jerárquicas de las imágenes de manera más eficiente. Estas mejoras hicieron que LeNet-5 se convirtiera en un referente en el reconocimiento de caracteres escritos a mano y, de hecho, se utilizó en sistemas bancarios para la lectura automática de cheques. Sin embargo, la falta de capacidad computacional de la época limitó su aplicación a problemas de mayor escala.

Tras el éxito de AlexNet, la comunidad de inteligencia artificial se volcó en la mejora de estas arquitecturas, desarrollando modelos más eficientes y profundos. En 2014, **VGGNet** introdujo una estructura más ordenada basada en **capas convolucionales de pequeño tamaño (3×3)**, lo que mejoró la precisión en la clasificación sin aumentar excesivamente la cantidad de parámetros. Ese mismo año, **GoogleNet** revolucionó el diseño de redes con la introducción de los **módulos Inception**, que permitían capturar múltiples escalas de características dentro de la misma capa convolucional, optimizando así el uso de los recursos computacionales. Posteriormente, en 2015, **ResNet** solucionó uno de los mayores problemas en redes profundas: el **desvanecimiento del gradiente**. A través de los **bloques residuales**, esta arquitectura permitió el entrenamiento de redes extremadamente profundas sin que la información se degradara en las capas más profundas del modelo.

Con el tiempo, las redes convolucionales han seguido evolucionando, incorporando nuevas técnicas y optimizaciones para mejorar su eficiencia y precisión. Desde los primeros experimentos con el **Neocognitron** hasta las arquitecturas modernas como **ResNet**, cada avance ha permitido que estas redes sean capaces de abordar problemas cada vez más complejos en el ámbito de la visión por computadora.

#### **Procesamiento tradicional vs. redes convolucionales**

Antes de la aparición de las redes convolucionales, el procesamiento de imágenes dependía en gran medida de **técnicas de ingeniería manual de características**. Estos métodos se basaban en el diseño de algoritmos específicos para extraer información relevante de las imágenes, utilizando transformaciones matemáticas y filtros predefinidos. Sin embargo, estos enfoques presentaban limitaciones significativas cuando se enfrentaban a problemas más complejos y diversos.

> **Cómo se almacena una imagen en un computador**
>
> Las imágenes digitales son representaciones de escenas visuales almacenadas en la memoria de un computador en forma de matrices de números. Cada imagen está compuesta por pequeñas unidades llamadas **píxeles**, que contienen información sobre el color o la intensidad de luz en cada punto.
>
> En una imagen en **escala de grises**, cada píxel tiene un único valor numérico que representa su nivel de intensidad, generalmente en un rango de **0 a 255**. El valor **0** representa el negro absoluto, el **255** representa el blanco y los valores intermedios corresponden a distintos niveles de gris.
>
> Por ejemplo, una imagen de **5 × 5 píxeles** se almacena en el computador como una matriz numérica:
>
> $$
> \begin{bmatrix}
> 0 &50 &100 & 150 & 200 \\
> 25 & 75 & 125 & 175 & 225 \\
> 50 & 100 & 150 & 200 & 250 \\
> 75 & 125 & 175 & 225 & 255 \\
> 100 & 150 & 200 & 250 & 255
> \end{bmatrix}
> $$
>
> Cada celda en esta matriz corresponde a un píxel de la imagen, donde los valores más bajos representan áreas oscuras y los más altos representan áreas más brillantes.
>
> Las imágenes en color se almacenan mediante la combinación de tres canales de color: **Rojo (R), Verde (G) y Azul (B)**. En este caso, cada píxel no tiene un único valor, sino tres valores, uno por cada canal de color.
>
> Por ejemplo, un píxel de una imagen en color puede estar definido por la siguiente combinación:
>
> $$
> \text{(R, G, B)} = (120, 200, 50)
> $$
>
> Esto significa que la intensidad del rojo es **120**, la del verde es **200** y la del azul es **50**, lo que en conjunto genera un tono específico.
>
> Cuando se almacena una imagen en color, el computador guarda tres matrices separadas, una para cada canal:
>
> $$
> R =
> \begin{bmatrix}
> 255 & 200 & 150 & 100 & 50 \\
> 255 & 200 & 150 & 100 & 50 \\
> 255 & 200 & 150 & 100 & 50 \\
> 255 & 200 & 150 & 100 & 50 \\
> 255 & 200 & 150 & 100 & 50
> \end{bmatrix}
> $$
>
> $$
> G =
> \begin{bmatrix}
> 50 & 100 & 150 & 200 & 255 \\
> 50 & 100 & 150 & 200 & 255 \\
> 50 & 100 & 150 & 200 & 255 \\
> 50 & 100 & 150 & 200 & 255 \\
> 50 & 100 & 150 & 200 & 255
> \end{bmatrix}
> $$
>
> $$
> B =
> \begin{bmatrix}
> 0 & 50 & 100 & 150 & 200 \\
> 0 & 50 & 100 & 150 & 200 \\
> 0 & 50 & 100 & 150 & 200 \\
> 0 & 50 & 100 & 150 & 200 \\
> 0 & 50 & 100 & 150 & 200
> \end{bmatrix}
> $$
>
> Cada píxel de la imagen se forma combinando los valores correspondientes en estas tres matrices.
>
> Las imágenes digitales pueden guardarse en diferentes formatos, como **JPEG, PNG o BMP**, dependiendo de cómo se comprimen y organizan los datos. Algunos formatos, como **BMP**, almacenan la imagen sin compresión, conservando todos los detalles pero ocupando más espacio. Otros, como **JPEG**, aplican algoritmos de compresión para reducir el tamaño del archivo a costa de una pequeña pérdida de calidad.
>
> Independientemente del formato, todas las imágenes se representan internamente como matrices de números, permitiendo que los computadores puedan procesarlas, analizarlas o modificarlas según sea necesario.
>

Los métodos tradicionales de visión artificial empleaban técnicas como **filtros de detección de bordes** (por ejemplo, Sobel o Canny), análisis de texturas mediante transformadas de Fourier o Wavelets, y descriptores como **SIFT (Scale-Invariant Feature Transform)** o **HOG (Histogram of Oriented Gradients)**. Estos algoritmos eran eficaces en tareas específicas, pero requerían un **diseño manual minucioso**, donde los ingenieros debían seleccionar las características más relevantes para cada problema.

Uno de los mayores desafíos de estos métodos era su **falta de adaptabilidad**. Dado que los filtros y descriptores eran estáticos, cualquier variación en la iluminación, escala, rotación o posición del objeto en la imagen podía afectar drásticamente el rendimiento del sistema. Además, en problemas más generales, como la clasificación de objetos en imágenes naturales, la ingeniería de características manual se volvía impracticable debido a la enorme variabilidad de los datos.

> [!note]
>
> Antes del auge de las redes convolucionales, el procesamiento de imágenes se realizaba mediante técnicas basadas en **filtros matemáticos** que extraían características visuales específicas. Un caso clásico es la **detección de bordes**, que se emplea en tareas como la segmentación de objetos dentro de una imagen.
>
> Uno de los métodos más utilizados es el **filtro de Sobel**, que detecta cambios abruptos en la intensidad de los píxeles, los cuales suelen estar asociados con los contornos de los objetos. Este filtro opera aplicando una convolución sobre la imagen con dos matrices predefinidas: una para detectar bordes en dirección horizontal y otra en dirección vertical.
>
> El funcionamiento del filtro de Sobel puede resumirse de la siguiente manera. Imaginemos una imagen en escala de grises donde queremos detectar los bordes de un objeto. Aplicamos dos filtros matemáticos en forma de matriz sobre la imagen, conocidos como **máscaras de Sobel**:
>
> Para detectar bordes horizontales, se podría usar una matriz como esta: 
> 
> $$
> G_x =
> \begin{bmatrix}
> -1 & 0 & 1 \\
> -2 & 0 & 2 \\
> -1 & 0 & 1
> \end{bmatrix}
> $$
> 
> Y para detectar bordes verticales, una como esta: 
> 
> $$
> G_y =
> \begin{bmatrix}
> -1 & -2 & -1 \\
>0 &0 &0 \\
>1 &2 &1
> \end{bmatrix}
> $$
> 
> Cada una de estas matrices se desplaza sobre la imagen, calculando un nuevo valor en cada posición mediante la suma ponderada de los píxeles cercanos. La magnitud del gradiente se obtiene combinando ambas matrices: 
> 
> $$
> G = \sqrt{G_x^2 + G_y^2}
> $$
>
> El resultado es una imagen en la que los bordes aparecen resaltados, mientras que las regiones con cambios suaves de intensidad quedan oscurecidas.
>
> Aunque este método es efectivo en ciertas condiciones, presenta limitaciones significativas cuando se enfrenta a imágenes más complejas:
>
> **Sensibilidad a la iluminación y el ruido**: Si la imagen tiene sombras o variaciones de brillo, los bordes pueden detectarse de manera inconsistente. Además, el ruido en la imagen puede generar bordes falsos, afectando la calidad del resultado.
>
> **Falta de adaptabilidad a distintas escalas**: Un objeto con bordes gruesos y otro con bordes finos pueden no ser detectados correctamente si el tamaño del filtro no está ajustado adecuadamente.
>
> **No captura información de alto nivel**: Aunque el filtro detecta bordes, no proporciona información sobre la estructura del objeto, su forma completa o su contexto dentro de la imagen.
>
> **Dificultad con rotaciones y traslaciones**: Si un objeto cambia de orientación o posición dentro de la imagen, los filtros tradicionales pueden no detectar sus bordes de la misma manera, lo que limita su capacidad de generalización.

#### **Comparación con las CNN**
A diferencia de los filtros tradicionales, una **red convolucional aprende sus propios filtros durante el entrenamiento**, ajustando automáticamente los valores óptimos para detectar bordes, texturas y formas más complejas. Además, las CNN pueden captar relaciones jerárquicas en las imágenes, combinando características simples en capas profundas para identificar objetos completos, sin necesidad de un ajuste manual de parámetros.

Mientras que el filtro de Sobel requiere ser diseñado y ajustado manualmente para cada tipo de imagen, una red convolucional se adapta dinámicamente al problema, logrando mejores resultados en tareas como el reconocimiento de objetos o la segmentación de imágenes.

La llegada de las **redes convolucionales** marcó un punto de inflexión en la visión por computadora al eliminar la necesidad de diseñar manualmente estas características. En lugar de aplicar filtros predefinidos, las CNN **aprenden automáticamente qué características son importantes** a través del entrenamiento en grandes conjuntos de datos. Mediante el uso de **capas convolucionales**, estas redes pueden detectar patrones jerárquicos, desde bordes simples en las capas iniciales hasta estructuras complejas en las capas más profundas.

Además, las CNN incorporan propiedades como la **invarianza a traslaciones**, permitiéndoles reconocer objetos en diferentes posiciones dentro de la imagen sin necesidad de ajustes manuales. Gracias a estas capacidades, las redes convolucionales han reemplazado por completo los métodos tradicionales en tareas como el reconocimiento facial, la conducción autónoma y el diagnóstico médico por imágenes, logrando niveles de precisión antes inalcanzables.

> [!note]
>
> La **invarianza a traslaciones** es una propiedad fundamental de las redes convolucionales que les permite **reconocer un mismo patrón sin importar su posición exacta en la imagen**. En otras palabras, una CNN puede detectar un objeto aunque este aparezca en diferentes partes de la imagen de entrada, lo que la hace mucho más eficiente que una red neuronal multicapa (**MLP**) tradicional.
>
> Esta propiedad surge gracias a dos mecanismos principales:
>
> **Uso de filtros convolucionales**: En una CNN, los filtros se aplican sobre toda la imagen mediante una operación de **convolución**, extrayendo características sin depender de su posición. De este modo, si un borde, una textura o una forma aparece en cualquier parte de la imagen, la red podrá detectarlo de manera uniforme.
>
> **Operaciones de pooling**: Para reforzar la invarianza a traslaciones, se utilizan técnicas como **max pooling**, que reducen la dimensionalidad de la imagen manteniendo solo las características más relevantes. Al seleccionar el valor máximo en regiones específicas, la red se vuelve más resistente a pequeños desplazamientos del objeto dentro de la imagen.
>
> En aplicaciones de visión por computadora, esta invarianza es esencial, ya que en el mundo real los objetos rara vez aparecen en la misma posición exacta en diferentes imágenes. Por ejemplo, en el reconocimiento facial, un rostro puede estar ligeramente desplazado o en distintas partes de la imagen, pero gracias a la invarianza a traslaciones, una CNN aún podrá reconocerlo correctamente.
>
> El verdadero punto de inflexión en la historia de las redes convolucionales llegó en 2012 con la aparición de **AlexNet**, desarrollada por **Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton**. Su impacto fue inmediato, ya que logró superar ampliamente a los modelos tradicionales en la competencia **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, donde la precisión en la clasificación de imágenes aumentó de manera significativa. Una de sus principales innovaciones fue la **profundidad de la red**, ya que contaba con múltiples capas convolucionales que le permitían extraer patrones más complejos en comparación con sus predecesoras. Además, introdujo la función de activación **ReLU (Rectified Linear Unit)**, lo que aceleró el entrenamiento al evitar problemas como la saturación del gradiente. Para reducir el sobreajuste, incorporó la técnica de **dropout**, que mejoraba la capacidad de generalización del modelo. Sin embargo, su mayor contribución fue el uso de **procesamiento en GPU**, lo que hizo viable el entrenamiento de redes profundas en tiempos razonables. Con estos avances, AlexNet marcó el inicio de la era del **Deep Learning**, consolidando el uso de redes convolucionales en el procesamiento de imágenes.

El procesamiento tradicional de imágenes, aunque sigue siendo útil en algunos contextos, ha quedado relegado a aplicaciones más específicas, mientras que las CNN se han convertido en la base de los sistemas modernos de visión por computadora.

### Fundamentos técnicos de las redes CNN

#### Limitaciones de las redes MLP en el procesamiento de imágenes

Las **redes neuronales multicapa (MLP)** fueron uno de los primeros enfoques utilizados para la clasificación de imágenes. Su estructura completamente conectada, en la que cada neurona de una capa recibe información de todas las neuronas de la capa anterior, permite modelar relaciones complejas entre las entradas. Sin embargo, cuando se aplican a imágenes, estas redes presentan **limitaciones estructurales y computacionales** que dificultan su uso eficiente.

Para entender estos problemas, imaginemos una imagen en **escala de grises** de **100 × 100 píxeles**. Para que una MLP pueda procesarla, cada píxel debe convertirse en una entrada independiente, lo que significa que la imagen se representa como un **vector de 10.000 valores** (100 × 100). Si esta imagen es procesada por una **primera capa oculta de 256 neuronas**, cada neurona necesitará un peso para cada una de las 10.000 entradas, además de un sesgo.

El número total de **pesos entrenables** en esta primera capa se obtiene multiplicando el número de entradas por el número de neuronas:

$$
\text{Total de pesos} = 256 \times 10.000 = 2.560.000
$$

Dado que cada neurona también tiene un sesgo asociado, el total de parámetros en la primera capa se incrementa en 256 unidades adicionales:

$$
2.560.000 + 256 = 2.560.256
$$

Este número de parámetros resulta muy elevado, y aún no hemos considerado las capas posteriores. Si la red contara con más capas ocultas de tamaño similar, la cantidad de conexiones crecería de manera exponencial, haciendo que el entrenamiento sea computacionalmente costoso y difícil de manejar. Además, entrenar un modelo con una gran cantidad de parámetros requiere **una enorme cantidad de datos**, ya que, de lo contrario, la red corre el riesgo de memorizar los ejemplos de entrenamiento en lugar de aprender patrones generales. Este fenómeno, conocido como **sobreajuste**, limita la capacidad de la red para generalizar sobre datos nuevos y disminuye su rendimiento en aplicaciones reales.

Otro problema fundamental es que al **convertir la imagen en un vector plano**, se pierde por completo la estructura espacial de los píxeles. En una imagen, los píxeles cercanos suelen estar relacionados entre sí y conforman bordes, texturas y formas reconocibles. Sin embargo, en una MLP, cada píxel se trata de manera aislada, sin ninguna información sobre su relación con los píxeles vecinos. Como resultado, la red no puede aprovechar las características locales de la imagen, lo que afecta su capacidad para reconocer patrones visuales de manera eficiente.

Además, este enfoque presenta **dificultades para manejar objetos en diferentes posiciones dentro de la imagen**. Si el mismo objeto aparece en una ubicación distinta, la red lo percibe como un conjunto completamente nuevo de valores de entrada, lo que impide la generalización. En la práctica, esto significa que un modelo entrenado con una imagen en una posición específica podría no ser capaz de reconocer el mismo objeto si aparece desplazado en la imagen.

Por último, el uso de redes MLP en imágenes implica **un alto consumo de memoria y tiempo de procesamiento**. La necesidad de almacenar y actualizar millones de pesos en cada iteración del entrenamiento hace que este proceso sea ineficiente, especialmente cuando se trabaja con conjuntos de datos grandes. Entrenar una red con millones de parámetros requiere hardware especializado y tiempos de cómputo prolongados, lo que limita su aplicabilidad en tareas prácticas de visión por computadora.

Estos problemas hacen que el uso de MLP para el procesamiento de imágenes sea poco eficiente y poco escalable. En la práctica, esta arquitectura ha sido reemplazada por enfoques más adecuados, diseñados para extraer información de manera más estructurada y eficiente.

#### **El concepto de kernel en el procesamiento de imágenes**

En el procesamiento de imágenes, un **kernel** es una pequeña matriz numérica utilizada para realizar operaciones sobre una imagen. Se le conoce también como **filtro** y es una herramienta fundamental en muchas técnicas de procesamiento digital, incluyendo la detección de bordes, el suavizado y la convolución en redes neuronales convolucionales.

Un kernel es una matriz de dimensiones reducidas, generalmente de **3 × 3**, **5 × 5** o **7 × 7**, que se desplaza sobre la imagen y modifica sus valores mediante una operación matemática específica. Su propósito es resaltar ciertas características de la imagen, como bordes, texturas o patrones particulares.

Para entender mejor su funcionamiento, imaginemos una imagen en escala de grises representada como una matriz de píxeles y un kernel de **3 × 3** aplicado sobre ella. En cada posición, el kernel multiplica sus valores por los valores correspondientes en la imagen y luego se suma el resultado, generando un nuevo valor en la posición central. Esta operación se repite a lo largo de toda la imagen, transformándola de acuerdo con el tipo de kernel utilizado.

Dependiendo de los valores que contenga, un kernel puede cumplir diferentes funciones. Por ejemplo, si los valores del kernel están diseñados para detectar cambios bruscos en la intensidad de los píxeles, se puede utilizar para **detección de bordes**. En cambio, si los valores están distribuidos de manera uniforme, se puede utilizar para **suavizar** la imagen y reducir el ruido.

El uso de kernels en el procesamiento de imágenes ha sido un método tradicional en visión artificial mucho antes de la aparición de las redes convolucionales. Sin embargo, en el caso de las CNN, estos filtros no son predefinidos manualmente, sino que son **aprendidos automáticamente** a partir de los datos de entrenamiento, lo que permite optimizar la extracción de características relevantes para la tarea específica.

A continuación, exploraremos la operación de convolución, que es el mecanismo mediante el cual se aplican los kernels a una imagen y se obtienen transformaciones que permiten extraer información útil.

#### **La operación de convolución**

La **convolución** es una operación matemática utilizada en el procesamiento de imágenes para aplicar filtros o **kernels** sobre una imagen y extraer características relevantes. Es el mecanismo fundamental en el que se basan las **redes neuronales convolucionales (CNN)** para capturar patrones espaciales en una imagen.

El proceso de convolución consiste en **desplazar un kernel** sobre la imagen de entrada y calcular una nueva matriz llamada **mapa de características** (**feature map**), que representa la respuesta del filtro en cada posición. En cada paso, los valores del kernel se multiplican por los valores correspondientes de la imagen, y el resultado se suma para obtener un nuevo valor en la salida.

> **Ejemplo de operación de convolución**
>
> Imaginemos una imagen en escala de grises de **5 × 5 píxeles** y un kernel de **3 × 3** diseñado para detectar bordes. La imagen y el kernel se representan de la siguiente manera:
>
> ##### **Imagen original**
> 
> $$
> \begin{bmatrix}
> 10 & 20 & 30 & 40 & 50 \\
> 20 & 30 & 40 & 50 & 60 \\
> 30 & 40 & 50 & 60 & 70 \\
> 40 & 50 & 60 & 70 & 80 \\
> 50 & 60 & 70 & 80 & 90
> \end{bmatrix}
> $$
>
> ##### **Kernel de detección de bordes**
> 
> $$
> \begin{bmatrix}
> -1 & -1 & -1 \\
>0 &0 &0 \\
>1 &1 &1
> \end{bmatrix}
> $$
>
> El kernel se superpone sobre la parte superior izquierda de la imagen, y en cada posición se realiza el producto elemento a elemento seguido de la suma. Por ejemplo, en la primera posición: 
>
> $$
> (-1 \times 10) + (-1 \times 20) + (-1 \times 30) + (0 \times 20) + (0 \times 30) + (0 \times 40) + (1 \times 30) + (1 \times 40) + (1 \times 50) = 0
> $$
>
> Este cálculo se repite desplazando el kernel sobre toda la imagen, generando el **mapa de características** resultante. El mapa de características representa las respuestas del filtro a los distintos patrones detectados en la imagen.
>

##### **Cálculo del tamaño de salida tras la convolución**

Cuando se aplica una convolución sobre una imagen, el tamaño del resultado final, conocido como **mapa de características**, depende del tamaño de la imagen de entrada y del tamaño del kernel utilizado. Para determinar cuántas veces puede encajar el kernel dentro de la imagen sin salirse de los límites, se emplea la siguiente fórmula:

$$
O = (I - K) + 1
$$

donde: 
- $O$ es el tamaño del **mapa de características** obtenido tras la convolución.
- $I$ es el tamaño de la **imagen de entrada** (número de filas o columnas).
- $K$ es el tamaño del **kernel** (número de filas o columnas).

Por ejemplo, si se tiene una imagen de **5 × 5 píxeles** y se aplica un kernel de **3 × 3**, el tamaño de la salida será:

$$
O = (5 - 3) + 1 = 3
$$

Esto significa que el **mapa de características** tendrá un tamaño de **3 × 3 píxeles**.

Este cálculo es el resultado natural de cómo el kernel se desplaza sobre la imagen. Al comenzar en la esquina superior izquierda y moverse una posición a la vez hasta alcanzar el borde, se obtiene un área de salida más pequeña que la imagen original. A medida que se utilicen diferentes tamaños de kernel o estrategias adicionales para manejar los bordes, este tamaño podrá ajustarse según las necesidades del modelo.

#### El concepto de padding.

En la operación de convolución, el padding es una técnica que permite añadir píxeles adicionales alrededor de la imagen antes de aplicar el kernel. Su propósito es evitar la reducción progresiva del tamaño de la imagen tras múltiples convoluciones y asegurar que los bordes sean procesados de manera adecuada. Sin padding, cada aplicación del kernel sobre la imagen disminuye sus dimensiones, lo que se traduce en una pérdida significativa de información a medida que se avanza en la arquitectura de la red.

El problema fundamental que motiva el uso de padding radica en la forma en que se desplaza el kernel sobre la imagen. Dado que la convolución solo puede aplicarse dentro de los límites definidos por la imagen, los píxeles ubicados en los bordes contribuyen menos veces al resultado final en comparación con los píxeles centrales. Esta desigualdad en la participación de los datos provoca que las regiones periféricas pierdan relevancia en el proceso de extracción de características, lo que puede afectar negativamente el rendimiento del modelo en tareas como la detección de contornos o la clasificación de objetos.

Para mitigar estos efectos, se pueden utilizar diferentes estrategias de padding. Una de las más comunes es el **zero-padding**, donde se agregan filas y columnas compuestas exclusivamente por ceros en los bordes de la imagen. Este método facilita la aplicación del kernel sin introducir información adicional y permite mantener la estructura del proceso de convolución sin afectar la distribución de valores en la imagen original. Otra variante es el padding basado en replicación, en el que los valores de los píxeles del borde se extienden hacia las nuevas filas y columnas añadidas, conservando así cierta coherencia en la distribución espacial de la imagen.

El impacto del padding sobre el tamaño final del mapa de características puede analizarse matemáticamente. Si se considera una imagen de entrada de $5 \times 5$ píxeles y un kernel de $3 \times 3$, sin aplicar padding, la salida resultante tendrá un tamaño de $3 \times 3$. Sin embargo, si se añade un borde de un píxel alrededor de la imagen, esta se expande a $7 \times 7$ píxeles, permitiendo que la convolución genere una salida con dimensiones de $5 \times 5$, es decir, manteniendo el tamaño original de la imagen de entrada.

Para calcular el tamaño de salida en presencia de padding, la fórmula se modifica de la siguiente manera:

$$
O = (I - K) + 2P + 1
$$

Donde:
- $O$ es el tamaño del mapa de características resultante.
- $I$ es el tamaño de la imagen de entrada.
- $K$ es el tamaño del kernel.
- $P$ es el tamaño del padding aplicado en los bordes.

Si se considera nuevamente una imagen de entrada de $5 \times 5$ píxeles, un kernel de $3 \times 3$ y un padding de $1$ píxel, el cálculo del tamaño de salida se realiza como sigue:

$$
O = (5 - 3) + 2(1) + 1 = 5
$$

Esta capacidad de preservar la estructura espacial de los datos hace que el padding sea una herramienta fundamental en el diseño de modelos convolucionales. Su empleo adecuado no solo permite que las imágenes se mantengan en un tamaño controlado a lo largo de múltiples capas, sino que también garantiza que los patrones en los bordes de la imagen sean considerados en el proceso de extracción de características.

#### **El concepto de stride en la convolución**

En la operación de convolución, el **stride** es el parámetro que define el número de píxeles en los que se desplaza el kernel a medida que recorre la imagen. En una convolución estándar, el stride es igual a **1**, lo que significa que el kernel se mueve un píxel a la vez en ambas direcciones (horizontal y vertical). Sin embargo, se puede aumentar el stride para que el kernel avance en pasos mayores, reduciendo así el tamaño del mapa de características resultante.

El uso de un stride mayor tiene un impacto directo en la cantidad de información retenida después de la convolución. Cuando el kernel se mueve de un píxel en un píxel, la salida mantiene una resolución relativamente alta. En cambio, cuando el stride es mayor, la imagen resultante es más pequeña porque se omiten ciertas posiciones en el proceso de filtrado.

Esta reducción del tamaño de salida es útil en arquitecturas profundas, ya que disminuye la cantidad de parámetros y el costo computacional. Sin embargo, un stride demasiado grande puede hacer que se pierdan detalles importantes de la imagen, afectando la capacidad de la red para capturar estructuras relevantes.

##### **Cálculo del tamaño de salida con stride y padding**

Cuando se combinan **stride** y **padding**, el tamaño del mapa de características se calcula con la siguiente fórmula:

$$
O = \frac{(I - K) + 2P}{S} + 1
$$

donde:
- $O$ es el tamaño del mapa de características de salida.
- $I$ es el tamaño de la imagen de entrada.
- $K$ es el tamaño del kernel.
- $P$ es el padding aplicado.
- $S$ es el stride, es decir, el número de píxeles que el kernel se desplaza en cada paso.

Si se considera una imagen de entrada de $5 \times 5$ píxeles, con un kernel de $3 \times 3$, un padding de $1$ y un stride de $2$, se obtiene:

$$
O = \frac{(5 - 3) + 2(1)}{2} + 1 = 3
$$

Esto indica que el mapa de características final tendrá un tamaño de $3 \times 3$ píxeles, en lugar de los $5 \times 5$ originales.

El stride es un parámetro esencial en redes convolucionales, ya que permite controlar la resolución de salida y ajustar el número de parámetros de la red. Su uso debe equilibrarse para evitar la pérdida excesiva de información y garantizar que se conserven las características más relevantes de la imagen. 

#### **El concepto de pooling en redes convolucionales**

El **pooling** es una operación utilizada en redes convolucionales para reducir la dimensión espacial de los mapas de características generados tras la convolución. A diferencia de la convolución, que aplica un conjunto de pesos entrenables para extraer patrones, el pooling no implica aprendizaje, sino que actúa como un filtro de reducción que conserva la información más relevante de cada región de la imagen.

La operación de pooling se lleva a cabo dividiendo la imagen en regiones no superpuestas y aplicando una función de agregación sobre cada una de ellas. Generalmente, se utiliza un tamaño de ventana de $2 \times 2$ con un desplazamiento de 2 píxeles, lo que reduce las dimensiones de la imagen a la mitad en cada dirección. Existen diferentes tipos de pooling, entre los cuales los más comunes son:

- **Max pooling**: selecciona el valor máximo dentro de cada región, lo que permite conservar las características más prominentes de la imagen, como los bordes o los puntos de alto contraste.
- **Average pooling**: calcula el valor promedio de cada región, generando una versión más suavizada del mapa de características.

A diferencia de la convolución, que transforma la imagen extrayendo nuevas características, el pooling solo reduce su tamaño manteniendo las características más representativas. Mientras que la convolución genera una representación jerárquica de los patrones presentes en la imagen, el pooling tiene un efecto más estructural, asegurando que las características detectadas sean más compactas y manejables para las capas posteriores de la red.

El objetivo fundamental del pooling es proporcionar **reducción de dimensionalidad** y **control de sobreajuste**. Al disminuir la cantidad de información sin perder las características clave, se logra un procesamiento más eficiente y se evitan problemas como la redundancia en la información. Además, el pooling introduce un nivel de **invarianza a pequeñas traslaciones**, ya que pequeños desplazamientos en la imagen no afectan significativamente la activación de los filtros.

> **Ejemplo de pooling**
>
> Para ilustrar el proceso de pooling, consideremos una imagen representada por la siguiente matriz de **4 × 4** píxeles:
>
> $$
> \begin{bmatrix}
> 1 & 3 & 2 & 4 \\
> 5 & 6 & 7 & 8 \\
> 9 & 2 & 4 & 3 \\
> 1 & 0 & 6 & 5
> \end{bmatrix}
> $$
>
> Si aplicamos **max pooling** con una ventana de $2 \times 2$ y desplazamiento de 2 píxeles, tomamos el valor máximo de cada bloque **no superpuesto**:
>
> 1. De la región $\begin{bmatrix} 1 & 3 \\ 5 & 6 \end{bmatrix}$ se toma el **6**.
> 2. De la región $\begin{bmatrix} 2 & 4 \\ 7 & 8 \end{bmatrix}$ se toma el **8**.
> 3. De la región $\begin{bmatrix} 9 & 2 \\ 1 & 0 \end{bmatrix}$ se toma el **9**.
> 4. De la región $\begin{bmatrix} 4 & 3 \\ 6 & 5 \end{bmatrix}$ se toma el **6**.
>
> El resultado del **mapa de características tras max pooling** es:
>
> $$
> \begin{bmatrix}
> 6 & 8 \\
> 9 & 6
> \end{bmatrix}
> $$
>
> Esto demuestra cómo el pooling reduce la dimensionalidad de la imagen original, pasando de **4 × 4** a **2 × 2**, pero manteniendo las características más representativas.
>
> El pooling, por lo tanto, juega un papel crucial en la reducción del tamaño de los mapas de características sin eliminar información esencial, permitiendo que las redes convolucionales sean más eficientes y generalicen mejor en tareas de visión por computadora.

#### **Generalización del proceso de convolución: casos de uso**

El proceso de convolución puede extenderse a distintos escenarios más complejos, donde es necesario considerar factores adicionales en el cálculo de las dimensiones de los mapas de características. Dos casos fundamentales de generalización son la aplicación de convoluciones sobre imágenes en color (**RGB**) y el uso de múltiples filtros sobre una misma imagen. En cada uno de estos casos, es posible calcular la dimensión de salida utilizando las reglas establecidas para la convolución con padding y stride.

##### **Caso 1: Convolución sobre imágenes RGB**

Las imágenes en color no se representan mediante una única matriz de valores de intensidad, como en el caso de imágenes en escala de grises, sino mediante tres canales independientes correspondientes a los colores **Rojo (R), Verde (G) y Azul (B)**. Cada uno de estos canales puede considerarse como una imagen en escala de grises y, por lo tanto, cada convolución se aplica individualmente sobre cada canal.

Si se tiene una imagen de entrada de tamaño $I \times I \times 3$ (donde el último valor indica que hay tres canales de color), se utiliza un kernel tridimensional de tamaño $K \times K \times 3$. Es importante notar que el kernel debe tener el mismo número de canales que la imagen para que la operación de convolución sea válida.

El cálculo del tamaño de salida se realiza aplicando la fórmula:

$$
O = \frac{(I - K) + 2P}{S} + 1
$$

Sin embargo, dado que la convolución se aplica de manera independiente a cada canal y se combinan los resultados en un solo mapa de características, la salida tendrá dimensiones $O \times O \times 1$.

> **Ejemplo**
>
> Si se tiene una imagen **RGB** de tamaño **128 × 128 × 3** y se aplica un **kernel de 5 × 5 × 3**, con un padding de **2** y un stride de **1**, se obtiene: 
> 
> $$
> O = \frac{(128 - 5) + 2(2)}{1} + 1 = 128
> $$
>
> Por lo tanto, el mapa de características resultante tendrá dimensiones **128 × 128 × 1**. Si se aplican múltiples filtros, la salida puede extenderse a más mapas de características, como veremos en el siguiente caso.
>

##### **Caso 2: Convolución con múltiples filtros**

En redes convolucionales, es común aplicar múltiples filtros a una misma imagen para extraer diferentes tipos de características. En este caso, en lugar de producir un único mapa de características como salida, se obtiene un conjunto de mapas, uno por cada filtro utilizado.

Si se tiene una imagen de entrada con tamaño $I \times I \times C$, donde $C$ es el número de canales de la imagen (como en el caso RGB donde $C=3$), y se aplica un conjunto de $F$ filtros de tamaño $K \times K \times C$, la salida tendrá dimensiones:

$$
O = \frac{(I - K) + 2P}{S} + 1
$$

pero ahora la profundidad de la salida será igual al número de filtros utilizados, es decir, tendrá dimensiones **$O \times O \times F$**.

> **Ejemplo**
>
> Si se tiene una imagen **RGB** de tamaño **128 × 128 × 3** y se aplican **32 filtros** de tamaño **5 × 5 × 3**, con un padding de **2** y un stride de **1**, se obtiene el mismo tamaño de salida en términos espaciales que en el caso anterior: 
>
> $$
> O = \frac{(128 - 5) + 2(2)}{1} + 1 = 128
> $$
>
> Pero dado que ahora hay **32 filtros**, la salida tendrá dimensiones **128 × 128 × 32**.
>
> Este caso muestra cómo las redes convolucionales pueden aprender múltiples representaciones de una misma imagen al aplicar distintos filtros en paralelo. Cada filtro detectará patrones diferentes en la imagen, como bordes, texturas o estructuras más complejas, enriqueciendo la información extraída en cada capa convolucional.
>

Estos ejemplos ilustran cómo la convolución puede extenderse a distintos escenarios, manteniendo un esquema de cálculo sistemático que permite controlar la dimensionalidad de las representaciones en cada etapa del modelo.

### **Arquitectura de una red convolucional**

Las redes neuronales convolucionales no solo aplican operaciones de convolución para extraer características de las imágenes, sino que integran estos procesos dentro de una arquitectura más extensa con el objetivo de realizar tareas diversas como clasificación, segmentación o detección de objetos. En el caso más general, una CNN combina varias capas convolucionales con mecanismos de reducción de dimensionalidad y finalmente conecta estas representaciones con una red neuronal completamente conectada que toma decisiones en función de las características extraídas.

El flujo de procesamiento en una **CNN para clasificación** puede dividirse en tres fases principales: 

**Extracción de características mediante capas convolucionales** 
La primera etapa de la red consiste en aplicar convoluciones a la imagen de entrada. Cada capa convolucional utiliza un conjunto de filtros que detectan patrones específicos como bordes, texturas o estructuras más complejas. En esta etapa, la imagen es transformada en una serie de mapas de características que resaltan la información relevante para la tarea de clasificación. 

**Reducción de dimensionalidad mediante pooling**
Dado que los mapas de características pueden ser de gran tamaño, se aplican operaciones de **pooling** para reducir la resolución espacial sin perder información clave. Esto permite disminuir la carga computacional y hacer la red más robusta a pequeñas variaciones en la imagen, como cambios de posición o escala. 

**Clasificación con capas completamente conectadas** 
Una vez que la imagen ha sido transformada en una representación compacta mediante convoluciones y pooling, la información es pasada a una red completamente conectada (**fully connected layer**). Esta parte de la red se comporta como un clasificador tradicional, donde cada neurona recibe información de todas las unidades de la capa anterior y aprende a asociar los patrones detectados con las categorías de salida.

> **Ejemplo: Flujo de una imagen a través de una CNN**
>
> Para ilustrar este proceso, consideremos una imagen de entrada de **128 × 128 × 3** píxeles que pasa por una CNN diseñada para clasificación. La arquitectura típica de esta red puede estructurarse de la siguiente manera:
>
> **Primera convolución**
>
> - Se aplican **32 filtros de 3 × 3 × 3** con un stride de 1 y padding de 1, generando una salida de **128 × 128 × 32**.
> - Se aplica una función de activación no lineal, generalmente **ReLU**, para introducir no linealidad en el modelo.
>
> **Max pooling**
>
> - Se reduce la resolución aplicando **max pooling de 2 × 2** con stride 2, obteniendo un mapa de características de **64 × 64 × 32**.
>
> **Segunda convolución**
>
> - Se aplican **64 filtros de 3 × 3 × 32** con stride 1 y padding 1, obteniendo un nuevo conjunto de mapas de **64 × 64 × 64**.
> - Se aplica ReLU para conservar las características más significativas.
>
> **Max pooling**
>
> - Se aplica nuevamente max pooling de **2 × 2**, reduciendo la resolución a **32 × 32 × 64**.
>
> **Tercera convolución**
>
> - Se utilizan **128 filtros de 3 × 3 × 64**, generando una salida de **32 × 32 × 128**.
> - Se aplica ReLU para mantener la no linealidad.
>
> **Flattening y conexión con la red completamente conectada**
>
> - Los mapas de características de **32 × 32 × 128** se "aplanan" en un vector de **131.072 valores**.
> - Se conecta con una capa **densa de 512 neuronas** con activación ReLU.
> - Se añade una capa de salida con activación **softmax** para producir una probabilidad asociada a cada clase.
>
> En esta arquitectura, las capas convolucionales extraen información estructurada de la imagen, el pooling reduce la dimensionalidad manteniendo los patrones clave, y la red densa final se encarga de tomar una decisión en base a las características aprendidas.
>

Esta combinación de operaciones permite a las redes convolucionales no solo reconocer imágenes, sino generalizar a nuevos ejemplos con gran precisión.

> [!important]
>
> **Por qué los mapas de características tienen más profundidad pero menor dimensión espacial**
>
> A medida que una imagen atraviesa las capas de una red convolucional, su representación cambia de manera significativa. En particular, los mapas de características que emergen tras cada convolución tienden a **ganar profundidad** mientras que su **dimensión espacial se reduce**. Este fenómeno no es un efecto colateral, sino una propiedad clave del diseño de las CNN que permite extraer información relevante de forma eficiente y optimizada para tareas como la clasificación.
>
> El aumento en la profundidad se debe a la aplicación de múltiples filtros en cada capa convolucional. Cada filtro opera sobre todos los canales de la imagen de entrada y responde a patrones específicos, generando un mapa de características independiente. Si la imagen de entrada posee tres canales (correspondientes a los valores de color en RGB) y se aplican treinta y dos filtros, la salida de esa capa será un conjunto de **treinta y dos mapas de características**, cada uno resaltando distintas estructuras dentro de la imagen original. Esta creciente profundidad permite que la red descomponga la imagen en representaciones cada vez más abstractas, capturando desde bordes y texturas en las primeras capas hasta patrones más complejos en niveles más profundos.
>
> Por otro lado, la reducción en la resolución espacial se produce debido a la manera en que la convolución opera sobre la imagen. Cada filtro actúa sobre una región local de la imagen en lugar de tratar cada píxel de forma individual, lo que disminuye el tamaño del mapa de características resultante. Además, la utilización de **stride** introduce un desplazamiento mayor en la aplicación del filtro, lo que acentúa esta reducción. A medida que la red profundiza, la incorporación de **operaciones de pooling** refuerza aún más este efecto, seleccionando únicamente los valores más representativos dentro de cada región procesada.
>
> Este comportamiento es fundamental para el funcionamiento de las redes convolucionales. El incremento en la profundidad permite que la red capture características más abstractas y representaciones de alto nivel, mientras que la reducción en el tamaño espacial optimiza el procesamiento y disminuye la redundancia en la información. En última instancia, esta transformación facilita la conexión con las capas completamente conectadas que se encargan de la clasificación, permitiendo que el modelo tome decisiones sobre una representación compacta pero altamente informativa de la imagen original.

#### **Ventajas de la arquitectura convolucional frente a la tradicional MLP**

Las redes neuronales convolucionales (CNN) han demostrado ser significativamente más eficaces que las redes neuronales completamente conectadas (MLP) cuando se trata de procesar imágenes. La diferencia fundamental entre ambas arquitecturas radica en la manera en que manejan la información espacial y en la eficiencia con la que aprenden patrones relevantes. Mientras que una MLP trata cada píxel como una entrada independiente, ignorando la estructura de la imagen, una CNN aprovecha las relaciones espaciales entre los píxeles, lo que le otorga múltiples ventajas en términos de eficiencia computacional, generalización y extracción de características.

Uno de los aspectos más evidentes en los que la arquitectura convolucional supera a la MLP es la **reducción en el número de parámetros**. En una red completamente conectada, cada neurona de una capa está enlazada con todas las neuronas de la capa anterior, lo que significa que una imagen de tamaño moderado genera una cantidad masiva de pesos que deben aprenderse. Si se tomara una imagen de entrada de $128 \times 128$ píxeles con tres canales de color, la primera capa de una MLP requeriría aproximadamente $49.152$ pesos por cada neurona en la capa oculta, sin contar los sesgos. En contraste, una CNN procesa la imagen mediante filtros que comparten parámetros a lo largo de toda la imagen, reduciendo drásticamente la cantidad de pesos ajustables y permitiendo que el modelo sea más eficiente sin perder capacidad de representación.

Además de reducir la cantidad de parámetros, las redes convolucionales conservan la **estructura espacial de la imagen**. En una MLP, la imagen de entrada se aplana en un vector, eliminando por completo la relación entre píxeles vecinos. Como resultado, la red pierde información sobre patrones locales como bordes, texturas o formas. En cambio, una CNN mantiene la organización bidimensional de la imagen en cada capa convolucional, lo que le permite detectar características significativas en distintos niveles de abstracción. Las primeras capas suelen capturar detalles básicos como líneas y contornos, mientras que las capas más profundas identifican estructuras más complejas, como partes de objetos y combinaciones de texturas.

Otro beneficio clave es la **invarianza a traslaciones**, que permite que una CNN reconozca un objeto sin importar su posición exacta dentro de la imagen. En una MLP, si un objeto aparece en una ubicación distinta de la imagen respecto a los datos de entrenamiento, la red no tiene forma de reconocerlo sin haber visto ejemplos específicos en esa misma posición. En cambio, debido a que los filtros convolucionales se aplican sobre toda la imagen de manera uniforme, una CNN puede identificar características independientemente de su ubicación, mejorando así su capacidad de generalización.

La **robustez ante variaciones** en la entrada es otra ventaja significativa. Mientras que una MLP es altamente sensible a cambios pequeños en los valores de los píxeles, una CNN puede lidiar mejor con variaciones en la escala, orientación e iluminación de los objetos. Esto se debe a la forma en que combina múltiples filtros y aplica operaciones como el pooling, que ayuda a preservar información clave mientras reduce la resolución espacial de los mapas de características. Gracias a esta propiedad, las CNN son más resistentes a imágenes ruidosas y pueden reconocer patrones con mayor consistencia.

Desde una perspectiva computacional, la arquitectura convolucional es mucho más eficiente en términos de memoria y procesamiento. Mientras que una MLP requiere una gran cantidad de conexiones que deben ser almacenadas y calculadas en cada paso, una CNN reutiliza sus filtros en toda la imagen, permitiendo que el cálculo sea distribuido de manera óptima en unidades de procesamiento paralelas, como las GPU. Esto hace posible entrenar redes profundas con millones de parámetros en tiempos razonables, algo impracticable con una MLP de tamaño equivalente.

En última instancia, la combinación de eficiencia en el uso de parámetros, preservación de la información espacial, invarianza a traslaciones y robustez ante variaciones hace que las redes convolucionales sean la arquitectura más adecuada para tareas de visión por computadora. Mientras que las MLP pueden ser útiles en problemas con entradas de baja dimensionalidad y poca estructura espacial, las CNN han demostrado ser la solución más efectiva para el reconocimiento de patrones visuales complejos, siendo la base de modelos modernos en clasificación de imágenes, detección de objetos y segmentación semántica.

#### **Arquitectura LeNet: origen, diseño e impacto en la visión por computadora**

A finales de la década de 1980, el procesamiento de imágenes mediante redes neuronales se encontraba limitado por el alto costo computacional de los modelos tradicionales. Las redes completamente conectadas (MLP) requerían un número prohibitivo de parámetros al trabajar con imágenes de tamaño moderado, lo que dificultaba su uso en aplicaciones prácticas. En este contexto, Yann LeCun y sus colaboradores desarrollaron **LeNet-5**, una arquitectura diseñada específicamente para el reconocimiento de patrones visuales mediante convoluciones, reduciendo significativamente la cantidad de parámetros entrenables y permitiendo el procesamiento eficiente de imágenes.

El diseño de LeNet-5 representó un cambio de paradigma en la inteligencia artificial aplicada a visión por computadora. Su estructura introdujo el concepto de **capas convolucionales intercaladas con capas de pooling**, estableciendo un modelo jerárquico donde las características simples se combinaban progresivamente en representaciones más complejas. Esta organización permitió el desarrollo de redes profundas capaces de detectar estructuras visuales sin la necesidad de una ingeniería manual de características, lo que facilitó la generalización del modelo a distintos conjuntos de datos. 

> [!note]
>
> **Yann LeCun: pionero de las redes neuronales convolucionales**
>
> Yann LeCun es un científico de la computación y especialista en inteligencia artificial reconocido por sus contribuciones al aprendizaje profundo y, en particular, al desarrollo de las **redes neuronales convolucionales (CNN)**. Nacido en **1960 en Francia**, se formó en ingeniería y computación en la Université Pierre et Marie Curie y obtuvo su doctorado en 1987 bajo la supervisión de Maurice Milgram, donde comenzó a trabajar en redes neuronales y algoritmos de aprendizaje.
>
> A finales de la década de 1980, LeCun desarrolló **LeNet-5**, una de las primeras arquitecturas convolucionales, diseñadas para el reconocimiento de caracteres escritos a mano. Su trabajo fue adoptado por empresas como AT&T y posteriormente por sistemas bancarios para la lectura automática de cheques. Durante su etapa en **Bell Labs**, también exploró la optimización de algoritmos de backpropagation y aprendizaje supervisado.
>
> Desde 2013, ha liderado la investigación en inteligencia artificial en **Meta (Facebook AI Research)** y es profesor en la Universidad de Nueva York. En 2018, recibió el **Premio Turing**, junto con Geoffrey Hinton y Yoshua Bengio, por sus contribuciones al aprendizaje profundo. Su trabajo ha sido clave en la evolución de la visión por computadora, el procesamiento de imágenes y el avance de la inteligencia artificial moderna.

La arquitectura de LeNet-5 sigue un esquema modular que integra convoluciones, reducción de dimensionalidad y capas completamente conectadas para la clasificación. La imagen de entrada, típicamente de **28 × 28 píxeles en escala de grises**, es procesada por una primera capa convolucional con **seis filtros de 5 × 5**, generando mapas de características de **24 × 24 × 6**. La reducción en la resolución se debe a la aplicación del kernel sin padding, lo que permite que el modelo detecte bordes y texturas en las primeras etapas.

Tras esta primera extracción de características, la red incorpora una capa de **submuestreo (pooling)** que reduce la resolución de cada mapa de características a **12 × 12 × 6**. Esta operación se realiza mediante una estrategia de **promediado**, que en lugar de tomar el valor máximo en cada región, calcula el promedio, lo que suaviza las respuestas del modelo y reduce la sensibilidad a pequeñas variaciones en la imagen.

El siguiente bloque de convolución aplica **dieciséis filtros de 5 × 5** sobre la salida del pooling anterior, generando un conjunto de mapas de características de **8 × 8 × 16**. Como en la capa anterior, se aplica nuevamente una operación de **submuestreo**, reduciendo la salida a **4 × 4 × 16**, compactando la información relevante mientras se mantiene la estructura jerárquica de la imagen.

Esta segunda convolución permite que la red extraiga representaciones más abstractas de la imagen original. Mientras que la primera capa detecta bordes y texturas simples, la segunda capa aprende combinaciones más sofisticadas, como esquinas, formas curvas o estructuras más amplias. Recuerda que el hecho de que la profundidad aumente mientras la dimensión espacial disminuye es una característica esencial de las redes convolucionales modernas que permite una representación compacta de la imagen mientras se mantiene una creciente cantidad de información útil para la clasificación final.

Una vez extraídas las representaciones de alto nivel, la arquitectura de LeNet-5 convierte los mapas de características en un vector unidimensional mediante una operación de **aplanado (flattening)**. Este vector es alimentado a una red neuronal completamente conectada con **120 neuronas**, seguida de una segunda capa densa con **84 unidades** y una capa de salida con activación **softmax**, encargada de asignar probabilidades a cada una de las clases del problema de clasificación. 

<img src=".\assets\1lvvWF48t7cyRWqct13eU0w.jpeg" alt="LeNet-5" />

El impacto de LeNet-5 en la inteligencia artificial y la visión por computadora ha sido profundo. Su diseño introdujo principios fundamentales que han sido adoptados por modelos más avanzados, como el uso de **pesos compartidos en las convoluciones**, la reducción progresiva de la dimensionalidad mediante pooling y la combinación de representaciones espaciales con redes completamente conectadas para la clasificación final.

Uno de los primeros usos prácticos de LeNet-5 fue en el reconocimiento automático de caracteres escritos a mano, aplicado a la lectura de cheques bancarios en sistemas comerciales. Gracias a su capacidad para identificar patrones con alta precisión y eficiencia computacional, la arquitectura se convirtió en la base de múltiples aplicaciones posteriores en reconocimiento de texto, detección de objetos y análisis de imágenes biomédicas.

A pesar de su antigüedad, la arquitectura de LeNet-5 sigue siendo utilizada en la actualidad como un punto de partida en la enseñanza de redes convolucionales y en experimentos de visión artificial donde se requiere una arquitectura ligera. Su diseño modular y eficiente ha demostrado que las redes convolucionales pueden ser escalables y adaptarse a problemas más complejos, sentando las bases para desarrollos posteriores como AlexNet, VGG y ResNet.

### **Arquitecturas AlexNet y VGGNet: impacto en las redes convolucionales**

##### Arquitectura AlexNet

El desarrollo de **AlexNet** en 2012 marcó un punto de inflexión en el avance de las redes convolucionales modernas. Diseñada por **Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton**, esta arquitectura revolucionó el campo de la visión por computadora al ganar el desafío **ImageNet Large Scale Visual Recognition Challenge (ILSVRC-2012)** con una precisión significativamente superior a la de los modelos previos. Su éxito validó la utilidad del **aprendizaje profundo** en tareas de reconocimiento de imágenes y consolidó el uso de redes convolucionales como el estándar en clasificación visual.

> [!note]
>
> El **ILSVRC (ImageNet Large Scale Visual Recognition Challenge)** fue una competición anual de visión por computadora organizada entre **2010 y 2017** basada en el conjunto de datos **ImageNet**, que contiene millones de imágenes etiquetadas en **1.000 categorías**. Su objetivo era evaluar y comparar algoritmos de reconocimiento de imágenes en tareas como **clasificación, detección y segmentación de objetos**.Este desafío se convirtió en un punto de referencia para el avance del **aprendizaje profundo**, ya que permitió demostrar el impacto de las **redes convolucionales (CNN)**. Modelos como **AlexNet (2012), GoogleNet (2014) y ResNet (2015)** lograron reducciones significativas en la tasa de error, estableciendo las bases para el desarrollo de arquitecturas modernas en visión por computadora.

En términos estructurales, **AlexNet** mantiene los principios fundamentales introducidos por **LeNet-5**, pero introduce modificaciones cruciales que mejoran su rendimiento. La red trabaja con imágenes de **224 × 224 × 3**, lo que representa una diferencia significativa con respecto a las arquitecturas anteriores, diseñadas para imágenes de menor resolución. Para manejar esta escala, la red incorpora **más capas convolucionales** y un mayor número de filtros, lo que le permite extraer características más complejas de los datos de entrada.

Una de sus principales innovaciones es la introducción de **ReLU (Rectified Linear Unit)** como función de activación en lugar de **sigmoide o tanh**, lo que acelera considerablemente el entrenamiento y mitiga el problema del **desvanecimiento del gradiente**. Además, AlexNet incorpora **dropout** en las capas completamente conectadas para reducir el sobreajuste, una técnica que se volvió esencial en redes profundas.

La arquitectura de AlexNet sigue un diseño en **ocho capas**, con cinco capas convolucionales seguidas de tres capas completamente conectadas. La primera convolución aplica **96 filtros de 11 × 11** con un stride de 4, lo que reduce drásticamente la dimensionalidad en la primera etapa. Posteriormente, se aplican convoluciones con **kernels más pequeños** (de 5 × 5 y 3 × 3), lo que permite refinar la extracción de características antes de conectar la representación con las capas densas finales.

Otro aspecto clave de AlexNet es su uso de **normalización por lotes (Local Response Normalization, LRN)**, un mecanismo que regula las activaciones dentro de la red para estabilizar el aprendizaje. Si bien esta técnica no se mantuvo en arquitecturas posteriores, en su momento resultó fundamental para mejorar la convergencia del modelo.

El impacto de AlexNet en la comunidad de visión por computadora fue inmediato. Su desempeño en **ImageNet** demostró que redes más profundas y con mayor capacidad de cómputo podían superar ampliamente los enfoques tradicionales en clasificación de imágenes. Esto abrió la puerta a arquitecturas aún más complejas, como **VGGNet**, que refinaron la estructura propuesta por AlexNet.

##### **Arquitectura VGGNet: refinamiento de la convolución profunda**

Desarrollada por el **Visual Geometry Group (VGG) de la Universidad de Oxford**, **VGGNet** introdujo una mejora clave en la arquitectura convolucional al simplificar el diseño de las redes profundas y optimizar la extracción jerárquica de características. Presentada en 2014, VGGNet se destacó en el **desafío ImageNet**, logrando un rendimiento sobresaliente con una estructura más ordenada y eficiente. 

> [!note]
>
> **ImageNet** es una base de datos masiva de imágenes etiquetadas, diseñada para entrenar y evaluar algoritmos de visión por computadora. Contiene **más de 14 millones de imágenes**, organizadas en **más de 20.000 categorías**, con una subdivisión estándar de **1.000 clases** utilizada en el desafío **ILSVRC**. Fue creada en **2009** por **Fei-Fei Li y su equipo en la Universidad de Stanford**, con el propósito de proporcionar un conjunto de datos extenso y diverso que permitiera a los modelos de inteligencia artificial aprender representaciones ricas y generalizables. Su impacto ha sido fundamental en el desarrollo del **aprendizaje profundo**, ya que permitió demostrar el poder de las redes convolucionales (CNN), impulsando avances significativos en tareas de **clasificación, detección y segmentación de objetos**.

A diferencia de AlexNet, que combinaba filtros de distintos tamaños en diferentes capas, VGGNet se basa en una arquitectura homogénea en la que todas las convoluciones utilizan **kernels de 3 × 3 con stride 1** y **padding de 1**. Este enfoque permite que la red capture patrones más complejos mediante la acumulación de múltiples capas convolucionales en lugar de depender de kernels grandes en las primeras etapas.

El diseño de VGGNet introduce el concepto de **profundidad creciente**, con modelos que van desde **VGG-11 hasta VGG-19**, dependiendo del número de capas convolucionales. En su versión más utilizada, **VGG-16**, la red consta de **13 capas convolucionales** seguidas de **tres capas completamente conectadas**, con pooling intercalado para reducir la dimensionalidad. Esta mayor profundidad permite que la red aprenda representaciones más ricas y abstractas en comparación con arquitecturas anteriores.

El uso de **múltiples capas de convolución pequeñas en lugar de una única capa con filtros más grandes** aporta ventajas clave. Por un lado, permite modelar características más detalladas y mejora la capacidad de generalización. Por otro lado, el menor número de parámetros por filtro facilita el entrenamiento y reduce la carga computacional en comparación con AlexNet.

VGGNet mantiene **ReLU** como función de activación y utiliza **max pooling de 2 × 2** con stride de 2 para reducir la resolución de los mapas de características. Sin embargo, una de sus desventajas es su alto consumo de memoria y tiempo de cómputo, ya que su profundidad incrementa la cantidad de operaciones requeridas para el entrenamiento.

A pesar de su elevado costo computacional, VGGNet tuvo un impacto significativo en la evolución de las CNN, ya que estableció un diseño modular que inspiró arquitecturas posteriores. Su estructura homogénea y su capacidad para modelar características complejas sentaron las bases para modelos más eficientes como **ResNet**, que introdujeron mecanismos avanzados de aprendizaje profundo. 

##### **Contribución de AlexNet y VGGNet a la evolución de las CNN**

Tanto **AlexNet como VGGNet** marcaron hitos en el desarrollo de redes convolucionales modernas. AlexNet demostró que redes profundas podían superar ampliamente los enfoques tradicionales, incorporando innovaciones como **ReLU, dropout y normalización por lotes** para mejorar la estabilidad y eficiencia del entrenamiento. VGGNet, por su parte, refinó la estructura convolucional con una **arquitectura homogénea y más profunda**, demostrando que múltiples capas pequeñas eran más efectivas que pocas capas con filtros grandes.

Ambos modelos allanaron el camino para arquitecturas posteriores como **ResNet**, que abordaron problemas como la degradación del gradiente en redes muy profundas, permitiendo la construcción de modelos aún más escalables y eficientes. La evolución de estas arquitecturas permitió que las CNN alcanzaran niveles de rendimiento comparables al reconocimiento humano en tareas de visión por computadora, consolidando el aprendizaje profundo como la base de la inteligencia artificial moderna.

#### Arquitecturas GoogleNet y ResNet

##### **Arquitectura GoogleNet: la introducción de módulos Inception**

La arquitectura **GoogleNet**, presentada por el equipo de Google en 2014, representó un cambio de paradigma en el diseño de redes convolucionales al introducir el concepto de **módulos Inception**. Este enfoque buscaba mejorar la eficiencia y profundidad de las redes sin incrementar de manera descontrolada el número de parámetros, un problema que afectaba a modelos previos como **AlexNet y VGGNet**.

GoogleNet fue el modelo ganador del desafío **ILSVRC-2014**, destacándose por alcanzar una precisión superior con un número de parámetros significativamente menor. Mientras que **VGG-16** tenía alrededor de **138 millones de parámetros**, GoogleNet logró superar su rendimiento con apenas **4 millones**, gracias a su diseño optimizado.

La clave de esta arquitectura radica en la utilización de **módulos Inception**, que permiten a la red capturar simultáneamente características a diferentes escalas sin aumentar desmesuradamente la complejidad computacional. A diferencia de las redes convencionales, donde cada capa aplica un único tipo de filtro a toda la imagen, los módulos Inception combinan convoluciones de **1 × 1, 3 × 3 y 5 × 5** en paralelo, junto con una operación de **pooling**. De esta manera, cada nivel de la red aprende múltiples representaciones de la imagen en distintos niveles de abstracción, mejorando la capacidad del modelo para detectar patrones complejos.

> [!note]
>
> El nombre **Inception** de los módulos utilizados en GoogleNet hace referencia indirecta a la película *Inception* (2010) de Christopher Nolan. En la película, el concepto central gira en torno a la idea de sueños dentro de sueños, lo que refleja la naturaleza jerárquica y anidada de los módulos Inception en la red neuronal.
>
> En el contexto de las CNN, los **módulos Inception** permiten que una misma capa procese la imagen a diferentes escalas simultáneamente, aplicando múltiples filtros de distintos tamaños en paralelo (por ejemplo, **1 × 1, 3 × 3 y 5 × 5**). Esto crea una estructura donde se capturan múltiples niveles de detalle en una sola capa, análoga a la idea de *niveles dentro de niveles* en la película.
>
> Además, los desarrolladores del modelo jugaron con esta referencia en la implementación, ya que en el código original de GoogleNet se incluía la línea de comentario:
>
> > **"We need to go deeper"**
>
> Esta frase es una cita directa de la película y hace alusión a la capacidad de las redes profundas para extraer representaciones de alto nivel en cada capa.

Otra innovación crucial de GoogleNet es el uso extensivo de **convoluciones de 1 × 1**, que funcionan como mecanismos de reducción de dimensionalidad. Antes de aplicar convoluciones más grandes, la red usa estos filtros más pequeños para disminuir la cantidad de mapas de características, lo que reduce la carga computacional y permite profundizar el modelo sin que el costo en memoria se vuelva prohibitivo.

Además, GoogleNet introduce **cabezas auxiliares**, que son pequeñas redes de clasificación insertadas en capas intermedias. Estas cabezas ayudan a mejorar el flujo del gradiente durante el entrenamiento, mitigando problemas de **desvanecimiento del gradiente** en redes profundas. Al proporcionar señales de aprendizaje adicionales, estas estructuras permiten que las capas más tempranas en la red se entrenen de manera más efectiva.

La arquitectura completa de GoogleNet cuenta con **22 capas profundas**, organizadas de manera que la red pueda extraer características con una gran eficiencia computacional. Su éxito demostró que **profundizar una red no siempre significa aumentar exponencialmente la cantidad de parámetros**, y que un diseño modular inteligente puede mejorar el rendimiento sin comprometer la viabilidad del entrenamiento.

##### **Arquitectura ResNet: solucionando el problema del gradiente en redes profundas**

Si bien modelos como **GoogleNet** lograron profundizar las redes sin aumentar drásticamente los parámetros, todavía existía un problema fundamental: **a medida que las redes se volvían más profundas, su entrenamiento se volvía más difícil**. Redes con más de **30 o 40 capas** comenzaban a sufrir problemas de **degradación del gradiente**, lo que impedía que capas más profundas aprendieran de manera efectiva.

Para abordar este problema, **Microsoft Research** presentó en 2015 la **Red Residual (ResNet)**, una arquitectura que introdujo el concepto de **conexiones residuales**. Este modelo ganó el desafío **ILSVRC-2015**, superando a GoogleNet y estableciendo un nuevo estándar en el diseño de redes profundas.

La idea central de ResNet es que en lugar de aprender una transformación directa entre la entrada y la salida de una capa, la red aprende una **diferencia residual**. Esto se implementa mediante **atajos o conexiones directas** que permiten que la información fluya sin modificaciones entre ciertas capas.

Matemáticamente, en una red tradicional, una capa aprende una transformación $H(x)$ sobre la entrada $x$. ResNet, en cambio, aprende una función residual:

$$
F(x) = H(x) - x
$$

Por lo tanto, la salida final se expresa como:

$$
H(x) = F(x) + x
$$

Esta reformulación tiene una ventaja clave: si las capas profundas no logran aprender una transformación útil, la red puede simplemente aprender la identidad, lo que permite que los gradientes fluyan de manera más efectiva sin desvanecerse. Gracias a esta estructura, **ResNet permitió la construcción de redes de cientos de capas sin sufrir problemas de degradación del gradiente**.

ResNet se presentó en varias versiones, desde **ResNet-18** hasta **ResNet-152**, donde el número indica la cantidad de capas. Su implementación estándar reemplazó las capas convolucionales tradicionales por **bloques residuales**, compuestos por combinaciones de convoluciones **3 × 3** y conexiones de salto.

Otra mejora clave de ResNet es que mantiene un **uso eficiente de los parámetros**. En lugar de incrementar masivamente el número de filtros en cada capa, como en VGGNet, ResNet optimiza la profundidad mediante la reutilización de información. Esta característica ha hecho que las redes residuales sean la base de modelos aún más avanzados, como **ResNeXt** o **DenseNet**, que refinan aún más el flujo de información dentro de redes profundas. 

##### **Impacto de GoogleNet y ResNet en la evolución de las redes convolucionales**

Tanto **GoogleNet** como **ResNet** introdujeron avances fundamentales que transformaron el diseño de redes neuronales convolucionales.

GoogleNet demostró que el uso de **múltiples tamaños de filtro en paralelo** permitía capturar características más ricas sin necesidad de incrementar la cantidad de parámetros. Su enfoque basado en **módulos Inception** estableció una arquitectura eficiente y escalable, que se convirtió en la base de modelos más recientes como **Inception-v3 y v4**.

Por otro lado, ResNet resolvió el problema de **entrenar redes extremadamente profundas** gracias a su mecanismo de **conexiones residuales**, permitiendo que modelos de más de **100 capas** fueran entrenables de manera efectiva. Su diseño se convirtió en un estándar para el desarrollo de arquitecturas profundas, siendo utilizado en diversas aplicaciones más allá de la visión por computadora, como el procesamiento de lenguaje natural y el reconocimiento de voz.

Ambas arquitecturas marcaron un antes y un después en la inteligencia artificial, consolidando el uso de redes convolucionales profundas y abriendo el camino a modelos más avanzados que hoy dominan el estado del arte en visión artificial.
