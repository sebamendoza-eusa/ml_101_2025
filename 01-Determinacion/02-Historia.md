# 2. **Breve recorrido histórico**

El **aprendizaje automático** puede que sea la rama de la inteligencia artificial que haya tenido un desarrollo más significativo desde sus inicios en los años 50 hasta la actualidad. Su historia está marcada por hitos clave en el desarrollo de modelos, algoritmos, y avances tecnológicos que han permitido la creación de sistemas cada vez más avanzados.

## Orígenes: La conferencia de Dartmouth, Frank Rosenblatt y el Perceptrón (1956-1957)

La **Conferencia de Dartmouth** (1956) es ampliamente considerada como el **punto de partida formal de la inteligencia artificial (IA)**. Organizada por **John McCarthy**, **Marvin Minsky**, **Claude Shannon** y **Nathaniel Rochester**, en esta conferencia se propuso investigar la idea de que las máquinas podían simular la inteligencia humana. Fue aquí donde se acuñó el término "inteligencia artificial".

Esta conferencia está profundamente ligada a la historia del **machine learning** porque sentó las bases del campo, plantando preguntas clave sobre cómo las máquinas podrían aprender. Aunque en aquel momento la IA se centraba en enfoques simbólicos basados en la manipulación de reglas y símbolos, los conceptos presentados en Dartmouth llevaron, indirectamente, a la evolución de enfoques más **datos-céntricos** como el **machine learning**.

En las décadas siguientes, el machine learning iría adquiriendo importancia a medida que surgieron enfoques que se apartaban de la programación explícita de reglas, acercándose más a la idea de que las máquinas podían aprender de la experiencia.

> [!tip]
> 
> Un dato curioso sobre la **Conferencia de Dartmouth** es que, a pesar de su gran impacto en la historia de la inteligencia artificial, los resultados inmediatos fueron bastante limitados. La reunión duró seis semanas y reunió a algunos de los grandes pioneros de la IA, pero muchos de los avances que hoy conocemos no se lograron en ese momento. De hecho, muchos de los asistentes subestimaron la dificultad real de crear máquinas inteligentes, creyendo que el progreso sería mucho más rápido. Lo que empezó como un proyecto optimista terminó siendo un camino mucho más largo y complejo.
> 
> Otra curiosidad es que solo cuatro científicos asistieron de manera continua: **John McCarthy**, **Marvin Minsky**, **Nathaniel Rochester**, y **Claude Shannon**. El resto de los investigadores que habían sido invitados no participaron activamente en las semanas que duró la conferencia.

A partir de aquí, uno de los primeros hitos importantes en la historia del aprendizaje automático fue el desarrollo del **Perceptrón**, creado por **Frank Rosenblatt** en 1957. Inspirado en las neuronas biológicas, el perceptrón fue el primer modelo computacional que intentó emular la capacidad del cerebro para procesar información y tomar decisiones. Su capacidad para realizar **tareas de clasificación** a partir de entradas simples fue revolucionaria en su tiempo, sentando las bases para lo que hoy conocemos como redes neuronales. El perceptrón, aunque limitado (ya que solo podía resolver problemas linealmente separables), fue la primera aproximación concreta al concepto de aprendizaje a partir de datos, lo que posteriormente derivó en modelos más complejos.

> [!important]
> 
> Frank Rosenblatt (1928-1971) fue un psicólogo estadounidense y pionero en el campo de la inteligencia artificial. Desarrolló el **Perceptrón** en 1957, uno de los primeros modelos de redes neuronales inspirado en las neuronas biológicas. Su trabajo sentó las bases para las redes neuronales modernas, aunque inicialmente fue criticado por sus limitaciones. Rosenblatt fue un visionario en el estudio de sistemas que aprenden a partir de datos. 
> 
> Puedes ampliar información en este enlace: [Frank Rosenblatt - Wikipedia](https://en.wikipedia.org/wiki/Frank_Rosenblatt).

El trabajo de Rosenblatt estaba en línea con los esfuerzos de otros pioneros de la IA como **John McCarthy**, quien ya había acuñado el término "inteligencia artificial" en 1956, o **Alan Turing**, que desde los años 30 ya había desarrollado su concepto de "máquina de Turing" y publicado en 1950 su famoso artículo ***Computing Machinery and Intelligence***. En él plantea lo que se terminó conociendo como el "Test de Turing". Mientras McCarthy y otros trabajaban en **enfoques simbólicos** de la IA, el trabajo de Rosenblatt con el perceptrón se enfocaba más en un **enfoque conexionista**, que más tarde fue retomado por las redes neuronales.

> [!Note]
>
> El **enfoque conexionista** es un paradigma en inteligencia artificial que se inspira en el funcionamiento del cerebro humano, utilizando redes neuronales artificiales para procesar y aprender de datos. A través de interconexiones de neuronas artificiales, este enfoque busca detectar patrones complejos mediante el aprendizaje, adaptándose a tareas como la clasificación y predicción.
>
> Aunque basado en el concepto de neurona, su alcance filosófico y matemático es más amplio que el simple uso de **Redes Neuronales Artificiales**. El principio central del conexionismo se basa en la idea de que el **conocimiento se almacena de manera distribuida** y **emerge del ajuste de múltiples interacciones simples**. Cualquier modelo de ML que haga una **representación paramétrica del conocimiento, aprenda ajustando sus conexiones internas y ejecute un procesamiento distribuido**, podrá considerarse que está trabajando bajo un enfoque conexionista.
>
> Así pues, en una representación paramétrica, el conocimiento (o la función predictiva del modelo) no reside en reglas explícitas (como en el enfoque simbólico de la IA), sino en los **parámetros** (pesos o coeficientes) del modelo. Por su parte, el aprendizaje por ajuste de parámetros es aquel en el que, mediante un proceso de entrenamiento, se realiza una **modificación continua y gradual de estos parámetros** basada en la experiencia y buscando optimizar una medida de rendimiento. Por último, el procesamiento distribuido permite construir la decisión o predicción final como resultado de la **combinación simultánea y ponderada de múltiples *características* (o variables)** de entrada. Cada *característica* tiene una "fuerza de conexión" (un peso) que contribuye a la salida.
>
> Incluso modelos de *Machine Learning* clásicos, como la Regresión Lineal, encajan en esta definición al utilizar **pesos ajustables** que actúan como **fuerzas de conexión sinápticas** para determinar una salida. La diferencia con las redes neuronales no es de naturaleza filosófica, sino de **arquitectura y complejidad computacional**. 
>
> El conexionismo, en esencia, es un concepto que abarca cualquier sistema de *Machine Learning* cuya capacidad de procesamiento y almacenamiento de conocimiento es **subsimbólica** y distribuida.

> [!important]
> 
> Alan Turing (1912-1954) fue un matemático británico y pionero en la informática. Es famoso por su trabajo en la **máquina de Turing**, un concepto fundamental en la teoría de la computación, y por su contribución al descifrado del código Enigma durante la Segunda Guerra Mundial. Turing también es conocido por proponer el **Test de Turing** para determinar la inteligencia de las máquinas. Su legado ha dejado una profunda huella en la inteligencia artificial.
> 
> Puedes ampliar información en: [Alan Turing - Wikipedia](https://en.wikipedia.org/wiki/Alan_Turing).

> [!important]
> 
> John McCarthy (1927-2011) fue un destacado informático estadounidense y uno de los padres fundadores de la inteligencia artificial (IA). En 1956, acuñó el término **inteligencia artificial** y fue pionero en el desarrollo de lenguajes de programación como **LISP**, esencial para la investigación en IA. Su visión ayudó a establecer la IA como un campo académico independiente y fue un defensor de la creación de máquinas capaces de realizar tareas humanas.
> 
> Para más información, visita: [John McCarthy - Wikipedia](https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)).

## El "Invierno de la IA" y el Redescubrimiento de las Redes Neuronales (Años 70-80)

Tras el entusiasmo inicial con el perceptrón, surgieron críticas importantes, en particular la que fue formulada en 1969 por parte de **Marvin Minsky** y **Seymour Papert** en su libro *Perceptrons*. Ellos demostraron matemáticamente que los perceptrones no podían resolver problemas no lineales (como el problema del XOR), lo que llevó a una disminución temporal del interés en el aprendizaje automático. Este periodo de estancamiento en la investigación se conoce como el **"Invierno de la IA"**.

> [!important]
> 
> Marvin Minsky (1927-2016) y Seymour Papert (1928-2016) fueron pioneros en inteligencia artificial y educación. Minsky cofundó el **Laboratorio de IA del MIT**, donde investigó sobre redes neuronales, robótica y teoría de la mente. Papert, por su parte, fue un educador visionario, creador del lenguaje de programación **Logo**, y promotor de la educación a través de la tecnología. Ambos fueron fundamentales en la evolución de la inteligencia artificial y su aplicación educativa.
> 
> Puedes ampliar información aquí: 
> [Marvin Minsky - Wikipedia](https://en.wikipedia.org/wiki/Marvin_Minsky) 
> [Seymour Papert - Wikipedia](https://en.wikipedia.org/wiki/Seymour_Papert).

> [!tip]
> 
> El **problema XOR** no puede ser resuelto por un perceptrón simple porque la operación XOR no es **linealmente separable**. Un perceptrón solo puede separar datos con una línea recta, pero en la operación XOR, las posibles salidas (1 y 0) no pueden dividirse de esta manera. Es necesario una **red neuronal multicapa** (capas ocultas) y un mecanismo de retropropagación para resolverlo. Sólo con estas técnicas pueden modelarse **relaciones no lineales** complejas.

Sin embargo, en la década de 1980, con el surgimiento de la **retropropagación del error** (***backpropagation***), los investigadores encontraron una solución para entrenar **redes neuronales multicapa**. Este avance, propuesto por **Geoffrey Hinton**, **David Rumelhart**, y **Ronald Williams** en 1986, permitió a las redes neuronales resolver problemas no lineales y entrenar modelos más profundos. Con la retropropagación, las redes neuronales ya no estaban limitadas a problemas lineales, lo que allanó el camino para lo que hoy conocemos como **deep learning**.

Este resurgimiento de las redes neuronales coincidió con el desarrollo de otros campos de la IA. Mientras la IA simbólica se centraba en sistemas expertos y la lógica formal, el enfoque conexionista basado en redes neuronales empezó a ganar terreno. La rivalidad entre enfoques simbólicos y conexionistas dominó gran parte de las discusiones teóricas en la IA durante esa época.

> [!important]
> 
> Geoffrey Hinton es un científico británico y uno de los pioneros en **redes neuronales** y **deep learning**. Su trabajo sobre la **retropropagación** y las redes neuronales profundas ha sido fundamental para el desarrollo de la inteligencia artificial moderna. Es conocido por su investigación en **aprendizaje profundo** y por haber formado parte del equipo que revolucionó la IA en los últimos años. En 2018, recibió el **Premio Turing** junto con Yann LeCun y Yoshua Bengio.
> 
> Puedes ampliar información aquí: [Geoffrey Hinton - Wikipedia](https://en.wikipedia.org/wiki/Geoffrey_Hinton).



## Avances en Computación y el Boom del Machine Learning (2000-presente)

El **aprendizaje automático** experimentó un nuevo auge en el siglo XXI, principalmente debido a dos factores clave: el aumento en el poder de **computación** (gracias al uso de GPUs y, más recientemente, TPUs) y la **disponibilidad de grandes volúmenes de datos** (big data). Estos avances permitieron el entrenamiento de modelos más grandes y profundos, conocidos como **redes neuronales profundas**, agrupados en una técnica a la que se ha denominado **deep learning**. Estos modelos son capaces de realizar tareas que antes eran impensables, como el **reconocimiento de imágenes**, la **traducción automática**, o la **conducción autónoma**.

**Geoffrey Hinton**, junto con **Yann LeCun** y **Yoshua Bengio**, son considerados los padres del **deep learning**. Su trabajo en redes neuronales convolucionales (CNN) y modelos de redes neuronales recurrentes (RNN) permitió grandes avances en áreas como la visión por computadora y el procesamiento del lenguaje natural.

> [!important]
> 
> Yann LeCun, conocido como uno de los padres del deep learning, es un científico de la computación francés y uno de los pioneros en el desarrollo de **redes neuronales convolucionales (CNN)**, fundamentales para la visión por computadora y el reconocimiento de imágenes. Su trabajo ha sido clave en la evolución del **deep learning** moderno. Actualmente es el jefe de IA en Meta (Facebook) y ha sido galardonado con el Premio Turing en 2018 junto a Geoffrey Hinton y Yoshua Bengio.
> 
> Puedes ampliar información aquí: [Yann LeCun - Wikipedia](https://en.wikipedia.org/wiki/Yann_LeCun).

## Conexión con la Historia de la IA

Sin duda el desarrollo del ML ha ido estrechamente ligado a los avances en la **inteligencia artificial** en general. Mientras el **machine learning** ha ido avanzando en el campo de los modelos conexionistas y basados en datos, otros enfoques de la IA, como la **IA simbólica** y los **modelos probabilísticos** (como las redes bayesianas), también han tenido su propio desarrollo paralelo. Por ejemplo, **John McCarthy** y **Marvin Minsky** siempre se han situado en el campo de la IA simbólica, basada en el procesamiento de símbolos y reglas, algo que predominó en los primeros años de la IA.

Por supuesto no podemos olvidar a **Alan Turing** y su trabajo sobre la computación, que también sentó las bases para que las máquinas pudieran realizar tareas que involucran lógica y razonamiento.

Por su parte, **Stuart Russell** y **Peter Norvig**, autores de uno de los textos más influyentes sobre IA, *Artificial Intelligence: A Modern Approach*, han contribuido a unificar diferentes enfoques en la IA, desde los modelos simbólicos hasta los conexionistas.

Hoy en día, las redes neuronales y el aprendizaje automático son fundamentales en el campo de la IA, impulsando avances en múltiples áreas. La integración de diferentes enfoques en la IA ha permitido desarrollar **sistemas híbridos** que combinan técnicas de machine learning con reglas simbólicas, ampliando las capacidades de los sistemas de IA.

**Para reflexionar...**

> **¿Qué impacto tuvo la retropropagación en el desarrollo de redes neuronales y cómo afectó a la historia del aprendizaje automático?**
> Reflexiona sobre cómo este avance resolvió problemas que el perceptrón no podía abordar y permitió el desarrollo de sistemas más complejos y capaces.

> **¿Cómo influyó el aumento en la capacidad computacional en el resurgimiento del machine learning en el siglo XXI?**
> Piensa en la conexión entre la disponibilidad de grandes cantidades de datos, el desarrollo de GPUs y TPUs, y la capacidad para entrenar redes neuronales profundas de manera eficiente.

> **¿Qué relación existe entre el desarrollo del machine learning y los enfoques simbólicos de la IA?**
> Considera cómo ambos enfoques han co-evolucionado y cómo en la actualidad se están integrando en sistemas más completos y flexibles.

> **¿Por qué el machine learning ha desplazado a otros enfoques en muchas aplicaciones modernas de la IA?**
> Reflexiona sobre las ventajas del aprendizaje a partir de datos frente a sistemas basados en reglas estáticas y cómo el big data ha sido un catalizador en esta transición.

> [!note]
> 
> La historia del **machine learning** es, en esencia, una historia de perseverancia, donde el avance en modelos matemáticos, la capacidad computacional y la disponibilidad de datos han permitido resolver problemas complejos y crear sistemas inteligentes capaces de aprender y adaptarse. Desde sus humildes comienzos con el perceptrón de Rosenblatt hasta los sofisticados modelos de deep learning actuales, el machine learning sigue siendo una de las áreas más dinámicas y prometedoras dentro de la inteligencia artificial.
