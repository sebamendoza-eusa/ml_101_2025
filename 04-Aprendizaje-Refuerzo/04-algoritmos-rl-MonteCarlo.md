# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Algoritmos de Aprendizaje por Refuerzo: Métodos de Monte Carlo

### Introducción

En capítulo anterior  hemos estudiado cómo resolver un problema de decisión secuencial mediante técnicas de programación dinámica. Estos métodos, como la iteración de políticas o la iteración de valores, parten de una hipótesis muy fuerte: **el conocimiento completo del entorno**, es decir, el acceso explícito a la función de transición $P(s' \mid s, a)$ y a la función de recompensa $R(s, a, s')$.

En muchos problemas del mundo real, esta suposición es irrealizable. El agente no conoce la dinámica del entorno ni cómo se generan las recompensas. Lo único que puede hacer es **interactuar con el entorno, registrar lo que ocurre, y aprender a partir de esa experiencia**. Es decir, el aprendizaje ya no se basa en planificación sobre un modelo, sino en **observación directa de episodios** completos: secuencias del tipo 

$$
(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

donde el agente actúa, observa las recompensas obtenidas y las consecuencias de sus acciones.

Este cambio de enfoque marca la entrada en una nueva clase de métodos de aprendizaje por refuerzo: los **métodos Monte Carlo**. Su característica fundamental es que el agente aprende a partir de **episodios simulados o reales**, sin necesidad de conocer ni estimar la estructura interna del entorno. En lugar de resolver las ecuaciones de Bellman directamente, lo que hacen es **aproximar el valor de un estado o una acción a partir del promedio de los retornos observados** tras múltiples ejecuciones.

La idea es sencilla: si un agente visita muchas veces un estado bajo una política dada, y registra los retornos totales obtenidos en cada uno de esos episodios, el promedio de esos retornos se puede usar como estimación del valor de dicho estado. Este enfoque hace innecesario conocer la probabilidad de transición o el modelo de recompensas: **basta con observar las consecuencias reales del comportamiento del agente**.

Sin embargo, esta simplicidad conceptual introduce también algunas limitaciones. Para que Monte Carlo funcione, es necesario que los episodios tengan un final claro (un estado terminal), ya que el retorno solo puede calcularse completamente si se conoce toda la secuencia de recompensas. Esto implica que Monte Carlo es especialmente útil en entornos **episódicos**, como juegos o simulaciones, pero puede presentar dificultades en tareas continuas o donde no se sabe cuándo terminará la interacción.

A pesar de ello, los métodos Monte Carlo representan un avance decisivo: permiten que un agente **aprenda directamente de la experiencia**, sin necesidad de acceder a estructuras ocultas del entorno. Esta es la motivación principal para su estudio, y lo que los convierte en el primer paso natural hacia métodos más generales como los algoritmos de diferencia temporal y los enfoques model-free más avanzados.

> ##### **Ejemplo 1: Aprendizaje en un videojuego de laberintos**
>
> Imaginemos un agente que debe aprender a moverse en un laberinto con recompensas repartidas en determinadas casillas. No se le proporciona el mapa, ni se le indica cómo se comportan sus acciones (puede que a veces girar a la izquierda tenga un pequeño error, o que el resultado dependa de alguna dinámica interna desconocida del entorno).
>
> Lo único que el agente puede hacer es explorar, registrar cada partida como una secuencia de estados, acciones y recompensas, y al final de cada episodio, calcular cuántos puntos ha acumulado. Si repite esto suficientes veces comenzando desde un estado dado, podrá estimar cuál es el **retorno medio esperado** al iniciar desde ese estado bajo su comportamiento actual.
>
> Este enfoque no requiere conocer ni cómo se distribuyen las recompensas ni cómo se transita entre estados. El conocimiento proviene **únicamente de la experiencia acumulada**. Esta es la esencia del enfoque Monte Carlo.
>

> ##### **Ejemplo 2: Sistema de recomendación basado en interacción**
>
> Consideremos un sistema de recomendación que sugiere productos a los usuarios en función de sus elecciones pasadas. No se conoce un modelo preciso del comportamiento del usuario, y tampoco se puede predecir exactamente cómo responderá a una recomendación concreta.
>
> Lo que puede hacerse es observar cómo reaccionan distintos usuarios ante diferentes recomendaciones a lo largo del tiempo: si hacen clic, si compran, si abandonan. Cada secuencia de interacción puede verse como un episodio. A partir de muchos de estos episodios, el sistema puede estimar qué combinaciones de recomendaciones suelen acabar en recompensas más altas (por ejemplo, ventas) para diferentes perfiles de usuario.
>
> De nuevo, no es necesario saber de antemano cómo transiciona el sistema entre estados ni cómo se calcula la recompensa exacta. Basta con **observar episodios y promediar los resultados**, exactamente como propone Monte Carlo.
>

### Predicción con métodos Monte Carlo

La predicción es una de las tareas fundamentales en el aprendizaje por refuerzo: consiste en estimar **cuánto retorno cabe esperar** si el agente comienza en un determinado estado y sigue una política dada. Formalmente, el objetivo es aproximar la función de valor $v_\pi(s)$, definida como el valor esperado del retorno total acumulado al iniciar en el estado $s$ y actuar conforme a una política $\pi$:

$$
v^\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]
$$

El **retorno** $G_t$ representa la suma de recompensas futuras a partir del instante $t$, ponderadas por el factor de descuento $\gamma \in [0,1)$. Su expresión general era

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

En problemas episódicos, donde la interacción termina en un paso $T$, la suma es finita y los términos más allá de $T$ se consideran nulos. Una propiedad fundamental del retorno es que satisface una relación recursiva inmediata: 

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

con $G_T = 0$ si $T$ es el estado terminal. Esta recursión permite calcular todos los retornos de un episodio recorriéndolo desde el final hacia el principio, una vez que se conocen todas las recompensas.

En los métodos Monte Carlo esta predicción se realiza **únicamente a partir de la experiencia**, sin necesidad de conocer el modelo del entorno ni resolver sistemas de ecuaciones. La idea clave consiste en ejecutar múltiples **episodios completos**, registrar las recompensas obtenidas a lo largo de cada trayectoria, y calcular el retorno total desde cada estado visitado aplicando la recursión anterior. 

Cada vez que el agente visita un estado $s$ siguiendo la política $\pi$, se observa un valor concreto de $G_t$ que constituye una **realización muestral** de la variable aleatoria cuyo valor esperado es $v_\pi(s)$. Por la **ley de los grandes números**, el promedio de un número suficientemente grande de estas realizaciones converge a la esperanza matemática. De este modo, si se dispone de $N(s)$ retornos observados desde el estado $s$, la estimación de $v_\pi(s)$ se obtiene como:

$$
\hat{v}^\pi(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G^{(i)}(s)
$$

Es decir, el valor del estado se estima mediante la **media aritmética de los retornos** efectivamente obtenidos en todas las visitas (o en las primeras visitas, según la variante elegida). Este proceso es sencillo de implementar y permite una estimación directa de $v_\pi(s)$, sin requerir la función de transición $\mathcal{P}$ ni la función de recompensa $\mathcal{R}$.

Esto convierte a Monte Carlo en una técnica natural para entornos **episódicos**, donde se puede identificar un comienzo y un final claro. En cada episodio, el agente sigue la política $\pi$, y se recopilan las secuencias del tipo:

$$
(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

Una vez finalizado el episodio, se puede calcular el retorno total a partir de cualquier punto de la secuencia. Si este procedimiento se repite muchas veces y se agrupan los retornos obtenidos desde un mismo estado, la **media de dichos retornos** se convierte en una estimación del valor de ese estado bajo la política observada.

Este proceso es sencillo de implementar y permite una estimación directa de $v_\pi(s)$, sin requerir la función de transición $P$ ni la función de recompensa $R$. La única condición es que l**a política debe visitar todos los estados relevantes con suficiente frecuencia para garantizar una buena cobertura**.

> ##### Ejemplo numérico
>
> Supongamos un entorno muy simple con tres estados: $s_0$, $s_1$ y $s_2$. La política es fija y estocástica, y los episodios siempre comienzan en $s_0$ y terminan en un estado terminal $s_T$. El factor de descuento es $\gamma = 1$. Consideramos los siguientes tres episodios observados:
>
> - Episodio 1: $s_0 \to s_1 \to s_2$ con recompensas $r_1 = 0$, $r_2 = 1$
> - Episodio 2: $s_0 \to s_2$ con recompensa $r_1 = 1$
> - Episodio 3: $s_0 \to s_1 \to s_1 \to s_2$ con recompensas $r_1 = 0$, $r_2 = 0$, $r_3 = 1$
>
> Podemos calcular el retorno total $G_t$ para cada visita al estado $s_0$:
>
> - Episodio 1: $G_0 = 0 + 1 = 1$
> - Episodio 2: $G_0 = 1$
> - Episodio 3: $G_0 = 0 + 0 + 1 = 1$
>
> En los tres casos, el retorno desde $s_0$ es 1. Si promediamos estos valores, obtenemos:
>
> $$
> v^\pi(s_0) \approx \frac{1 + 1 + 1}{3} = 1
> $$
>
> Este valor se ha estimado sin conocer las transiciones ni las recompensas a priori, simplemente observando los resultados de ejecutar la política.
>

#### First-visit y every-visit Monte Carlo

En los métodos Monte Carlo para la predicción del valor de una política, existen dos variantes según cómo se utilicen las visitas a los estados en un episodio: ***first-visit*** y ***every-visit***. Ambas buscan estimar la función de valor $v_\pi(s)$ como promedio de retornos observados tras visitar el estado $s$ siguiendo la política $\pi$, pero difieren en la cantidad de muestras que extraen de cada episodio.

First-visit utiliza exclusivamente **la primera aparición de cada estado** dentro de un episodio. Una vez detectada esa primera visita, se calcula el retorno total $G_t$ desde ese momento y se emplea como muestra única para ese episodio y ese estado. En cambio, Every-visit aprovecha **todas las apariciones del estado** en un mismo episodio. Cada visita genera su propio valor de retorno y contribuye de manera independiente a la estimación de $v_\pi(s)$. Aunque ambas variantes convergen teóricamente al mismo valor esperado, en la práctica pueden comportarse de manera muy distinta en cuanto a eficiencia estadística y velocidad de convergencia.

First-visit resulta especialmente útil cuando los episodios son largos o los estados aparecen muchas veces de forma repetida, ya que evita correlaciones estadísticas dentro del mismo episodio. Esto mejora la estabilidad de la estimación, aunque puede requerir una mayor cantidad de episodios, dado que se extrae menos información de cada uno. En este sentido, **es más robusto estadísticamente, pero menos eficiente desde el punto de vista muestral**.

Every-visit, por su parte, aprovecha todas las visitas posibles a un estado, lo que **incrementa la velocidad de aprendizaje** en términos de volumen de datos. Esto es ventajoso en episodios cortos o cuando se dispone de pocos datos, pero **puede introducir varianza adicional** si las visitas múltiples no aportan información verdaderamente independiente. En entornos donde los estados tienden a repetirse en bucles, las muestras pueden volverse redundantes o correlacionadas, lo que afecta a la calidad de la estimación.

Una analogía puede ayudar a comprender mejor esta diferencia. Supongamos que se desea estimar la temperatura media de una habitación. Aplicando el enfoque first-visit, se toma una única medida por día, quizás al entrar por primera vez. Esto produce un conjunto pequeño de datos, pero cada medida representa un contexto distinto. En cambio, siguiendo el enfoque every-visit, se toman múltiples medidas durante todo el día, incluso cada pocos minutos. El volumen de datos aumenta considerablemente, pero muchas observaciones pueden reflejar situaciones muy similares, añadiendo poco valor informativo. En este caso, aunque se acumulan más datos, la calidad marginal de cada muestra puede ser menor.

##### Cálculo del promedio y uso de un parámetro $\alpha$ constante

En los métodos de Monte Carlo para la estimación de funciones de valor, el enfoque más natural consiste en calcular el promedio aritmético de los retornos observados tras cada visita a un estado. Si denotamos por $N(s)$ el número de veces que se ha visitado el estado $s$, y por $G^{(1)}, G^{(2)}, \dots, G^{(N(s))}$ los retornos obtenidos en cada una de esas visitas, la estimación de $v_\pi(s)$ es simplemente

$$
v_\pi(s) = \frac{1}{N(s)} \sum_{k=1}^{N(s)} G^{(k)}.
$$

Para evitar almacenar todos los retornos y poder actualizar la estimación de forma incremental tras cada nuevo episodio, se utiliza la conocida fórmula recursiva del promedio. Suponiendo que disponemos de la estimación $Q_{n}$ después de $n-1$ retornos y recibimos un nuevo retorno $G_{n}$, la nueva estimación se calcula como

$$
V_{n+1} = V_{n} + \frac{1}{n}\bigl[ G_{n} - V_{n} \bigr].
$$

Esta expresión es exacta y garantiza que $V_{n}$ sea la media aritmética de todos los retornos observados hasta el momento. El factor que multiplica al error $(G_{n} - V_{n})$ es la **tasa de aprendizaje** $\alpha_n = 1/n$, que depende del número de visitas. Es importante destacar que esta tasa decrece con el tiempo, de modo que cada nueva muestra tiene un peso cada vez menor. Esta propiedad es esencial para que la estimación converja al valor verdadero en entornos estacionarios.

Sin embargo, en muchos problemas de aprendizaje por refuerzo el entorno no es estacionario: las probabilidades de transición o las recompensas pueden cambiar con el tiempo. En tales casos, utilizar una tasa de aprendizaje decreciente como $1/n$ resulta inadecuado, porque las muestras antiguas —que ya no reflejan la realidad actual— seguirían teniendo el mismo peso que las recientes. El agente necesitaría adaptarse rápidamente a los cambios, olvidando gradualmente la experiencia pasada. Una forma sencilla y eficaz de conseguirlo es reemplazar la tasa variable $1/n$ por un valor **constante** $\alpha \in (0,1]$. La regla de actualización se convierte entonces en

$$
v_\pi(s) \leftarrow v_\pi(s) + \alpha \bigl[ G_t - v_\pi(s) \bigr],
$$

donde $G_t$ es el retorno observado en la última visita al estado $s$. Es importante subrayar que con esta modificación **ya no se calcula el promedio aritmético** de todos los retornos, sino una **media ponderada exponencialmente** en la que los retornos más recientes tienen mayor peso. En la práctica, se elige $\alpha$ pequeño (por ejemplo, $0.1$ o $0.01$) para suavizar las oscilaciones, o valores mayores si se necesita una respuesta rápida.

¿Por qué introducir $\alpha$ constante en un algoritmo de Monte Carlo, cuando el método original se basa en promedios exactos? La razón es doble. Por un lado, permite familiarizarnos con un mecanismo de actualización que será central en los algoritmos de diferencia temporal (TD), donde el uso de una tasa de aprendizaje constante es la norma. Por otro lado, y simples, mantener un contador de visitas por estado resulta tedioso, mientras que con $\alpha$ fijo las cuentas se simplifican. No obstante, debe quedar claro que la versión canónica de Monte Carlo emplea el promedio exacto; el uso de $\alpha$ constante es una variante práctica para entornos no estacionarios o para propósitos didácticos.

Para reflexionar…

> **¿Qué sucede con la estimación si elegimos $\alpha = 1$? ¿Y si elegimos $\alpha$ muy próximo a $0$?**  
> **Clave**: Con $\alpha=1$, la estimación se reemplaza completamente por el último retorno observado, ignorando toda la historia previa (alta varianza, nula memoria). Con $\alpha$ muy pequeño, la estimación cambia muy lentamente y apenas se adapta a los cambios del entorno, pero es muy estable.

> **¿Por qué el promedio exacto con $\alpha_n = 1/n$ no es adecuado para entornos no estacionarios?**  
> **Clave**: Porque a medida que $n$ crece, $\alpha_n$ se hace muy pequeño, y las nuevas muestras apenas modifican la estimación. Si el entorno cambia, la estimación quedaría anclada en valores pasados y no podría seguir la nueva dinámica.



Ambos métodos —first‑visit y every‑visit, así como el uso de promedios exactos o de $\alpha$ constante— son consistentes estadísticamente y conducen a estimaciones correctas si se acumulan suficientes muestras (bajo los supuestos adecuados). La decisión sobre qué variante emplear debe guiarse por el tipo de entorno, la duración típica de los episodios, la frecuencia de visitas a los estados y la necesidad de adaptación a cambios no estacionarios. First‑visit tiende a ser más conservador y robusto; every‑visit es más agresivo en el aprovechamiento de datos, a costa de asumir posibles correlaciones internas. Por su parte, el promedio exacto es el estimador natural de Monte Carlo, mientras que el uso de $\alpha$ constante introduce una ponderación exponencial que puede ser ventajosa en entornos cambiantes o para simplificar la implementación.

Este análisis justifica que todas estas estrategias formen parte del conjunto de herramientas estándar para estimar funciones de valor cuando se dispone de muestras completas de episodios. La elección entre ellas representa un compromiso entre calidad estadística, eficiencia en el uso de los datos y capacidad de adaptación.

##### Ejemplo práctico: comparación entre first-visit y every-visit Monte Carlo

Supongamos un entorno muy simple con dos estados: $A$ (no terminal) y $B$ (terminal). La política $\pi$ es determinista: desde $A$ siempre se toma la misma acción, que lleva a $B$ con una recompensa que puede variar. El factor de descuento es $\gamma = 1$.

Generamos **un único episodio** con la siguiente trayectoria (los números son las recompensas recibidas al salir de cada estado):

$$
A \xrightarrow{r=0} A \xrightarrow{r=0} A \xrightarrow{r=5} B
$$

Es decir, el agente comienza en $A$, permanece dos veces en $A$ (obteniendo recompensa 0 en cada transición) y finalmente transita a $B$ con recompensa $+5$. El episodio termina al llegar a $B$.

Observamos que el estado $A$ ha sido visitado **tres veces** dentro del mismo episodio: en los instantes $t=0$, $t=1$ y $t=2$ (asumiendo que el tiempo comienza en 0). Calculemos los retornos $G_t$ desde cada visita, teniendo en cuenta que después de la última visita ya no hay más recompensas.

- Desde la primera visita ($t=0$): le quedan por delante las recompensas $0$, $0$ y $5$.  
  $G_0 = 0 + 0 + 5 = 5$
- Desde la segunda visita ($t=1$): le quedan $0$ y $5$.  
  $G_1 = 0 + 5 = 5$
- Desde la tercera visita ($t=2$): solo le queda la recompensa $5$ al salir hacia $B$.  
  $G_2 = 5$

  **First-visit Monte Carlo**: solo considera la **primera** vez que aparece $A$ en el episodio, es decir, $t=0$. Por tanto, la muestra para $v_\pi(A)$ es $G_0 = 5$.

  **Every-visit Monte Carlo**: considera **todas** las visitas a $A$: $t=0$, $t=1$ y $t=2$. Las muestras son $\{5, 5, 5\}$. El promedio también es $5$.

En este caso, ambos métodos dan el mismo resultado porque los retornos desde todas las visitas coinciden. Para que la diferencia sea apreciable, necesitamos un episodio donde los retornos desde distintas visitas al mismo estado sean **diferentes**. Esto ocurre cuando el estado aparece en momentos tales que la cantidad de recompensa futura varía.

Vamos a considerar ahora una situación algo diferente a la anterior. Supongamos un episodio más largo:

$$
A \xrightarrow{r=1} A \xrightarrow{r=2} A \xrightarrow{r=3} B
$$

(es decir, tres transiciones desde $A$ con recompensas 1, 2 y 3, y luego termina en $B$). Las visitas a $A$ ocurren en $t=0$, $t=1$ y $t=2$ (antes de cada transición). Calculemos los retornos ($\gamma=1$):

- Desde $t=0$: $G_0 = 1 + 2 + 3 = 6$
- Desde $t=1$: $G_1 = 2 + 3 = 5$
- Desde $t=2$: $G_2 = 3$

Ahora:

- **First-visit** usa solo $G_0 = 6$.
- **Every-visit** usa los tres valores: $(6+5+3)/3 = 14/3 \approx 4.667$.

Si tuviéramos más episodios, la estimación de $v_\pi(A)$ convergería al mismo valor verdadero (que sería el retorno esperado desde $A$). Pero en una muestra finita, las estimaciones son diferentes. First-visit tiene menor varianza porque no introduce correlación entre muestras del mismo episodio, pero desperdicia información. Every-visit aprovecha más datos, pero puede incurrir en sesgo si las visitas no son independientes. Podemos ver esto en la siguiente tabla

| Instante | Estado | Acción   | Recompensa inmediata | Retorno $G_t$ desde ese instante |
| -------- | ------ | -------- | -------------------- | -------------------------------- |
| t=0      | A      | ...      | 1 (al salir)         | 1+2+3 = 6                        |
| t=1      | A      | ...      | 2                    | 2+3 = 5                          |
| t=2      | A      | ...      | 3                    | 3                                |
| t=3      | B      | terminal | -                    | 0                                |

- **First-visit** → solo usa la fila t=0 ($G=6$)
- **Every-visit** → usa las tres filas ($G=6,5,3$)

La elección entre first-visit y every-visit es un compromiso entre **varianza** y **sesgo** (o eficiencia de muestra). En la práctica, every-visit suele preferirse cuando se combina con aproximación de funciones (como redes neuronales) porque proporciona más actualizaciones por episodio, mientras que first-visit es más común en análisis teóricos por su menor correlación.

**Para reflexionar…**

> **¿En qué tipo de entornos podría ser preferible every-visit aunque introduzca correlación entre muestras?** 
> **Clave**: Cuando los episodios son muy largos y los estados aparecen muchas veces, every-visit aprovecha toda la experiencia. También cuando se usa descuento ($\gamma<1$), el impacto de las visitas lejanas es pequeño.

### Control con métodos de Monte Carlo

En el apartado anterior hemos visto cómo utilizar métodos Monte Carlo para estimar el valor de los estados bajo una política dada. Sin embargo, en el aprendizaje por refuerzo el objetivo habitual no es evaluar una política fija, sino **aprender una política que maximice el retorno esperado**. Este problema, conocido como *control*, requiere que el agente no solo cuantifique el valor de los estados o acciones, sino que también mejore progresivamente su comportamiento a partir de la experiencia.

Para abordar esta tarea con Monte Carlo, el primer paso es estimar la función **acción-valor** $q_\pi(s, a)$, que representa el retorno esperado al ejecutar la acción $a$ en el estado $s$ y continuar siguiendo la política $\pi$ a partir de ahí. A diferencia de la función $v_\pi(s)$, que mide el valor del estado bajo una política, $q_\pi(s, a)$ proporciona información más detallada sobre qué acción tomar en cada situación, y por tanto permite **comparar opciones**.

Una vez estimado $q_\pi(s, a)$, el agente puede utilizar esa información para **mejorar su política**. La idea consiste en adoptar una política que, en cada estado, elija la acción con mayor valor esperado según las estimaciones actuales. A este proceso se le conoce como política **greedy** con respecto a $q$.

No obstante, si el agente actúa siempre de forma greedy con respecto a sus valores actuales, puede dejar de explorar acciones menos conocidas que podrían ser mejores. Por esta razón, se utiliza una estrategia llamada **$\epsilon$-greedy**, que con probabilidad $1 - \epsilon$ selecciona la mejor acción conocida (explotación) y con probabilidad $\epsilon$ elige una acción al azar (exploración). Este enfoque permite **equilibrar exploración y explotación** mientras se sigue mejorando la política.

El ciclo completo del algoritmo de *control Monte Carlo* consiste, por tanto, en observar episodios generados por una política $\pi$, estimar los valores $q_\pi(s, a)$ mediante promedios de retornos observados, y usar esa información para construir una nueva política mejorada $\pi'$. A lo largo de muchos episodios, este proceso permite **converger a una política óptima** bajo condiciones adecuadas.

Este tipo de aprendizaje se puede llevar a cabo de dos formas principales:

- En el enfoque **on-policy**, el agente estima el valor de la política que realmente ejecuta y mejora progresivamente esa misma política. Es el caso del algoritmo *Monte Carlo control on-policy* con $\epsilon$-greedy, donde se evalúa y mejora una política estocástica a medida que se interactúa con el entorno.
- En el enfoque **off-policy**, el agente estima el valor de una política objetivo $\pi$ mientras sigue una política de comportamiento diferente $\mu$, que le permite explorar más ampliamente el entorno. Este enfoque requiere técnicas adicionales como el *importance sampling*, que ajustan las estimaciones para corregir la diferencia entre ambas políticas.

A lo largo de los siguientes apartados se mostrará cómo implementar el algoritmo de control Monte Carlo paso a paso, aplicando estimaciones de $q_\pi(s, a)$, estrategias $\epsilon$-greedy y actualizaciones iterativas de la política. Se analizarán también sus propiedades, limitaciones y se propondrá una práctica completa para su puesta en marcha.


#### Enfoque ""on-policy"

Cuando se utiliza Monte Carlo para resolver un problema de control, el objetivo ya no es simplemente estimar el valor de los estados bajo una política fija, sino aprender directamente una **política óptima** que maximice el retorno esperado. Para lograrlo, el agente debe combinar la estimación de la función $q_\pi(s,a)$ con un procedimiento de mejora de la política basado en dichas estimaciones.

El algoritmo se apoya en dos pilares fundamentales. Por un lado, se mantiene una política **estocástica $\epsilon$-greedy** que equilibra exploración y explotación. Por otro lado, se lleva a cabo una **estimación por promedio de retornos** de la función acción-valor $q(s,a)$ a partir de múltiples episodios generados siguiendo esa política. A medida que se recogen más datos, la política se mejora utilizando los valores actuales de $q$.

El ciclo de aprendizaje se articula en torno a la siguiente secuencia de pasos:

Al comenzar, se inicializan los valores $q(s,a)$ para todos los pares estado-acción, normalmente a cero o con un pequeño valor aleatorio. Se define también una política inicial $\pi$ basada en esos valores, que suele ser $\epsilon$-greedy respecto a $q$.

A continuación, se ejecuta un número determinado de episodios de interacción con el entorno. En cada episodio, el agente sigue su política actual, recogiendo una secuencia completa de transiciones de la forma $(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$. Una vez finalizado el episodio, se recorren las transiciones hacia atrás para calcular el retorno acumulado $G_t$ de cada par $(s_t, a_t)$.

Para cada par $(s,a)$ encontrado en el episodio, se acumulan los retornos observados y se actualiza la estimación de $q(s,a)$ como el promedio de todos los retornos recogidos hasta ese momento. Esta estimación puede basarse en el método *first-visit* o *every-visit*, según se considere solo la primera aparición o todas las apariciones del par dentro del episodio.

Después de cada episodio, se reconstruye la política actual haciendo que, en cada estado $s$, se elija con mayor probabilidad la acción $a$ que maximiza el valor estimado $q(s,a)$, pero permitiendo también con probabilidad $\epsilon$ elegir otras acciones. Este mecanismo garantiza que la política siga explorando incluso cuando tiene ya valores altos de retorno estimado.

Este proceso se repite durante muchos episodios. Si el entorno es finito, todos los estados y acciones son visitados con suficiente frecuencia, y el valor de $\epsilon$ es adecuado, entonces se garantiza la **convergencia a una política óptima** con probabilidad 1. En la práctica, el número de episodios necesarios depende de la variabilidad del entorno, de la calidad de la exploración y de la precisión deseada.

La ventaja principal de este enfoque es su **simplitud conceptual**: no requiere conocer el modelo del entorno (ni la función de transición ni la recompensa esperada), y se apoya únicamente en la observación de secuencias reales. Su desventaja, en cambio, es que requiere completar episodios completos para poder realizar actualizaciones, lo que puede ser costoso en dominios con episodios largos o poco frecuentes.

En resumen, el control on-policy mediante Monte Carlo permite aprender políticas óptimas a partir de la experiencia, manteniendo una política estocástica que se ajusta iterativamente según los valores observados de retorno. Este enfoque constituye una base sólida sobre la que se construyen métodos más avanzados, como aquellos que combinan Monte Carlo con aprendizaje por diferencia temporal.

> [!note]
>
> **¿Por qué calcular los retornos hacia atrás?**
>
> En los métodos Monte Carlo, después de cada episodio necesitamos conocer el **retorno acumulado** $G_t$ desde cada paso $t$ hasta el final del episodio. Calcularlos de forma ingenua (desde el principio hacia el final) sería muy ineficiente. La solución es **recorrer el episodio desde el último paso hacia el primero**, acumulando la recompensa paso a paso.
>
> En efecto, supongamos **un episodio con 4 pasos** (índices $t=0,1,2,3$) y $\gamma = 1$ (sin descuento). Las recompensas obtenidas son:
>
> | Paso $t$ | Acción | Recompensa $r_{t+1}$ |
> | -------- | ------ | -------------------- |
> | 0        | $a_0$  | 2                    |
> | 1        | $a_1$  | 3                    |
> | 2        | $a_2$  | 1                    |
> | 3        | $a_3$  | 5 (y termina)        |
>
> **Cálculo hacia adelante (ineficiente)**
>
> Si recorremos el episodio hacia adelante, para cada $t$ tenemos que sumar todas las recompensas que quedan desde ese punto hasta el final:
>
> - $G_0 = r_1 + r_2 + r_3 + r_4 = 2 + 3 + 1 + 5 = 11$
> - $G_1 = r_2 + r_3 + r_4 = 3 + 1 + 5 = 9$
> - $G_2 = r_3 + r_4 = 1 + 5 = 6$
> - $G_3 = r_4 = 5$
>
> Observa que para calcular $G_0$ hemos sumado 4 números; para $G_1$, 3 números; etc. En total realizamos $4+3+2+1 = 10$ sumas. Además, estamos repitiendo sumas una y otra vez (por ejemplo, $r_4$ se suma en todos los $G_t$).
>
> **Cálculo hacia atrás (eficiente)**
>
> Comenzamos desde el último paso y usamos la relación recursiva $G_t = r_{t+1} + G_{t+1}$:
>
> - En $t=3$: $G_3 = r_4 = 5$
> - En $t=2$: $G_2 = r_3 + G_3 = 1 + 5 = 6$
> - En $t=1$: $G_1 = r_2 + G_2 = 3 + 6 = 9$
> - En $t=0$: $G_0 = r_1 + G_1 = 2 + 9 = 11$
>
> Ahora solo hemos realizado **4 sumas** (una por paso). Cada retorno se obtiene directamente a partir del anterior, sin volver a sumar desde cero. **La ventaja del cálculo hacia atrás** es que cada retorno se obtiene en un solo paso: $G_t = r_{t+1} + G_{t+1}$.
>
> Recuerda que **no podemos calcular $G_t$ en el momento de visitar $s_t$** durante el episodio porque en ese momento aún no conocemos las recompensas futuras. Hay que esperar a que el episodio termine para saber el retorno total. El cálculo hacia atrás aprovecha que ya tenemos toda la información.

##### Discusión y consideraciones prácticas

El enfoque de control on-policy mediante métodos Monte Carlo constituye una de las estrategias más accesibles y conceptualmente claras para aprender políticas óptimas en entornos desconocidos. Al no requerir un modelo explícito del entorno, resulta especialmente útil en escenarios donde la dinámica del sistema es compleja, no se dispone de ecuaciones formales de transición, o simplemente no se tiene acceso a información completa sobre los efectos de las acciones.

Una de sus principales virtudes es que se basa únicamente en la experiencia real generada al interactuar con el entorno. Esto lo convierte en un método completamente *model-free*, que no necesita estimar ni utilizar funciones de transición $P$ ni funciones de recompensa $R$. Toda la información relevante se obtiene directamente a partir de los episodios observados, lo cual simplifica considerablemente la implementación inicial y permite aplicar el método en entornos reales sin suposiciones fuertes.

Sin embargo, esta dependencia de episodios completos introduce una limitación estructural importante: para poder actualizar los valores de $q(s,a)$, es necesario esperar a que el episodio finalice. Esto hace que el enfoque no sea fácilmente aplicable en tareas continuas o en dominios donde los episodios son muy largos o difíciles de completar. Además, el tiempo entre la acción y la retroalimentación puede introducir una alta varianza en las estimaciones, especialmente en dominios con resultados muy variables o recompensas retardadas.

El uso de políticas $\epsilon$-greedy garantiza la exploración suficiente del espacio de decisiones, pero también ralentiza la convergencia a la política óptima, ya que introduce aleatoriedad incluso cuando se ha identificado ya una buena acción. Reducir gradualmente el valor de $\epsilon$ a medida que mejora la política puede ser una solución eficaz para acelerar la convergencia sin perder capacidad exploratoria, aunque ello conlleva un ajuste fino de hiperparámetros y una cierta planificación del proceso de aprendizaje.

El proceso de actualización mediante promedios acumulados tiene la ventaja de ser estadísticamente consistente, pero puede resultar ineficiente en términos de velocidad de aprendizaje. En comparación con métodos basados en diferencias temporales, las actualizaciones de Monte Carlo no aprovechan el hecho de que muchas veces se puede estimar parcialmente el valor de un estado sin esperar al final del episodio. Este aspecto será clave cuando abordemos en el siguiente bloque los algoritmos TD, que combinan ventajas de Monte Carlo con aprendizaje incremental.

En resumen, el control on-policy con Monte Carlo ofrece un marco robusto para el aprendizaje de políticas óptimas, con la condición de que los episodios puedan observarse por completo y la exploración se mantenga activa. Su transparencia lo convierte en un excelente punto de partida para el estudio de algoritmos de refuerzo, pero también revela la necesidad de enfoques más eficientes cuando el entorno es complejo o el feedback es escaso. En estos casos, la extensión hacia métodos off-policy o algoritmos híbridos puede aportar mejoras sustanciales.

##### Ejemplo numérico: control Monte Carlo en un entorno simple

A diferencia de la programación dinámica, los métodos de Monte Carlo **no requieren conocer el modelo del entorno** (probabilidades de transición y recompensas esperadas). En su lugar, el agente **interactúa con el entorno** (real o simulado) y aprende a partir de la experiencia directa, generando episodios completos y promediando los retornos observados.

En este ejemplo aplicaremos **Monte Carlo para control** (aprender una política óptima) utilizando una estrategia **on‑policy** con política $\epsilon$-greedy y estimación de la función acción‑valor $q(s,a)$ mediante **first‑visit**.

Recordemos el problema del agente saltarín en su versión estocástica. Un agente se mueve de izquierda a derecha por una fila de casillas, pudiendo **avanzar** o **saltar** para evitar agujeros. 

![image-20260412104231969](./assets/image-20260412104231969.png)

El agente comienza en la casilla **0** y debe llegar a la meta en la casilla **3**. Hay un agujero en la casilla **1** y una casilla segura en la **2**. Las acciones son:

- **Avanzar** (acción 0): se mueve +1 casilla.
- **Saltar** (acción 1): se mueve +2 casillas.


El éxito de cada una de las acciones posibles en cada estado estaba sujeto a una probabilídad de éxito. En cuanto a la finalizacion del episodio este se producirá si el agente cae en un agujero (casilla 1) o se sale del tablero (casilla 4) o si llega a la meta (casilla 3). En el primer caso el episodio termina con una penalización, en el segundo, el agente recibe una recompensa grande.

En cuanto a las recompensas y penalizaciónes supondremos que caer en agujero (casilla 1) retorna **-10**; salir del tablero (casilla 4)  **-5** y llegar a la meta (casilla 3) **+20**. Además consideraremos que existen costes por cada accion de avanzar o de saltar igual a **-1**

Supongamos por último que el problema tiene asociado un factor de descuento $\gamma=0.9$

El grafo del MDP era el siguiente:



![image-20260412101744254](./assets/image-20260412101744254.png)



El mapa de **transiciones estocásticas**, que aunque son desconocidas para el agente el entorno las aplica es el siguiente:

| Estado | Acción  | Resultado posible        | Probabilidad |
| ------ | ------- | ------------------------ | ------------ |
| 0      | avanzar | va a 1 (r=-10, terminal) | 0.8          |
| 0      | avanzar | se queda en 0 (r=-1)     | 0.2          |
| 0      | saltar  | va a 2 (r=-1)            | 0.9          |
| 0      | saltar  | va a 1 (r=-10, terminal) | 0.1          |
| 2      | avanzar | va a 3 (r=+19, terminal) | 0.9          |
| 2      | avanzar | se queda en 2 (r=-1)     | 0.1          |
| 2      | saltar  | va a 4 (r=-5, terminal)  | 0.8          |
| 2      | saltar  | se queda en 2 (r=-1)     | 0.2          |

El algoritmo funcionará de la manera siguiente:

1. Inicializamos $q(s,a) = 0$ para todos los pares estado‑acción.

2. Supondremos una política inicial: $\epsilon$-greedy con $\epsilon = 0.2$ (por ejemplo). Como todos los $q$ son iguales, al principio las acciones se eligen al azar con igual probabilidad. En cualquier caso, en un 20% de las ocasiones el algoritmo elegirá explotación

3. Se genera un episodio siguiendo la política del paso 2 y, comenzando en el estado 0, se registra cada paso como una tupla $(s_t, a_t, r_{t+1}, s_{t+1})$.
4. Se calculan los retornos y se actualizan los valores $q(s,a)$
5. Se vuelve al paso 3. Cabe la posibilidad de que existe un parámetro de decaimiento de $\epsilon$ que habrá que aplicar a la política.

Tendremos en cuenta los siguientes hiperparámetros:

- $\gamma$ = 0.9 (descuento)  
- $\alpha$ = 0.1 (tasa de aprendizaje, constante)  
- ε = 0.2 (exploración en política ε‑greedy)  

Simulemos algunos episodios:

**Episodio 1:** Sorteamos la política inicial. Como en $k=0$ $q(s,a)=0\, \forall (s,a)$ podemos elegir cualquier secuencia. La secuencia que se elige es: (saltar, avanzar, (T)). $T$ hace referecnia al estado terminal.

- Paso 0: estado 0 → se elige acción `saltar` (por azar).  
  Resultado: va a estado 2, recompensa -1.  
- Paso 1: estado 2 → se elige acción `avanzar` (por azar).  
  Resultado: va a estado 3 (meta), recompensa +19. Episodio termina.

  **Cálculo de retornos desde atrás:**  

- Empezamos con G = 0 (después del final).  
- Último paso (t=1): G₁ = 19 + $\gamma$·0 = 19.  
- Paso anterior (t=0): G₀ = -1 + $\gamma$·19 = -1 + 0.9·19 = -1 + 17.1 = 16.1.

**Nuevos valores de Q** (primera visita, actualización incremental con $\alpha$=0.1):  

- Q(2, avanzar) ← 0 + 0.1·(19 – 0) = 1.9  
- Q(0, saltar)  ← 0 + 0.1·(16.1 – 0) = 1.61  

**Tabla Q tras episodio 1:**

| (s,a)        | valor |
| ------------ | ----- |
| (0, avanzar) | 0     |
| (0, saltar)  | 1.61  |
| (2, avanzar) | 1.9   |
| (2, saltar)  | 0     |

**Episodio 2**

- Paso 0: estado 0 → Q(0,saltar)=1.61 > Q(0,avanzar)=0.  
  Con probabilidad 1‑ε = 0.8 se elige la acción greedy: `saltar`.  
  Resultado: va a estado 2, recompensa -1.  
- Paso 1: estado 2 → Q(2,avanzar)=1.9 > Q(2,saltar)=0.  
  Con probabilidad 0.8 se elige `avanzar`.  
  Resultado: va a estado 3, recompensa +19. Episodio termina.

  **Valores actuales de Q** (antes de actualizar):  
  Los de la tabla anterior.

  **Aplicación de política ε‑greedy**:  
  En ambos pasos se usó explotación (no hubo exploración porque el sorteo cayó en el 80%).

  **Cálculo de retornos desde atrás**:  
  Mismos que en episodio 1: G₁ = 19, G₀ = 16.1.

  **Nuevos valores de Q** (actualización incremental):  

- Q(2, avanzar) ← 1.9 + 0.1·(19 – 1.9) = 1.9 + 0.1·17.1 = 1.9 + 1.71 = **3.61**  
- Q(0, saltar)  ← 1.61 + 0.1·(16.1 – 1.61) = 1.61 + 0.1·14.49 = 1.61 + 1.449 = **3.059**

**Tabla Q tras episodio 2:**

| (s,a)        | valor |
| ------------ | ----- |
| (0, avanzar) | 0     |
| (0, saltar)  | 3.059 |
| (2, avanzar) | 3.61  |
| (2, saltar)  | 0     |

**Episodio 3 (ejemplo de episodio malo por estocasticidad)**

**1. Secuencia que se elige** (política greedy, pero entorno estocástico juega en contra):  

- Paso 0: estado 0 → acción greedy: `saltar` (porque 3.059 > 0).  
  El entorno, con probabilidad 0.1 (en la versión estocástica), hace que el salto caiga en el agujero (estado 1) en lugar de en el estado 2.  
  Resultado: va a estado 1, recompensa -10. Episodio termina inmediatamente (solo un paso).

  **Valores actuales de Q**: los de la tabla anterior.

  **Aplicación de política ε‑greedy**:  
  Se eligió acción greedy (no hubo exploración), pero el entorno introdujo su propia aleatoriedad.

  **Cálculo de retornos desde atrás**:  
  Solo hay un paso: G₀ = -10 (no hay más recompensas).

  **Nuevos valores de Q** (solo actualizamos el par visitado):  

- Q(0, saltar) ← 3.059 + 0.1·(-10 – 3.059) = 3.059 + 0.1·(-13.059) = 3.059 – 1.3059 = **1.7531**  

**Tabla Q tras episodio 3:**

| (s,a)        | valor |
| ------------ | ----- |
| (0, avanzar) | 0     |
| (0, saltar)  | 1.753 |
| (2, avanzar) | 3.61  |
| (2, saltar)  | 0     |



Los tres episodios simulados ilustran el comportamiento del algoritmo de control Monte Carlo on‑policy con una política $\epsilon$-greedy y actualización de $q(s,a)$. Aunque el número de episodios es reducido, ya se observan algunas tendencias significativas.

En el **episodio 1**, al partir de valores nulos, las acciones se eligieron de forma completamente aleatoria (por el desempate). Se obtuvo una trayectoria favorable (saltar desde 0 hacia 2, luego avanzar hacia la meta), lo que permitió inicializar $Q(0,\text{saltar})$ y $Q(2,\text{avanzar})$ con valores positivos (1.61 y 1.9 respectivamente). En el **episodio 2**, la política $\epsilon$-greedy ya favorecía esas mismas acciones (explotación), y se repitió la misma trayectoria exitosa. Como consecuencia, ambos valores aumentaron hasta 3.059 y 3.61, consolidando la preferencia por saltar en el estado 0 y avanzar en el estado 2.

El **episodio 3** muestra un caso menos favorable debido a la estocasticidad del entorno: a pesar de que el agente volvió a elegir la acción greedy (saltar), el resultado real fue el agujero (con un 10% de probabilidad), obteniendo una recompensa de -10. Este evento negativo redujo drásticamente $Q(0,\text{saltar})$ de 3.059 a 1.753, demostrando cómo una única mala experiencia puede penalizar temporalmente una acción que, en promedio, es beneficiosa. Sin embargo, con suficientes episodios, el promedio de los retornos convergería al valor esperado verdadero (aproximadamente 13.3 para saltar desde 0 y 18.68 para avanzar desde 2), tal como se calculó mediante iteración de valores. La tabla final tras el episodio 3 refleja esa incertidumbre inicial, pero la política $\epsilon$-greedy seguiría eligiendo mayoritariamente saltar en 0 (1.753 sigue siendo superior a 0) y avanzar en 2 (3.61 frente a 0).

> Este ejemplo práctico permite observar varias propiedades fundamentales de los métodos Monte Carlo en problemas de control
>
> 1. **Aprendizaje a partir de la experiencia**: el agente no necesita conocer las probabilidades de transición; simplemente interactúa con el entorno (real o simulado) y registra los retornos obtenidos.
> 2. **Compromiso exploración‑explotación**: la política $\epsilon$-greedy garantiza que, con probabilidad $\epsilon$, se sigan probando acciones distintas de las que actualmente parecen mejores, lo que evita quedarse atrapado en una política subóptima. En los episodios mostrados no ocurrió exploración, pero si $\epsilon$ fuera mayor, aparecerían episodios donde se elige avanzar desde 0 o saltar desde 2, proporcionando información sobre esas acciones.
> 3. **Sensibilidad a la estocasticidad**: la caída ocasional en el agujero reduce el valor estimado de la acción saltar, pero el mecanismo de promediado (con $\alpha$ constante) permite que con más episodios el valor se estabilice en torno a la media verdadera.
> 4. **Actualización incremental**: el uso de $\alpha = 0.1$ simplifica los cálculos y permite que las estimaciones se adapten gradualmente, aunque a costa de no recordar el número exacto de visitas. En un entorno estacionario, lo ideal sería usar $\alpha = 1/n$ para obtener el promedio exacto; aquí se ha optado por una tasa constante para facilitar la comprensión y mostrar la adaptabilidad.
>
> En resumen, el agente saltarín ha comenzado a aprender una política razonable (saltar en 0 y avanzar en 2) tras solo tres episodios, aunque sus estimaciones aún distan de los valores óptimos. Con un número elevado de episodios (decenas o cientos), los valores $Q$ convergerían hacia los obtenidos mediante programación dinámica, y la política se volvería prácticamente determinista (eligiendo siempre la mejor acción en cada estado). Este ejemplo pone de manifiesto la esencia del aprendizaje por refuerzo: mejorar el comportamiento a partir de la interacción directa con el entorno, sin necesidad de conocer su modelo interno.
>

**Para reflexionar…**

> **¿Qué habría ocurrido si en el episodio 3, en lugar de elegir la acción greedy, el agente hubiera explorado (con probabilidad $\epsilon$) y hubiera elegido avanzar desde 0? ¿Cómo afectaría eso a la estimación de $Q(0,\text{avanzar})$?** 
>
> **Clave**: Habría obtenido un retorno probablemente muy negativo (80% de caer en el agujero, -10; 20% de permanecer en 0, -1), lo que mantendría bajo el valor de avanzar. La política seguiría prefiriendo saltar.

##### Control off-policy con Monte Carlo

Hasta este punto hemos trabajado bajo un supuesto clave: el agente genera su experiencia actuando según la **misma política** que está tratando de evaluar o mejorar. A este enfoque se lo denomina aprendizaje *on-policy*. Sin embargo, existen situaciones prácticas donde esto no es posible ni deseable. Puede que el agente quiera aprender a partir de **experiencias pasadas** generadas por otra política, o bien que necesite **evaluar una política ideal** sin haberla puesto aún en práctica. En estos escenarios se recurre a métodos **off-policy**.

La idea central del aprendizaje off-policy es **desacoplar la política de comportamiento** (la que se utiliza para generar experiencia) de la **política objetivo** (la que se desea evaluar o mejorar). Formalmente, se parte de dos políticas distintas:

- $\mu$: política de comportamiento, que se usa para interactuar con el entorno.
- $\pi$: política objetivo, que se desea evaluar o mejorar.


El reto es que los episodios observados provienen de las decisiones tomadas por $\mu$, no por $\pi$. Esto introduce un sesgo potencial, ya que las secuencias de estados y acciones no siguen la distribución de probabilidad inducida por la política $\pi$ que se quiere evaluar. Para **corregir este sesgo**, se emplea una técnica estadística conocida como **importance sampling**.

El *importance sampling* permite reponderar cada episodio observado con un **factor de corrección** que mide cuánto se desvían las decisiones tomadas por $\mu$ respecto a lo que habría hecho $\pi$. Para cada trayectoria $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ se calcula el **peso de importancia**:

$$
\rho(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}
$$

Este factor indica la proporción entre la probabilidad de que la política objetivo $\pi$ hubiera generado esa trayectoria, frente a la política de comportamiento $\mu$. A mayor valor de $\rho(\tau)$, mayor confianza en que la trayectoria observada es representativa de la política $\pi$.

Una vez calculado este peso, puede utilizarse para ajustar las estimaciones de retorno. Por ejemplo, si se desea estimar el valor $v_\pi(s)$ mediante first-visit Monte Carlo, se acumulan los retornos $G_t$ ponderados por $\rho(\tau)$, de modo que:

$$
V^\pi(s) = \frac{\sum_{\tau \in \mathcal{E}} \rho(\tau) \cdot G_t(\tau)}{\sum_{\tau \in \mathcal{E}} \rho(\tau)}
$$

donde $E_s$ es el conjunto de episodios en los que se visitó el estado $s$ por primera vez. De forma análoga, puede calcularse $q_\pi(s, a)$ ponderando solo aquellos episodios donde se observó el par $(s, a)$.

Este enfoque es estadísticamente consistente: si se observa un número suficiente de episodios generados por $\mu$ y se aplican correctamente los pesos, las estimaciones convergen a los valores reales de la política $\pi$. Sin embargo, la principal dificultad es la **alta varianza** que puede producirse. Si $\pi$ y $\mu$ son muy distintas, los pesos de importancia pueden volverse inestables y generar estimaciones poco fiables. Este problema es especialmente agudo en secuencias largas, donde el producto de muchos cocientes puede crecer o decrecer exponencialmente.

Por esta razón, los métodos off-policy con Monte Carlo suelen utilizar variantes **con truncamiento o suavizado de pesos**, como el *importance sampling ponderado*, o bien recurrir a aproximaciones por *bootstrapping* en métodos TD, que ofrecen menor varianza a costa de introducir algo de sesgo.

El valor fundamental de estos métodos es que permiten **reutilizar experiencia** generada por otras políticas, agentes o simulaciones, lo que los hace extremadamente útiles en contextos donde la exploración activa es costosa, arriesgada o limitada. También permiten **evaluar múltiples políticas** a partir de un único conjunto de datos, lo que los convierte en una herramienta crucial en aprendizaje por refuerzo offline o en simulación basada en datos históricos.

###### Ejemplo: estimación de $v_\pi(s)$ con Monte Carlo off-policy

Supongamos un entorno muy simple con tres estados $s_0$, $s_1$ y $s_2$, y dos acciones posibles en cada estado: $a_0$ y $a_1$. El objetivo es estimar el valor $v_\pi(s_1)$, es decir, el retorno esperado al partir del estado $s_1$ y seguir la política $\pi$. La política objetivo $\pi$ es **determinista** y en $s_1$ siempre selecciona $a_1$.

Sin embargo, los episodios observados provienen de una política de comportamiento $\mu$ que actúa de forma **aleatoria uniforme**, seleccionando $a_0$ y $a_1$ con probabilidad 0.5 en cada estado. Esto hace necesario reponderar los episodios observados con un factor de *importance sampling* que corrija esta diferencia de comportamiento.

Supongamos que tenemos los siguientes tres episodios observados, todos iniciando en $s_1$:

- Episodio 1: $(s_1, a_1) \to s_2$, $G = 1$
- Episodio 2: $(s_1, a_0) \to s_0$, $G = 0$
- Episodio 3: $(s_1, a_1) \to s_2$, $G = 1$

Queremos estimar $v_\pi(s_1)$ mediante first-visit Monte Carlo off-policy. Solo consideraremos aquellos episodios en los que se haya seguido la acción que la política $\pi$ habría tomado, es decir, aquellos donde se ha ejecutado $a_1$ en $s_1$.

Veamos los factores de importancia para cada episodio:

- Episodio 1: la acción tomada coincide con $\pi$, luego:

$$
\rho = \frac{\pi(a_1 \mid s_1)}{\mu(a_1 \mid s_1)} = \frac{1}{0.5} = 2
$$

- Episodio 2: la acción tomada no coincide con $\pi$, entonces:

$$
\rho = \frac{\pi(a_0 \mid s_1)}{\mu(a_0 \mid s_1)} = \frac{0}{0.5} = 0
$$

- Episodio 3: de nuevo coincide con $\pi$:
  
$$
\rho = \frac{1}{0.5} = 2
$$

Aplicamos ahora la estimación de $v_\pi(s_1)$ como promedio ponderado:

$$
V^\pi(s_1) = \frac{2 \cdot 1 + 0 \cdot 0 + 2 \cdot 1}{2 + 0 + 2} = \frac{4}{4} = 1
$$

Este valor indica que, si se siguiera siempre la política $\pi$ desde $s_1$, el retorno esperado observado sería 1, basado en los episodios compatibles con esa política.

Este ejemplo sencillo muestra cómo es posible **evaluar una política determinista** utilizando únicamente muestras generadas por una política completamente aleatoria, gracias al uso del *importance sampling*. Observamos también que los episodios incompatibles con $\pi$ (como el segundo) no aportan información útil para esta estimación y reciben peso cero.

Este tipo de técnicas es fundamental en escenarios reales donde la política de interés aún no se ha ejecutado o donde los datos provienen de interacciones previas bajo otras estrategias. Permiten **aprovechar al máximo la información disponible**, aunque al precio de una mayor complejidad estadística y sensibilidad a la varianza de los pesos.

#### Consideraciones finales sobre los métodos Monte Carlo

Los métodos de Monte Carlo ofrecen una vía directa e intuitiva para aprender a partir de la experiencia completa del agente en forma de episodios. Su punto de partida es radicalmente distinto al de la programación dinámica: no se requiere conocimiento alguno del modelo del entorno, y el aprendizaje se basa exclusivamente en las secuencias observadas de interacciones reales, sin necesidad de simulaciones internas ni planificación explícita.

Esta aproximación es particularmente adecuada en dominios donde los episodios tienen una estructura bien definida, como juegos, procesos finitos o tareas que pueden completarse de forma natural. La estimación del valor de un estado o de una acción se realiza a partir de promedios acumulados de retornos, lo cual garantiza una convergencia estadística sólida siempre que se respete la hipótesis de exploración suficiente.

El principal punto fuerte de estos métodos es su claridad conceptual y la transparencia de su implementación. A diferencia de los métodos basados en diferencias temporales, los algoritmos Monte Carlo no requieren ninguna suposición adicional sobre la estructura del entorno, más allá de la capacidad de generar episodios y observar las recompensas asociadas. Esta simplicidad los convierte en una excelente puerta de entrada al aprendizaje por refuerzo, y permite construir de manera progresiva la intuición sobre el valor de los estados, las políticas y el papel del retorno.

Ahora bien, esta misma dependencia de episodios completos introduce limitaciones operativas importantes. Los métodos Monte Carlo no son aplicables en entornos continuos donde no existe un final definido, o en tareas donde el feedback se obtiene de manera muy tardía o parcial. Además, el hecho de esperar hasta el final del episodio para actualizar los valores puede introducir varianza elevada, especialmente cuando la duración de los episodios es muy variable.

El control on-policy mediante políticas $\epsilon$-greedy permite mejorar progresivamente las decisiones del agente sin abandonar del todo la exploración. Este mecanismo mantiene un delicado equilibrio entre refinar las acciones buenas conocidas y seguir explorando alternativas, aunque no siempre de manera eficiente. Por su parte, el control off-policy con *importance sampling* permite utilizar experiencia obtenida por otras políticas, incluso cuando estas difieren completamente de la política objetivo. Esta capacidad es clave en tareas de aprendizaje desde datos históricos, aunque introduce desafíos estadísticos importantes por la alta varianza de los pesos de corrección.

En términos de su posición dentro del ecosistema del aprendizaje por refuerzo, los métodos Monte Carlo constituyen una transición natural entre la planificación basada en modelos y el aprendizaje incremental de los métodos TD. En muchos sentidos, ofrecen lo mejor del aprendizaje basado en experiencia, pero sin la eficiencia computacional de los algoritmos que actualizan tras cada paso.

Como resumen conceptual, puede decirse que los métodos Monte Carlo:

- Proporcionan una forma robusta de **evaluar políticas** sin conocer el modelo.
- Pueden usarse para **mejorar políticas** mediante estrategias exploratorias como $\epsilon$-greedy.
- Permiten **aprovechar episodios pasados** para el aprendizaje off-policy, usando técnicas como el *importance sampling*.
- Se enfrentan a limitaciones prácticas cuando los episodios son largos, indefinidos o costosos de generar.


En el siguiente capítulo se abordarán los **métodos basados en diferencias temporales (TD)**, que introducen una idea clave: la posibilidad de actualizar los valores sin esperar al final del episodio. Este cambio de paradigma abre la puerta a algoritmos más eficientes y generalizables, que constituyen el núcleo de la mayoría de sistemas modernos de aprendizaje por refuerzo
