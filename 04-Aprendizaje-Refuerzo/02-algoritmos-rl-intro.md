# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Algoritmos de Aprendizaje por Refuerzo. Introducción

### El agente en el entorno: objetivo y formalización del problema

Ya hemos visto cómo el aprendizaje por refuerzo clásico se basaba en la formalización de la interacción agente-entorno mediante procesos de decisión de Markov (MDP).

Un **Proceso de Decisión de Markov (MDP)** se definía formalmente como una 5-tupla:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

donde:

- $\mathcal{S}$ era el conjunto de **estados** posibles.
- $\mathcal{A}$ era el conjunto de **acciones** disponibles para el agente.
- $\mathcal{P}$ era la **función de transición**, que da la probabilidad de pasar al estado $s'$ al ejecutar la acción $a$ en el estado $s$.
- $\mathcal{R}$ era la **función de recompensa**, que devuelve el valor esperado de la recompensa al realizar la acción $a$ en el estado $s$ y transitar a $s'$.
- $\gamma \in [0,1)$ era el **factor de descuento**, que determina el peso relativo de las recompensas futuras frente a las inmediatas.


Esta formalización proporciona una base matemática clara para definir los elementos del problema: el espacio de estados, el conjunto de acciones, la función de transición, la función de recompensa y el objetivo del agente. 

En los procesos de decisión de Markov, el **agente** es la entidad que **interactúa activamente con el entorno** a lo largo del tiempo, eligiendo acciones en función del estado en que se encuentra. A cada paso temporal $t$, el agente observa el estado actual $s_t$, selecciona una acción $a_t$ del conjunto de acciones disponibles $\mathcal{A}$, y como resultado recibe una recompensa $r_{t+1}$ y transita al nuevo estado $s_{t+1}$.

Esta interacción da lugar a una secuencia temporal de la forma:

$$
s_0, a_0, r_1, s_1, a_1, r_2, s_2, \dots
$$

En este contexto, el comportamiento del entorno puede ser **determinista o estocástico**, si bien en la mayoría de aplicaciones reales se asume la existencia de **incertidumbre** en la transición de estados. Es decir, al ejecutar una misma acción en un mismo estado, el resultado no siempre es el mismo: puede variar según una distribución de probabilidad. Este componente estocástico, representado por $\mathcal{P}$, es esencial para modelar fenómenos como ruido, aleatoriedad ambiental o imprecisión en la ejecución.

El papel del agente es **elegir acciones** que le permitan alcanzar un objetivo. Pero este objetivo no se expresa como un estado concreto a alcanzar, sino como la **maximización del retorno acumulado de recompensas a lo largo del tiempo**. En este sentido, el agente actúa en un entorno incierto con el propósito de obtener, en promedio, la mayor utilidad posible.

Formalmente, el **objetivo del agente** es maximizar el **retorno total esperado**, que como ya se ha definido, es:
$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Esta expresión incorpora la recompensa inmediata y todas las futuras, ponderadas por el factor de descuento $\gamma$. El agente debe entonces tomar decisiones que influyan en la secuencia de estados y recompensas, de forma que la suma total $G_t$ sea máxima **en expectativa**.

Por esto es por lo que el problema de decisión se convierte en una cuestión de planificación bajo incertidumbre: dado que el entorno es estocástico, el agente no puede garantizar un resultado único, pero sí puede influir en la distribución de trayectorias para que, en promedio, se obtenga el mayor retorno posible. 

Así, una vez que se entiende que el agente busca maximizar el retorno acumulado esperado en un entorno incierto, surge la necesidad de describir formalmente cómo toma decisiones. Esta interacción entre el agente y el entorno se estructura en torno a tres elementos fundamentales, que permiten modelar, evaluar y optimizar su comportamiento:

- **La política**: es la estrategia de decisión del agente. Describe qué acción tomar en cada estado, pudiendo ser determinista o estocástica. Es el objeto que define el comportamiento del agente y será el núcleo de los algoritmos de aprendizaje.

- **La función estado-valor**: asigna a cada estado $s$ el valor esperado del retorno si el agente comienza en $s$ y sigue su política desde ese punto. Esta función permite evaluar **cuán deseable es estar en un determinado estado**, considerando lo que puede esperarse a futuro.

- **La función acción-valor**: extiende la anterior al par estado-acción $(s, a)$, cuantificando la utilidad esperada de realizar la acción $a$ en el estado $s$, y seguir luego la política del agente. Esta función permite **comparar directamente distintas decisiones** posibles en un mismo estado.


Estos tres componentes —la política, la función de valor y la función de acción-valor— constituyen la base operativa del aprendizaje por refuerzo. Son las herramientas que el agente utiliza para razonar sobre el entorno, predecir consecuencias y mejorar su comportamiento a través de la experiencia o del cálculo.

En las siguientes secciones abordaremos cada uno de ellos con detalle, comenzando por la **definición formal de la política** y su papel en la toma de decisiones.

#### La política ($\pi$): Cómo actua el agente

Una vez que hemos entendido que el agente busca maximizar el retorno acumulado en un entorno incierto, necesitamos responder a la siguiente pregunta fundamental: **¿cómo decide el agente qué acción tomar en cada momento?**. La respuesta a esta pregunta se formaliza mediante el concepto de **política**.

Podemos pensar en la política como el **comportamiento planificado del agente**: una regla que le indica qué hacer en cada estado en que se encuentre. Dado que el agente no controla directamente las transiciones del entorno (son estocásticas), su única herramienta de decisión es elegir las acciones con criterio. La política es, en este sentido, **la estrategia de actuación que define la interacción del agente con el entorno**.

La política se denota como $\pi$, y puede representarse de dos formas, según el tipo de decisión que describa:

En su forma más general, $\pi(a \mid s)$ es una **función de probabilidad** que indica la **probabilidad de ejecutar la acción $a$ estando en el estado $s$**. Este tipo de política se denomina **estocástica**.

En el caso más sencillo, $\pi(s)$ es una **función determinista** que asigna a cada estado una única acción. Es decir, el agente tiene una regla fija del tipo “si estoy en el estado $s$, siempre hago la acción $a$”.

Ambas formas son válidas. De hecho, incluso cuando el entorno es determinista, puede ser útil definir políticas estocásticas, por ejemplo para introducir aleatoriedad controlada en la exploración o para facilitar el análisis teórico.

Podemos imaginar que el agente lleva consigo un pequeño “manual de instrucciones” —su política— donde, al llegar a un estado $s$, consulta qué acción le conviene ejecutar. Este manual puede contener reglas precisas (deterministas) o recomendaciones con distintos niveles de confianza (estocásticas). A lo largo del aprendizaje, el agente ajusta este manual con el objetivo de obtener mejores resultados en el futuro.

En muchos entornos simples y completamente observables, una política determinista puede ser suficiente e incluso óptima. Sin embargo, en situaciones donde hay **incertidumbre en las transiciones**, **recompensas variables** o **información parcial**, las políticas estocásticas permiten una toma de decisiones más flexible. También son necesarias en métodos que utilizan muestreo (como Monte Carlo o métodos de gradiente de política), donde se necesita explorar distintas trayectorias.

Una vez definida la política $\pi$, podemos calcular el **retorno esperado** de seguir dicha política desde un estado dado. Esto nos permitirá evaluar si la política actual es buena o necesita ser mejorada. En las próximas secciones introduciremos precisamente estas herramientas: las **funciones de valor**, que cuantifican el rendimiento esperado de una política y permiten compararlas entre sí.

Por esto es por lo que la política no solo es el medio por el cual el agente actúa, sino también el objeto central del aprendizaje: todo el objetivo del aprendizaje por refuerzo consiste, en última instancia, en **encontrar una política que maximice el retorno esperado**.

#### **La función estado-valor: evaluar la calidad de un estado**

Una vez que el agente ha definido una política $\pi$, es natural preguntarse **cuán buena es esa política** y si conviene mantenerla o modificarla. Para responder a esta cuestión necesitamos herramientas que permitan evaluar su rendimiento. Una de ellas, y posiblemente la más importante, es la **función estado-valor**, también conocida como **función de valor**.

Esta función, denotada como $v_\pi(s)$, asigna a cada estado $s$ el **valor esperado del retorno total** que obtendría el agente **si comenzara en ese estado y siguiera la política $\pi$ en adelante**. Es decir, mide **cuán valioso es estar en un determinado estado**, suponiendo que a partir de ahí el agente se comporta conforme a su política.

Recordemos el anterior ejemplo del tablero bidimensional 4x4 y supongamos que el agente se encuentra en la casilla $s_8$, en la tercera fila de nuestro tablero, y sigue una política $\pi$ que intenta llegar a la casilla meta $s_{15}$. Podemos tratar de representar el **valor esperado de las recompensas acumuladas** que el agente obtendrá si empieza desde $s_8$ y aplica esa política hasta llegar a un estado terminal.

Si la política está bien diseñada, y consigue alcanzar el estado $s_{15}$ en unos 5 pasos con cierta probabilidad, entonces este valor será cercano al valor de la recompensa de la meta, ponderado por el descuento temporal. Si, por el contrario, la política es ineficaz y el agente termina muchas veces en estados absorbentes sin recompensa, entonces el valor esperado será bajo o incluso nulo.

Realmente vemos como este valor no refleja una certeza, sino una **esperanza matemática**, calculada como promedio sobre todas las trayectorias posibles inducidas por la política $\pi$ y por la dinámica estocástica del entorno.

Formalmente, la función estado-valor bajo una política $\pi$ se define como:

$$
v_\pi(s) = \mathbb{E}_{\pi} \left[ G_t | S_t = s \right] = \mathbb{E}_{\pi} \left[ \sum_{k} \gamma^k \, R_{t+k+1} | S_t = s \right]
$$

Es decir, es el **valor esperado** del **retorno descontado**, condicionado a que el agente comienza en el estado $s$ y sigue la política $\pi$ a partir de ese instante.

Veamos con más detalle cada uno de los elementos presentes en la fórmula de $v_\pi(s)$:

- **$\mathbb{E}_\pi[\;]$**: denota una **esperanza matemática** (valor esperado) tomada sobre todas las posibles trayectorias que el agente puede seguir, suponiendo que actúa según la política $\pi$. La esperanza se calcula considerando la estocasticidad del entorno y de la propia política, si esta es probabilística.

- **$s_t = s$**: indica que el agente comienza en el estado $s$ en el instante $t$.

- **$\sum_{k=0}^\infty \gamma^k , r_{t+k+1}$**: es la **definición del retorno total** desde el paso $t$ en adelante. Suma todas las recompensas futuras, cada una multiplicada por un factor $\gamma^k$ que penaliza las recompensas lejanas. Este descuento refleja la preferencia por obtener recompensas lo antes posible.

- **$\gamma \in [0,1)$**: es el **factor de descuento**. Si $\gamma$ es cercano a 1, el agente se comporta de forma más “paciente”, valorando también recompensas futuras. Si $\gamma$ es bajo, se comporta de forma más miope, dando preferencia a resultados inmediatos.

- **$R_{t+k+1}$**: representa la **recompensa recibida** al pasar del estado $s_{t+k}$ al estado $s_{t+k+1}$ tras ejecutar la acción correspondiente.

  La función $v_\pi(s)$ cumple un papel crucial: **permite comparar estados entre sí** en términos de su valor futuro. Un estado con un valor alto bajo una política dada es preferible, ya que iniciar desde él conduce a un mayor retorno esperado. Esta función puede ser usada tanto para evaluar la calidad de una política como para guiar la mejora del comportamiento del agente.

  En las próximas secciones veremos cómo esta función se relaciona con otra componente clave como es la **función acción-valor**.

#### **La función acción-valor: evaluar decisiones concretas**

Mientras que la función estado-valor $v_\pi(s)$ nos permite valorar **cuán bueno es estar en un estado** si se sigue una política determinada, a menudo nos interesa ir un paso más allá: **¿qué pasa si el agente se plantea ejecutar una acción concreta en ese estado?**. La respuesta a esta pregunta es lo que proporciona la **función acción-valor**, también conocida como función $q$.

Esta función permite al agente **comparar distintas decisiones posibles** en un mismo estado antes de ejecutarlas. No se limita a decir “este estado es bueno”, sino que responde a “si en este estado tomo esta acción, ¿qué resultado puedo esperar?”.

Podemos pensar en ella como una especie de “evaluador interno” que estima el impacto futuro de cada acción en cada situación. Este conocimiento es clave cuando el agente necesita elegir entre varias alternativas disponibles, especialmente si la política aún no es definitiva.

Recordemos el ejemplo del tablero bidimensional 4x4. Supongamos que el agente se encuentra en la casilla $s_{13}$ del tablero, justo al lado de la casilla $s_{14}$, que a su vez está adyacente al estado objetivo $s_{15}$. El agente podría tener dos acciones disponibles: moverse a la derecha ($a_2$, que lo lleva a $s_{14}$) o hacia arriba ($a_3$, que lo lleva a $s_9$). La función $q_\pi(s_{13}, a_2)$ estima el valor esperado de tomar la acción $a_2$ en el estado $s_{13}$, y luego continuar actuando según la política $\pi$ desde $s_{14}$ en adelante. Si este valor es alto, podría indicar que ese camino conduce con buena probabilidad hacia el objetivo. Por el contrario, si $q_\pi(s_{13}, a_3)$ es bajo, entonces moverse hacia arriba puede alejar al agente de la meta, o incluso acercarlo a estados absorbentes que cancelen el episodio.

Así, mientras $V^\pi(s_{13})$ resume el valor esperado de estar en $s_{13}$ en general, $q_\pi(s_{13}, a)$ descompone ese valor en función de las distintas **acciones iniciales** disponibles. Esto da al agente un instrumento más fino para **decidir con criterio cuál es la mejor acción en cada situación**.

Cuando el agente dispone de la función $q_\pi(s, a)$, no necesita conocer $v_\pi(s)$ para actuar: puede elegir directamente la acción con mayor valor esperado entre todas las disponibles en $s$. De hecho, esta función será fundamental cuando tratemos de aprender no solo la política, sino la propia función de valor a partir de la experiencia.

La definición formal de la función acción-valor bajo una política $\pi$ es la siguiente:

$$
q_\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k  r_{t+k+1} | S_t = s, A_t = a \right]
$$

Esta expresión cuantifica el **retorno esperado descontado** que el agente obtendrá si se encuentra en el estado $s$, ejecuta la acción $a$ en ese instante y luego sigue la política $\pi$ a partir de $t+1$.

Analicemos sus componentes con atención. El estado inicial es $s_t = s$, pero a diferencia de la función $V$, aquí se fija también la acción inicial $a_t = a$. Es decir, se asume que el agente **elige explícitamente** la acción $a$ en ese estado. A partir de ahí, el resto del comportamiento se determina por la política $\pi$, y la esperanza se calcula como el promedio sobre todas las trayectorias posibles inducidas por esta.

El término $\sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}$ sigue siendo el retorno acumulado, afectado por el factor de descuento $\gamma$, que penaliza las recompensas lejanas. La esperanza $\mathbb{E}_\pi[\cdot]$ tiene en cuenta tanto la estocasticidad de las transiciones del entorno como la posible aleatoriedad de la política a partir del segundo paso.

Esta función será crucial no solo para evaluar políticas, sino también para **mejorarlas**, ya que permite al agente identificar qué acciones tienen mayor potencial en cada estado. En las próximas secciones exploraremos cómo estas funciones se relacionan entre sí y cómo se utilizan para construir algoritmos de planificación y aprendizaje.

#### **Relación entre la función estado-valor y la función acción-valor**

Las funciones $v_\pi(s)$ y $q_\pi(s, a)$ no son independientes. Al contrario, existe una relación estrecha entre ellas que refleja la estructura misma del proceso de decisión. Comprender esta relación es clave para diseñar algoritmos que permitan evaluar y mejorar políticas de manera eficiente.

Imaginemos de nuevo el tablero 4x4 y centrémonos en el estado $s_{13}$. Por un lado, la función estado-valor $v_\pi(s_{13})$ nos dice, en promedio, cuánto vale estar en esa casilla si a partir de ahí el agente sigue la política $\pi$. Por otro lado, la función acción-valor $q_\pi(s_{13}, a)$ nos dice cuánto vale tomar una acción concreta, digamos moverse a la derecha ($a_2$), y luego continuar con $\pi$. La conexión entre ambas es inmediata: **el valor de estar en un estado no es más que el promedio de los valores de las acciones posibles en ese estado, ponderado por la probabilidad de que la política elija cada una de ellas**.

En otras palabras, si el agente se encuentra en el estado $s$, puede optar por diferentes acciones según su política $\pi$. Cada una de esas acciones tiene asociado un valor $q_\pi(s, a)$. El valor global del estado $v_\pi(s)$ es simplemente la media de esos valores, donde los pesos son las probabilidades $\pi(a|s)$ que la política asigna a cada acción. Esta relación se expresa mediante la siguiente ecuación:

$$
v_\pi(s) = \sum_{a} \pi(a|s) \cdot q_\pi(s, a)\label{relacion_v_q}
$$

Volvamos al ejemplo del tablero. Supongamos que en el estado $s_{13}$ la política $\pi$ del agente le lleva a elegir la acción de moverse a la derecha ($a_2$) con una probabilidad del 80% y la acción de moverse hacia arriba ($a_3$) con un 20%. Si conociéramos los valores $q_\pi(s_{13}, a_2)$ y $q_\pi(s_{13}, a_3)$, podríamos calcular el valor del estado $s_{13}$ como $0.8 \cdot q_\pi(s_{13}, a_2) + 0.2 \cdot q_\pi(s_{13}, a_3)$. Así, el valor de un estado es, en esencia, la esperanza de los valores de las acciones que se pueden tomar en él.

En resumen, $v_\pi$ y $q_\pi$ son dos caras de la misma moneda. Conocer $q_\pi$ permite obtener $v_\pi$ promediando sobre acciones. Esta interrelación es la base de muchos algoritmos  que veremos en el resto del tema y también subyace en los métodos de aprendizaje por refuerzo, donde se estiman estas funciones a partir de la experiencia.

En las siguientes secciones veremos cómo aprovechar estas relaciones para calcular de manera iterativa la política óptima, ya sea conociendo el modelo del entorno (programación dinámica) o aprendiendo directamente de la interacción (aprendizaje por refuerzo).

#### **De la interacción al razonamiento: hacia las ecuaciones de Bellman**

A estas alturas, hemos formalizado todos los componentes clave que permiten representar cómo un agente interactúa con un entorno incierto en el marco de los MDP. El agente observa un estado $s_t$, elige una acción $a_t$, recibe una recompensa $r_{t+1}$ y transita a un nuevo estado $s_{t+1}$. Este ciclo se repite a lo largo del tiempo, generando una trayectoria.

Para orientar su comportamiento, el agente sigue una política $\pi$, y evalúa su rendimiento en términos del retorno total esperado. Hemos introducido dos funciones fundamentales que permiten cuantificar este retorno:

- La **función estado-valor** $v_\pi(s)$, que mide cuán buena es una posición inicial $s$ al seguir la política $\pi$.

- La **función acción-valor** $q_\pi(s, a)$, que estima el retorno esperado si se toma una acción concreta $a$ en el estado $s$ y luego se sigue la política.

  Estas funciones describen el rendimiento esperado de la política, pero **no son conocidas de antemano**. Para poder utilizarlas en algoritmos de planificación o mejora de políticas, necesitamos una manera de **calcularlas o aproximarlas**.

  Este es precisamente el objetivo de las **ecuaciones de Bellman**: establecer **relaciones recursivas** entre los valores de los estados y acciones, de forma que podamos descomponer el problema global de evaluación de una política en **subproblemas locales** que conectan los estados actuales con sus sucesores.

##### **Punto de partida: la definición del valor esperado**

Recordemos las expresiones que definen formalmente las funciones de valor:

$$
v_\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k , R_{t+k+1} | S_t = s \right]
$$

$$
q_\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k , R_{t+k+1} | S_t = s, A_t = a \right]
$$

Ambas funciones miden el **retorno futuro esperado**, pero lo hacen agregando todas las recompensas futuras en una suma infinita. Este tipo de definición no es práctica para el cálculo directo, ya que **depende de toda la trayectoria completa**.

La idea de Bellman fue proponer un enfoque recursivo: dado que el retorno $G_t$​ es una suma, podemos expresar la recompensa acumulada **como una recompensa inmediata más el retorno futuro**. Es decir, si retorno $G_t$ se definia como **la suma de las recompensas futuras**, progresivamente descontadas por el factor $\gamma$:
$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
\label{retorno}
$$

podríamos expresar lo anterior así:
$$
G_t = R_{t+1} + \gamma \, G_{t+1}\label{G_recursivo}
$$

Esta simple identidad es la clave para construir una **relación de consistencia interna** para las funciones de valor. A partir de aquí, se derivan las **ecuaciones de Bellman**, que permiten calcular el valor de un estado en función de sus estados sucesores.

##### **Primera ecuación de Bellman: función estado-valor**

La **ecuación de Bellman para $v_\pi(s)$** nos dice que el valor de un estado bajo una política $\pi$ es igual al **valor esperado de la recompensa inmediata más el valor descontado del siguiente estado**, considerando que el agente sigue la política $\pi$. Se expresa así:

$$
v_π(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \bigl[ r + γ \, v_π(s') \bigr]
$$
Vamos a ver qué expresa cada componente de la ecuación anterior:

- $\pi(a \mid s)$: probabilidad de que la política $\pi$ seleccione la acción $a$ cuando el agente se encuentra en el estado $s$.

- $p(s', r \mid s, a)$: probabilidad conjunta de que, tras ejecutar la acción $a$ en el estado $s$, el entorno transite al estado $s'$ y se reciba una recompensa $r$. Esta función captura tanto la dinámica de transición como la distribución de recompensas.

- $r$: es la recompensa inmediata que se obtiene en esa transición concreta (no una esperanza). Aparece dentro del sumatorio porque estamos promediando sobre todos los posibles valores de $r$.

- $\gamma \, v_\pi(s')$: valor descontado del estado siguiente $s'$. $v_\pi(s')$ es el valor esperado del retorno desde $s'$ siguiendo la política $\pi$, y $\gamma$ descuenta ese valor porque se recibe un paso más tarde.

  Esta ecuación es un promedio ponderado: primero por la política del agente (qué acciones ejecuta) y luego por la estocasticidad del entorno (qué estados se alcanzan). La ecuación de Bellman **expresa el valor de un estado como una combinación de valores de sus estados sucesores**, lo que permite construir algoritmos iterativos para resolverla.

  Es interesante ver como puede obtener la ecuación anterior partiendo de las definiciones del retorno esperado ($G_t$) y la funcion estado valor ($v_\pi(s)$). En efecto: La función de valor de un estado bajo una política $\pi$ se definía como el **valor esperado del retorno al empezar en ese estado y seguir la política $\pi$**:

$$
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]\label{definicion_v}
$$

Esta función nos informa acerca de la recompensa final que se espera si el agente actúa conforme a una determinada política. Cuanto mayor sea $v_\pi(s)$, más interesante era partir de ese estado $s$.

Para ello volvamos a la expresión de $v_\pi(s)$ pero introduciendo la forma del retorno vista en la ecuacion ($\ref{G_recursivo}$)
$$
v_π(s) = \mathbb{E}_π[ R_{t+1} + γ G_{t+1} \mid S_t = s ]
$$
Teniendo en cuenta que la esperanza matematica es una funcion lineal, podríamos separar la suma así:

$$
v_π(s) = \underbrace{\mathbb{E}_π[ R_{t+1} \mid S_t = s ]}_{\text{(A)}} + γ \underbrace{\mathbb{E}_π[ G_{t+1} \mid S_t = s ]}_{\text{(B)}}\label{v_A_B}
$$
Centrémonos primero en el término (A). La recompensa inmediata $R_{t+1}$ va a depender de dos cosas: Primero, de la acción $A_t$ que se toma en $S_t$, lo que viene dado por la política del agente, $\pi(A_t=a|S_t=s)$; y segundo, del modelo de la transición que proporciona el entorno, $p(S_{t+1}=s' \mid S_t=s, A_t=a)$. Para calcular la esperanza, debemos promediar sobre todas las acciones posibles (según la política π) y sobre todos los posibles resultados de la transición ($p$) (siguiente estado $s'$ y recompensa $r$). Recuerda que:

- La política π da la probabilidad de elegir cada acción $a$: $\pi(a|s)$.

- El entorno, dado $s$ y $a$, produce una transición a $s'$ con recompensa $r$ con probabilidad $p(s', r \mid s, a)$.

  Por tanto:

$$
\mathbb{E}_π[ R_{t+1} \mid S_t = s ] = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \; r
$$

Observa que el sumatorio anterior sobre $r$ recorre todos los valores posibles de la recompensa; el término $r$ dentro del sumatorio es el valor numérico de esa recompensa.

Vamos ahora con el termino (B). Aqui, $\mathbb{E}_π[ G_{t+1} \mid S_t = s ]$ es la esperanza del retorno **a partir del siguiente paso**, condicionada solo al estado actual. Para relacionarlo con el valor del estado en $t+1$, debemos condicionar también a la acción $A_t$ y al siguiente estado $S_{t+1}$. Para ello se puede aplicar la ley de la esperanza total haciendo un condicionamiento en etapas:

1. Primero condicionamos a la acción $A_t = a$ (con probabilidad $\pi(a|s)$).

2. Dado $a$, el entorno transita a $S_{t+1}=s'$ con recompensa $r$ con probabilidad $p(s', r \mid s, a)$.

3. Una vez conocido $S_{t+1}=s'$, el retorno futuro $G_{t+1}$ no depende de cómo se llegó allí (ya que estamos suponiendo un proceso de Markov), y su esperanza es precisamente el valor del estado $s'$ bajo la política π: $v_π(s')$.

   Así:

$$
\mathbb{E}_π[ G_{t+1} \mid S_t = s ] = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \; \mathbb{E}_π[ G_{t+1} \mid S_{t+1}=s' ]
$$

Y ya tendríamos que la esperanza a partir del instante $t+1$ **dado que $S_t=s$** expresada en funcion de la esperanza en dicho $t+1$ pero dado un $S_{t+1}=s'$ .  

Ahora bien, como por definición de vimos que
$$
v_π(s') = \mathbb{E}_π[ G_{t+1} \mid S_{t+1}=s' ]
$$
tendríamos entonces que:
$$
\mathbb{E}_π[ G_{t+1} \mid S_t = s ] = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \; v_π(s')
$$

Por último, si unimos los términos tal y como teníamos en $(\ref{v_A_B})$, nos quedaría que:
$$
v_π(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \bigl[ r + γ \, v_π(s') \bigr]
$$
Y ya tendríamos la **ecuación de Bellman para la función valor de estado** . Vemos como se relaciona el valor de un estado con la recompensa inmediata esperada más el valor descontado del siguiente estado, promediado sobre todas las acciones según la política π y sobre todas las transiciones posibles del entorno $p$.

**Segunda ecuación de Bellman: función estado-acción**

Igualmente podríamos obtener un resultado análogo para la funcion de valor estado-acción. Basta con partir de la expresion $(\ref{relacion_v_q})$ y por comparación tendríamos:
$$
q_π(s,a) = \sum_{s', r} p(s', r \mid s, a) \bigl[ r + γ \sum_{a'} \pi(a'|s') q_π(s', a') \bigr]
$$
Esta expresión constituye la **segunda ecuación de Bellman** para la funcion estado-acción. Los componentes de esta ecuación se pueden describir del mismo modo que en el caso de la primera ecuación de Bellman

En este caso, el agente ya ha ejecutado la acción $a$ en el estado $s$, y a partir de $s'$ continúa siguiendo su política $\pi$. El valor de $q_\pi(s, a)$ se construye como una combinación de las decisiones futuras posibles desde $s'$, teniendo en cuenta el comportamiento inducido por la política.

> Estas ecuaciones constituyen el **núcleo computacional del aprendizaje por refuerzo**. Son las herramientas que permiten:
>
> - Evaluar una política **sin necesidad de simular episodios completos.**
>
> - Propagar información desde los estados sucesores hacia los anteriores.
>
> - Construir algoritmos como **iteración de valores**, **mejora de políticas** o **Q-learning**, que resolveremos en próximos módulos.
>
>   Pero, sobre todo, las ecuaciones de Bellman revelan una idea central: **el valor de un estado o acción no se define de forma aislada, sino en función de los estados futuros que se pueden alcanzar desde él**. Esta conexión entre presente y futuro es lo que hace del aprendizaje por refuerzo una disciplina secuencial y predictiva.



### Clasificación de los algoritmos de aprendizaje por refuerzo

Hasta ahora hemos establecido el lenguaje común de los Procesos de Decisión de Markov (MDP) y la ecuación de Bellman. Sin embargo, si todo parte de la misma ecuación, ¿por qué existen decenas de algoritmos aparentemente diferentes? La respuesta es que cada familia de algoritmos responde a una pregunta distinta: ¿conocemos el modelo del entorno o tenemos que aprender de la experiencia? ¿Podemos esperar a que termine un episodio o necesitamos aprender en cada paso? ¿La política que usamos para explorar es la misma que queremos mejorar? Estas tres decisiones dan lugar a cuatro grandes familias que, como ramas de un árbol, comparten el mismo tronco pero se orientan hacia problemas distintos.

#### El papel del modelo del entorno: planificación frente a aprendizaje

Una primera forma de clasificar los algoritmos se basa en el grado de conocimiento que el agente tiene sobre el entorno. Este tipo de conocimiento —es decir, disponer de la función de transición $p(s' \mid s,a)$ y de la función de recompensa esperada $R(s,a,s')$— proporciona al agente una visión completa de la dinámica del entorno. Cuando se conoce el modelo, el agente no necesita aprender exclusivamente a partir de la experiencia directa, sino que puede **anticipar mentalmente** las consecuencias de sus acciones mediante simulación interna. A este proceso se le denomina **planificación**, y consiste en evaluar y seleccionar acciones mediante razonamiento sobre el modelo, sin necesidad de ejecutar dichas acciones en el entorno real.

Por ejemplo, si el agente conoce que al ejecutar la acción $a$ en el estado $s$ tiene un 80% de probabilidad de llegar al estado $s'$ con una recompensa asociada, puede calcular de antemano el valor esperado de esa acción y compararlo con otras opciones sin necesidad de probarlas. Esta capacidad de simulación permite optimizar el comportamiento del agente sin requerir una gran cantidad de episodios de interacción, lo que puede ser especialmente importante en entornos donde cada prueba es costosa, arriesgada o limitada.

> **Un robot** que debe cruzar un puente inestable, con el modelo de probabilidades de derrumbe, podría calcular el valor esperado de cada acción sin arriesgarse a caer. No necesita cien episodios de prueba y error; planifica *offline*. Así evita costes, desgaste y accidentes. En entornos donde un fallo es caro (quirófano, central nuclear, inversión financiera), simular es más seguro que aprender por las malas.

Los algoritmos que se basan en esta idea reciben el nombre de **model‑based**, ya que utilizan un modelo explícito del entorno como entrada para sus cálculos. El uso de modelos permite resolver de forma más o menos exacta las ecuaciones de Bellman, que vinculan el valor de los estados con el de sus sucesores. A partir de estas ecuaciones, el agente puede construir o mejorar políticas de actuación mediante técnicas de planificación iterativa.

Sin embargo, en el mundo real es raro encontrar escenarios en los que el agente tenga acceso explícito al modelo. Esto significa que no conoce de antemano las reglas que rigen la dinámica de las transiciones entre estados ni las recompensas que pueden obtenerse. En otras palabras, las funciones $p(s' \mid s, a)$ y $R(s, a, s')$ son aquí desconocidas o inobservables directamente. Esta situación es muy común en entornos complejos, dinámicos o parcialmente observables, como la interacción con usuarios humanos, la navegación en el mundo físico o la toma de decisiones en sistemas donde los mecanismos internos no son accesibles.

En estos casos se recurre a algoritmos **model‑free**, que prescinden de cualquier conocimiento previo sobre el modelo del entorno. En lugar de razonar a partir de un modelo simbólico, el agente aprende directamente de su experiencia, es decir, de las secuencias de interacción que recoge al actuar: $(s_t, a_t, r_{t+1}, s_{t+1})$. A partir de estas transiciones observadas, el agente intenta aproximar funciones de valor y encontrar políticas cada vez mejores sin necesidad de conocer la distribución de probabilidad de las transiciones ni la recompensa esperada exacta.

Este enfoque conlleva una diferencia fundamental: el agente no puede planificar internamente sus acciones mediante simulaciones, porque no dispone de un modelo que le permita hacerlo. En su lugar, debe interactuar con el entorno real para recopilar información, explorar diferentes posibilidades y corregir sus decisiones basándose en los resultados obtenidos. Este proceso requiere técnicas específicas para equilibrar la exploración con la explotación, gestionar la incertidumbre y adaptar el comportamiento de forma progresiva.

> **¿Pueden considerarse los métodos model‑based como RL?**
>
> La respuesta depende de cómo definamos “aprendizaje”. Si entendemos RL como aprender de la experiencia a través de la interacción, entonces las técnicas model‑based no deberían ser consideradas RL. Cuando conocemos el modelo no es necesaria la interacción del agente; no se prueban acciones, no hay retroalimentación del entorno. Solo existe una planificación previa con un modelo conocido. No hay “refuerzo” porque no hay ensayo ni error, solo simulación.
>
> Sin embargo, si consideramos RL como cualquier método que resuelve MDPs, entonces los métodos model‑based sí suelen incluirse en los libros de texto (como el de Sutton & Barto) por dos razones: primero, hacen uso de la ecuación de Bellman para resolver el problema, y segundo, el objetivo final —evaluar políticas o encontrar la óptima— es compartido con el resto de técnicas de RL que no se basan en el conocimiento previo del modelo. 
>
> En cualquier caso, es importante tener clara la distinción clave: el origen de la información. En las técnicas model‑based se utilizan expectativas basadas en simulaciones; en las técnicas model‑free se emplean muestras reales.

#### Aprendizaje por episodios completos o paso a paso

Dentro de los métodos **model‑free**, surge una nueva distinción. Supongamos que estamos enseñando a un robot a salir de un laberinto. Cada intento contado desde la entrada hasta la salida (o hasta que se queda sin batería) es un **episodio**. Una forma natural de aprender es esperar a que termine el episodio, observar qué recompensa total ha obtenido y luego ajustar las valoraciones de los estados que se visitaron. Esto es lo que hacen los **métodos de Monte Carlo**. Son intuitivos, pero requieren que los episodios terminen y no aprenden nada hasta el final. Si el laberinto es enorme y el robot tarda mucho en llegar a la meta, el aprendizaje se vuelve muy lento.

¿Y si pudiéramos aprender después de cada paso, sin esperar al final? Esa es la idea de los **métodos de diferencia temporal (TD)**. En lugar de usar la recompensa total del episodio, usan la recompensa inmediata más el valor estimado del siguiente estado. Así, el agente puede actualizar sus creencias en cada transición, incluso antes de saber si llegará a la meta. Esta capacidad de **aprender en línea** es una de las grandes ventajas de los métodos TD sobre los de Monte Carlo.

#### Aprendizaje on‑policy vs. off‑policy

Dentro de los **métodos de diferencia temporal (TD)**, nos encontramos con una decisión de diseño fundamental: ¿cómo gestionamos la exploración mientras aprendemos la política óptima?

Imaginemos que un agente está aprendiendo a navegar por un laberinto. Para descubrir la mejor ruta, debe probar caminos nuevos (explorar), pero también quiere aprovechar lo que ya sabe que funciona (explotar). La política que realmente usa para moverse por el laberinto es la **política de comportamiento**. Por otro lado, la política que el agente quiere aprender al final, la que le llevaría al objetivo de la forma más rápida, es la **política objetivo**.

Aquí surge la cuestión: ¿puede el agente aprender la política objetivo mientras se comporta con una política de comportamiento diferente (más exploratoria)? La respuesta divide a los algoritmos en dos grandes familias.

**On‑policy (sobre la política)**: algoritmos como **SARSA** aprenden el valor de la **misma política que están ejecutando**. Si el agente está explorando (tomando caminos aleatorios de vez en cuando), SARSA aprenderá el valor de esa política exploratoria, con todas sus desviaciones. Es como aprender a conducir teniendo en cuenta que a veces podemos equivocarnos o que hay tráfico; nuestra valoración refleja la realidad con la que realmente nos encontramos. Esto hace que SARSA sea **estable y conservador**, pero puede llevarle a aprender una política que no sea la óptima si la exploración nunca cesa.

**Off‑policy (fuera de la política)**: algoritmos como **Q‑learning** son más ambiciosos. Pueden aprender el valor de la **política óptima** mientras se comportan de forma exploratoria. Es como si un navegador GPS te indicara la ruta ideal (sin atascos, sin desvíos) mientras tú realmente estás haciendo paradas o tomando caminos alternativos. Q‑learning actualiza sus estimaciones suponiendo que, desde el siguiente estado, el agente siempre elegirá la mejor acción posible, independientemente de la acción que vaya a tomar realmente. Esto permite aprender la política óptima incluso mientras se explora, lo que hace a Q‑learning más **flexible y potente**, pero también potencialmente más **inestable**, porque las estimaciones pueden saltar si la exploración es muy ruidosa.

> **On‑policy** aprende sobre lo que realmente haces; **off‑policy** aprende sobre lo que deberías hacer idealmente. Ambos enfoques son útiles, pero la elección depende de si se prefiere estabilidad (SARSA) o la capacidad de aprender la política óptima con exploración (Q‑learning).

#### Aprendizaje basado en valor vs. aprendizaje directo de políticas

Hasta ahora, todos los algoritmos que hemos visto comparten una misma filosofía: primero aprenden cuánto vale cada estado o cada acción, es decir, la función de valor, y luego, a partir de ese conocimiento, deciden qué hacer. Es como si un estudiante, antes de elegir qué carrera estudiar, se dedicara a calcular el salario esperado de cada opción; una vez que tiene los números, elige la que da mayor beneficio. Estos son los métodos basados en valor.

Pero existe otra forma de pensar: ¿y si el agente aprendiera directamente qué acción tomar en cada situación, sin pasar por la estimación intermedia de cuánto vale cada cosa? Esa es la idea de los métodos de gradiente de política. En lugar de construir una tabla de valores, estos algoritmos parametrizan directamente la política (por ejemplo, con una red neuronal) y la van ajustando para maximizar la recompensa esperada. Es como si el estudiante, en lugar de calcular salarios, probara diferentes carreras, viera cuál le gusta más y ajustara su estrategia de elección directamente, sin necesidad de asignar números concretos a cada opción.

¿Cuándo son especialmente útiles estos métodos? En primer lugar, cuando el espacio de acciones es continuo. Imagina un brazo robótico que debe aplicar una fuerza concreta, por ejemplo 3,27 Newtons. No podemos enumerar todas las fuerzas posibles porque serían infinitas; los métodos basados en valor tienen problemas con acciones continuas porque necesitarían una tabla infinita, mientras que los métodos de gradiente de política pueden aprender directamente a producir la fuerza adecuada como salida de una función continua. En segundo lugar, cuando la política óptima es estocástica, como en el póker, donde a veces la mejor estrategia no es hacer siempre lo mismo, sino actuar con cierta aleatoriedad (por ejemplo, farolear el 30% de las veces). Los métodos basados en valor tienden a producir políticas deterministas, mientras que los métodos de gradiente de política pueden aprender explícitamente distribuciones de probabilidad sobre las acciones.

Un ejemplo concreto ayuda a entender la diferencia. En un método basado en valor como Q‑learning, el agente aprende que en el estado "tengo un par de ases" la acción "subir la apuesta" tiene un valor de 100; por tanto, cuando esté en ese estado, subirá la apuesta siempre (política determinista). En un método de gradiente de política como REINFORCE, el agente aprende directamente que en ese mismo estado debe subir la apuesta con probabilidad 0,8 y pasar con probabilidad 0,2; no sabe exactamente cuánto "vale" cada acción, solo la probabilidad de elegirla.

Estos métodos han ganado mucha popularidad en los últimos años, especialmente combinados con redes neuronales profundas, dando lugar a algoritmos muy potentes como PPO (Proximal Policy Optimization) o A3C (Asynchronous Advantage Actor‑Critic), que han sido clave en éxitos como el entrenamiento de robots o la victoria en juegos complejos. Su principal ventaja es que pueden manejar problemas donde las acciones no son discretas o donde la aleatoriedad es parte de la estrategia óptima. Su desventaja es que suelen necesitar más interacciones con el entorno para converger y son más sensibles a los hiperparámetros.

> Los métodos basados en valor aprenden un mapa de lo bueno que es cada sitio, mientras que los métodos de gradiente de política aprenden directamente el camino a seguir. Ambos son herramientas útiles; la elección depende de si tenemos acciones discretas o continuas, y de si la mejor política es determinista o estocástica.

#### Predicción frente a control: dos objetivos fundamentales

Otra distinción relevante en el estudio de algoritmos de RL tiene que ver con el objetivo que se persigue en cada caso. En algunas ocasiones, el interés se centra únicamente en evaluar el comportamiento de una política fija, es decir, en estimar cuánto retorno se puede esperar si se siguen siempre las mismas decisiones. Esta tarea recibe el nombre de **predicción**, y da lugar a algoritmos que calculan o aproximan las funciones de valor $v_\pi(s)$ o $q_\pi(s,a)$ para una política dada. La predicción es útil, por ejemplo, para evaluar soluciones predefinidas o como paso intermedio en métodos de mejora.

Sin embargo, el objetivo más habitual en aprendizaje por refuerzo es el de **control**, es decir, encontrar una política que sea óptima o, al menos, mejor que las anteriores. En estos casos, el agente debe combinar la estimación de valores con la mejora de la política, de forma iterativa. El control requiere, por tanto, métodos que no solo evalúan, sino que también ajustan las decisiones para maximizar el retorno esperado. Este proceso suele implicar mecanismos de exploración, mejora de política y actualización continua.

#### Conclusiones

Dos de las dimensiones presentadas en el apartado anterior —el conocimiento del modelo (model‑based frente a model‑free) y el objetivo de la tarea (predicción frente a control)— permiten construir una clasificación cruzada que organiza los algoritmos fundamentales del aprendizaje por refuerzo. La siguiente tabla resume esta tipología:

|                 | **Predicción** (evaluar una política fija)                   | **Control** (encontrar la política óptima)                   |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Model‑based** | Evaluación de políticas mediante planificación exacta. <br>Ej.: resolución de las ecuaciones de Bellman con modelo conocido. | Planificación iterativa con mejora de política. <br>Ej.: iteración de valor, iteración de política. |
| **Model‑free**  | Estimación de valores a partir de experiencia. <br>Ej.: métodos de Monte Carlo y diferencias temporales (TD(0)). | Aprendizaje de políticas óptimas a partir de interacción. <br>Ej.: Q‑learning, SARSA. |

Cada celda de la tabla agrupa una familia de algoritmos que comparten la misma respuesta a las dos preguntas fundamentales: ¿conocemos el modelo? y ¿queremos predecir o controlar? 

Si el agente dispone de un modelo completo del entorno, puede planificar sin necesidad de interactuar. En el caso de la **predicción**, se limita a evaluar una política fija resolviendo las ecuaciones de Bellman de forma exacta. En el caso del **control** (celda superior derecha), la programación dinámica combina evaluación y mejora iterativa de la política para encontrar la política óptima; los ejemplos más representativos son la iteración de valores o la iteración de políticas.

Cuando el modelo es desconocido, el agente debe aprender de la experiencia. Para la **predicción**, los métodos de Monte Carlo estiman valores promediando retornos de episodios completos, mientras que los métodos de diferencia temporal (TD) como TD(0) lo hacen paso a paso, combinando muestreo con bootstrapping. Para el **control**, surgen dos grandes algoritmos: SARSA, que es on‑policy y aprende el valor de la política que realmente se ejecuta (con toda su exploración), y Q‑learning, que es off‑policy y aprende directamente la política óptima mientras se comporta de forma exploratoria.

Además de lo dicho, no olvidemos que existe una cuarta familia que responde a una pregunta diferente: ¿y si aprendemos directamente la política, sin pasar por la función de valor? Son los **métodos de gradiente de política**, que parametrizan la política y la ajustan para maximizar la recompensa esperada. Esta familia es especialmente útil cuando el espacio de acciones es continuo (por ejemplo, la fuerza que debe aplicar un brazo robótico) o cuando la política óptima es estocástica (como en el póker, donde a veces conviene farolear). Al igual que Q‑learning y SARSA, estos métodos son model‑free y se utilizan para control; constituyen una alternativa a los métodos basados en valor, no una categoría ortogonal a las dos dimensiones de la tabla.

Y para finalizar lo más importante: No existe un “mejor algoritmo” universal. La elección depende de si disponemos de modelo, de si el problema es episódico o continuo, de si podemos permitirnos explorar sin miedo, del tamaño del espacio de estados y acciones, o de si la política óptima es determinista o estocástica. El aprendizaje por refuerzo es un campo de compromisos, y cada algoritmo representa un punto de equilibrio distinto en ese espacio de diseño. En los capítulos siguientes desgranaremos algunas de estas familias con más detalle: sus ecuaciones, sus algoritmos emblemáticos y los problemas para los que son más adecuadas.
