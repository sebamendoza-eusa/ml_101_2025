# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Algoritmos de Aprendizaje por Refuerzo

### Objetivos del módulo

> - Comprender las condiciones bajo las cuales los algoritmos clásicos pueden aplicarse para resolver MDPs.
> - Estudiar los algoritmos de programación dinámica y su relación con el modelo del entorno.
> - Entender los métodos de Monte Carlo como forma de estimar funciones de valor a partir de experiencia completa.
> - Analizar los métodos basados en diferencias temporales (TD) como combinación de Monte Carlo y actualización incremental.
> - Comparar los tres enfoques en términos de requisitos, eficiencia, precisión y aplicabilidad.
> - Aplicar cada enfoque a entornos sencillos mediante ejemplos y simulaciones en Python.

### Introducción

Tal como hemos visto en el módulo anterior, el aprendizaje por refuerzo se basa en la formalización de la interacción agente-entorno mediante procesos de decisión de Markov (MDP).

Un **Proceso de Decisión de Markov (MDP)** se definía formalmente como una 5-tupla:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

donde:

- $\mathcal{S}$ es el conjunto de **estados** posibles.
- $\mathcal{A}$ es el conjunto de **acciones** disponibles para el agente.
- $\mathcal{P}(s' \mid s, a)$ es la **función de transición**, que da la probabilidad de pasar al estado $s'$ al ejecutar la acción $a$ en el estado $s$.
- $\mathcal{R}(s, a, s')$ es la **función de recompensa**, que devuelve el valor esperado de la recompensa al realizar la acción $a$ en el estado $s$ y transitar a $s'$.
- $\gamma \in [0,1)$ es el **factor de descuento**, que determina el peso relativo de las recompensas futuras frente a las inmediatas.

Esta formalización proporciona una base matemática clara para definir los elementos del problema: el espacio de estados, el conjunto de acciones, la función de transición, la función de recompensa y el objetivo del agente. 

#### El rol de la función de valor

En el marco de la estructura del MDP el agente busca **maximizar el retorno acumulado esperado** a través de su interacción con el entorno. Para lograrlo, necesita una forma de **evaluar cuán deseables son los distintos estados y acciones**. Este es precisamente el papel que cumplen las **funciones de valor**.

Recordemos que el retorno $G_t$ se define como **la suma de las recompensas futuras**, posiblemente descontadas:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Así, la función de valor de un estado bajo una política $\pi$ se definía como el **valor esperado del retorno al empezar en ese estado y seguir la política $\pi$**:

$$
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]
$$

Esta función captura, por tanto, la **bondad esperada de un estado** si el agente actúa conforme a una determinada política. Cuanto mayor sea $v_\pi(s)$, más prometedor es estar en el estado $s$.

De forma similar, la **función acción-valor** se define como el valor esperado de tomar una acción $a$ en el estado $s$ y continuar después siguiendo la política $\pi$:

$$
q_\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

Estas funciones no solo sirven para evaluar decisiones pasadas, sino que son el núcleo de los algoritmos de optimización de políticas. De hecho, **una política es óptima si, en cada estado, selecciona siempre la acción que maximiza la función acción-valor**.

Por tanto, en todos los enfoques clásicos de RL (ya sea programación dinámica, Monte Carlo o TD), el primer paso fundamental es **estimar correctamente las funciones de valor**. A partir de estas estimaciones se pueden derivar políticas mejores, comparar estrategias, o construir modelos de decisión que guíen el comportamiento futuro del agente.

En resumen, la función de valor cumple una doble función:

1. **Evaluar el comportamiento** de una política dada (predicción).
2. **Guiar la mejora** de la política en algoritmos de control.

Este papel central es el motivo por el cual muchas técnicas de RL —especialmente las clásicas— giran en torno al **cálculo, estimación y actualización de funciones de valor**.

#### Algoritmos *model-based* vs. *model-free*

Una primera forma de clasificar estos algoritmos se basa en el grado de conocimiento que el agente tiene sobre el entorno. Este tipo de conocimiento —es decir, disponer de la función de transición $\mathcal{P}(s' \mid s,a)$ y de la función de recompensa esperada $\mathcal{R}(s,a,s')$— proporciona al agente una visión completa de la dinámica del entorno. Cuando se conoce el modelo, el agente no necesita aprender exclusivamente a partir de la experiencia directa, sino que puede **anticipar mentalmente** las consecuencias de sus acciones mediante simulación interna. A este proceso se le denomina **planificación**, y consiste en **evaluar y seleccionar acciones mediante razonamiento sobre el modelo**, sin necesidad de ejecutar dichas acciones en el entorno real.

Por ejemplo, si el agente conoce que al ejecutar la acción $a$ en el estado $s$ tiene un 80% de probabilidad de llegar al estado $s'$ con una recompensa asociada, puede calcular de antemano el valor esperado de esa acción y compararlo con otras opciones sin necesidad de probarlas. Esta capacidad de simulación permite **optimizar el comportamiento del agente sin requerir una gran cantidad de episodios de interacción**, lo que puede ser especialmente importante en entornos donde cada prueba es costosa, riesgosa o limitada.

Los algoritmos que se basan en esta idea reciben el nombre de **model-based**, ya que utilizan un **modelo explícito del entorno** como entrada para sus cálculos. El uso de modelos permite resolver de forma más o menos exacta las **ecuaciones de Bellman**, que vinculan el valor de los estados con el de sus sucesores. A partir de estas ecuaciones, el agente puede construir o mejorar políticas de actuación mediante **técnicas de planificación iterativa**.

Entre las técnicas más representativas dentro de este enfoque encontramos la **iteración de valor** y la **iteración de política**. En la iteración de valor, se actualizan de forma repetida los valores de los estados hasta alcanzar una convergencia que refleje el comportamiento óptimo bajo el modelo. En la iteración de política, el proceso alterna entre una fase de evaluación de la política actual y una fase de mejora de dicha política, utilizando el modelo para ambas etapas.

Este enfoque basado en modelo es particularmente útil cuando el entorno es completamente conocido, o cuando existe la posibilidad de construir un modelo fiable a partir de datos, como ocurre en entornos simulados, videojuegos, o sistemas con dinámica física bien caracterizada. En estos casos, los algoritmos model-based permiten resolver el problema de toma de decisiones de forma precisa y eficiente, sin requerir exploración extensiva en el entorno real.

> **Ejemplo:** ¿Cómo funcionan los algoritmos **de iteración de valor** e **iteración de política**?
>
> Consideramos un entorno con tres estados: $s_0$, $s_1$ y $s_2$. El estado $s_2$ es absorbente (no se sale de él). El agente puede ejecutar solo una acción en cada estado y la transición es determinista. Las transiciones y recompensas son las siguientes:
>
> - Desde $s_0$, se transita a $s_1$ con recompensa 0.
> - Desde $s_1$, se transita a $s_2$ con recompensa 1.
> - Desde $s_2$, se permanece en $s_2$ con recompensa 0.
>
> Usamos un factor de descuento $\gamma = 1$ y comenzamos con valores iniciales $V(s) = 0$ para todos los estados. La ecuación de Bellman para este caso es:
>
> $$
> V(s) = \mathcal{R}(s) + \gamma \cdot V(s')
> $$
>
> Aplicamos iteración de valor:
>
> **Iteración 1:**
>
> - $V(s_2) = 0$ (estado absorbente)
> - $V(s_1) = 1 + V(s_2) = 1$
> - $V(s_0) = 0 + V(s_1) = 1$
>
> **Iteración 2:**
>
> - $V(s_2) = 0$
> - $V(s_1) = 1 + V(s_2) = 1$
> - $V(s_0) = 0 + V(s_1) = 1$
>
> Los valores se han estabilizado. El algoritmo ha convergido y proporciona una evaluación consistente de cada estado.
>
> Apliquemos ahora un razonamiento similar para ver cómo funciona la iteración de política. Utilizaremos el mismo entorno, pero ahora el agente tiene **dos acciones** en $s_0$ y $s_1$:
>
> - En $s_0$:  
>   - Acción $a_0$ lleva a $s_1$ con recompensa 0.  
>   - Acción $a_1$ lleva a $s_2$ con recompensa 0.
>
> - En $s_1$:  
>   - Acción $a_0$ lleva a $s_2$ con recompensa 1.  
>   - Acción $a_1$ lleva a $s_2$ con recompensa 0.
>
> Supongamos una política inicial $\pi$ que selecciona $a_1$ en todos los estados. Inicializamos $V(s) = 0$.
>
> **Paso 1: Evaluación de política**
>
> Usamos la política para calcular $V^\pi$ resolviendo las siguientes ecuaciones:
>
> - $V(s_1) = \mathcal{R}(s_1, a_1) + \gamma V(s_2) = 0$
> - $V(s_0) = \mathcal{R}(s_0, a_1) + \gamma V(s_2) = 0$
> - $V(s_2) = 0$
>
> **Paso 2: Mejora de política**
>
> Ahora revisamos si hay alguna acción mejor en cada estado evaluando todas las posibles:
>
> - En $s_1$:  
>   - $a_0$: recompensa 1 + $V(s_2) = 1$  
>   - $a_1$: recompensa 0 + $V(s_2) = 0$ → mejor $a_0$
>
> - En $s_0$:  
>   - $a_0$: recompensa 0 + $V(s_1) = 0$  
>   - $a_1$: recompensa 0 + $V(s_2) = 0$ → indiferente
>
> Actualizamos la política en $s_1$ para tomar $a_0$.
>
> Ahora, con la nueva política (mejorada), volvemos a evaluar los valores y a iterar el proceso hasta que la política deje de cambiar. En ese punto se alcanza la política óptima.
>
> Estos ejemplos muestran cómo los algoritmos model-based, al disponer de la función de transición y de recompensa, pueden simular el comportamiento del entorno y resolver las ecuaciones de Bellman de forma iterativa. La **iteración de valor** actualiza los valores directamente, mientras que la **iteración de política** alterna entre evaluación y mejora, permitiendo una optimización sistemática de las decisiones del agente.

---

Vamos ahora con otro escenario completamente distinto. Y es que en muchos problemas del mundo real, el agente no tiene acceso explícito al modelo del entorno. Esto significa que no conoce de antemano las **reglas que rigen la dinámica de las transiciones** entre estados ni las recompensas que pueden obtenerse. En otras palabras, las funciones $\mathcal{P}(s' \mid s, a)$ y $\mathcal{R}(s, a, s')$, que en un MDP model-based están disponibles, son aquí **desconocidas o inobservables directamente**. Esta situación es muy común en entornos complejos, dinámicos o parcialmente observables, como la interacción con usuarios humanos, la navegación en el mundo físico o la toma de decisiones en sistemas donde los mecanismos internos no son accesibles.

En estos casos se recurre a algoritmos **model-free**, que prescinden de cualquier conocimiento previo sobre el modelo del entorno. En lugar de razonar a partir de un modelo simbólico, el agente **aprende directamente de su experiencia**, es decir, de las secuencias de interacción que recoge al actuar: $(s_t, a_t, r_{t+1}, s_{t+1})$. A partir de estas transiciones observadas, el agente intenta aproximar funciones de valor y encontrar políticas cada vez mejores sin necesidad de conocer la distribución de probabilidad de las transiciones ni la recompensa esperada exacta.

Este enfoque conlleva una diferencia fundamental respecto al aprendizaje *model-based*: **el agente no puede planificar internamente sus acciones** mediante simulaciones, porque no dispone de un modelo que le permita hacerlo. En su lugar, debe **interactuar con el entorno real** para recopilar información, explorar diferentes posibilidades y corregir sus decisiones basándose en los resultados obtenidos. Este proceso requiere técnicas específicas para equilibrar la exploración con la explotación, gestionar la incertidumbre y adaptar el comportamiento de forma progresiva.

Pese a estas limitaciones, los algoritmos model-free han demostrado ser muy potentes y versátiles. Entre los más representativos se encuentran **Q-learning**, que permite aprender funciones acción-valor sin necesidad de seguir una política fija, y **SARSA**, que actualiza los valores en función de la política seguida por el agente. Además, los métodos **basados en diferencia temporal (TD)** permiten combinar las ventajas del aprendizaje por Monte Carlo (basado en episodios) con la eficiencia de las actualizaciones incrementales, sin requerir el conocimiento del modelo.

Gracias a estas técnicas, es posible abordar una amplia gama de tareas de aprendizaje por refuerzo sin necesidad de conocer ni construir un modelo explícito del entorno, lo que convierte a los métodos model-free en una herramienta esencial para escenarios realistas y complejos.

> **Ejemplo 1: navegación en un entorno desconocido**
>
> Supongamos que un robot móvil debe aprender a desplazarse por un edificio sin un plano previo. Cada vez que intenta avanzar por un pasillo, gira en una dirección o cruza una puerta, recibe una recompensa que depende de si ha avanzado en la dirección correcta o se ha desviado. El robot no conoce las reglas del entorno: no sabe si una acción le llevará a un pasillo bloqueado, a una sala, o a una salida. En este caso, no dispone de una función $\mathcal{P}(s' \mid s, a)$ que le permita simular mentalmente los resultados de sus acciones.
>
> Lo que puede hacer es actuar, observar lo que sucede, y registrar transiciones del tipo $(s_t, a_t, r_{t+1}, s_{t+1})$. A lo largo de muchos intentos, puede estimar qué secuencias conducen a estados más favorables y ajustar su comportamiento en consecuencia. Este aprendizaje se realiza sin tener un modelo del entorno, solo mediante la **acumulación y actualización de experiencia**. Algoritmos como Q-learning permiten estimar directamente los valores de las acciones y mejorar la política gradualmente.
>
> **Ejemplo 2: interacción con usuarios en un sistema de recomendación**
>
> Consideremos ahora un sistema que recomienda contenido (películas, productos, noticias) a usuarios en función de su comportamiento previo. Cada vez que el sistema sugiere un ítem, recibe una recompensa binaria: 1 si el usuario interactúa (por ejemplo, hace clic), 0 si lo ignora. El sistema no conoce de antemano las preferencias exactas del usuario ni cómo estas pueden cambiar con el tiempo. Además, las acciones tomadas hoy afectan a lo que el usuario verá mañana, y por tanto no hay un modelo claro que permita predecir exactamente la transición entre estados.En este escenario, el sistema recopila episodios del tipo $(s_t, a_t, r_{t+1}, s_{t+1})$, donde el estado $s_t$ representa, por ejemplo, el historial de interacciones, la acción $a_t$ es el ítem recomendado, y $r_{t+1}$ indica si la recomendación fue exitosa. Al no disponer de una función de transición explícita, el sistema debe **aprender a optimizar sus recomendaciones exclusivamente a partir de los datos observados**. Algoritmos como SARSA o métodos TD pueden aplicarse para estimar el valor esperado de diferentes políticas de recomendación, ajustando la estrategia a medida que se acumula experiencia.
>
> Estos ejemplos muestran cómo el aprendizaje model-free permite resolver tareas complejas en entornos reales, incluso cuando no es posible disponer de un modelo formal del entorno o de sus reglas de transición. En lugar de planificar con un modelo conocido, el agente **descubre patrones de éxito o fracaso directamente a partir de la experiencia** acumulada.

#### Predicción vs. Control

Otra distinción relevante en el estudio de algoritmos de RL tiene que ver con el objetivo que se persigue en cada caso. En algunas ocasiones, el interés se centra únicamente en evaluar el comportamiento de una política fija, es decir, en estimar cuánto retorno se puede esperar si se siguen siempre las mismas decisiones. Esta tarea recibe el nombre de **predicción**, y da lugar a algoritmos que calculan o aproximan las funciones de valor $v_\pi(s)$ o $q_\pi(s,a)$ para una política dada. La predicción es útil, por ejemplo, para evaluar soluciones predefinidas o como paso intermedio en métodos de mejora.

Sin embargo, el objetivo más habitual en aprendizaje por refuerzo es el de control, es decir, encontrar una política que sea óptima o, al menos, mejor que las anteriores. En estos casos, el agente debe combinar la estimación de valores con la mejora de la política, de forma iterativa. El control requiere, por tanto, métodos que no solo evalúan, sino que también ajustan las decisiones para maximizar el retorno esperado. Este proceso suele implicar mecanismos de exploración, mejora de política y actualización continua.

#### Tipología de los algoritmos en RL

Ambas dimensiones —el uso de modelo y la naturaleza de la tarea— permiten construir una clasificación cruzada de los algoritmos más importantes. Por un lado, existen algoritmos de predicción basados en modelo, que resuelven las ecuaciones de Bellman de forma exacta cuando el modelo es conocido. Por otro lado, los métodos de predicción sin modelo, como Monte Carlo o TD(0), estiman los valores a partir de muestras. En cuanto al control, los algoritmos model-based recurren a planificación combinada con mejora iterativa de política, mientras que los model-free aplican estrategias como Q-learning o SARSA para aprender directamente a partir de la experiencia.

En las secciones siguientes estudiaremos en detalle estas familias de algoritmos, comenzando por aquellos que suponen conocimiento completo del entorno y utilizan planificación explícita. Posteriormente, abordaremos los métodos model-free, que representan el enfoque más general y aplicable cuando el entorno no está completamente especificado. A lo largo del módulo, se utilizarán ejemplos prácticos, visualizaciones y código para facilitar la comprensión de cada técnica y su aplicación.

En la siguiente tabla podemos ver un esquema de clasificación de los distintos algoritmos tal y como se ha comentado más arriba

|                 | **Predicción**                                               | **Control**                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Model-based** | Evaluación de políticas mediante planificación exacta.<br />Ej.: resolución de las ecuaciones de Bellman con modelo conocido. | Planificación iterativa con mejora de política. <br> Ej.: iteración de valor, iteración de política. |
| **Model-free**  | Estimación de valores a partir de experiencia. <br> Ej.: métodos de Monte Carlo y diferencias temporales | Aprendizaje de políticas óptimas a partir de interacción. <br> Ej.: Q-learning, SARSA,… |

### Programación dinámica

#### **Condiciones de aplicabilidad**

La programación dinámica (PD) proporciona un conjunto de algoritmos clásicos que permiten **resolver problemas de decisión secuencial** cuando se dispone de una descripción completa del entorno. Su aplicación en el aprendizaje por refuerzo se basa en la formalización previa del entorno como un **Proceso de Decisión de Markov (MDP)**.

Sin embargo, no en todos los casos es posible aplicar directamente estos algoritmos. Existen **condiciones necesarias** que deben cumplirse para que la programación dinámica sea una opción viable.

Para aplicar programación dinámica, el agente debe tener acceso completo al modelo del entorno, lo que significa disponer explícitamente de:

- La **función de transición** $\mathcal{P}(s' \mid s, a)$, que proporciona la **probabilidad de transición** al estado $s'$ al ejecutar la acción $a$ en el estado $s$.
- La **función de recompensa** $\mathcal{R}(s, a, s')$, que determina el valor esperado de la **recompensa inmediata** tras la transición de $s$ a $s'$ al ejecutar la acción $a$.

Este conocimiento permite al agente **simular internamente** los efectos de sus decisiones, sin necesidad de interactuar físicamente con el entorno. En otras palabras, el agente puede planificar su comportamiento mediante **razonamiento sobre el modelo**, en lugar de aprender a partir de la experiencia directa.

##### Aplicabilidad: entornos donde el modelo es accesible

Este marco de suposiciones —es decir, conocer por adelantado las reglas de transición del entorno y su función de recompensa— restringe el uso de la programación dinámica a una clase de problemas muy específicos. No obstante, existen varios contextos relevantes en los que este conocimiento sí está disponible, lo que permite aplicar estos algoritmos de forma efectiva y precisa.

Uno de los escenarios más comunes es el de las **simulaciones artificiales**, especialmente en entornos de tipo lúdico o computacional. Por ejemplo, juegos como el **ajedrez**, el **Go** o incluso variantes simplificadas como el **tres en raya** presentan una dinámica completamente definida y determinista: el estado actual del tablero, las acciones legales y el resultado de cada movimiento pueden modelarse sin ambigüedad. En estos casos, la programación dinámica permite calcular el valor esperado de cada estado del juego y diseñar políticas óptimas de actuación mediante razonamiento interno, sin necesidad de ejecutar partidas completas.

También se encuentran aplicaciones en **modelos físicos** cuya dinámica está bien caracterizada mediante ecuaciones matemáticas. Por ejemplo, en el caso de un **robot móvil que se desplaza sobre un plano sin obstáculos**, el movimiento puede describirse mediante funciones deterministas basadas en la cinemática del sistema. Si se conocen las restricciones físicas, el espacio de estados es discreto y acotado, y las recompensas están definidas (por ejemplo, alcanzar un objetivo o evitar zonas de penalización), entonces se puede resolver el problema de navegación utilizando programación dinámica.

Otra clase de entornos donde esta aproximación resulta útil es la de los **problemas de planificación determinista o estocástica**, siempre que se disponga de un **modelo explícito** construido a partir de datos históricos o de especificaciones formales. Por ejemplo, en un sistema de **planificación logística en una red de almacenes y rutas**, es posible modelar la probabilidad de éxito de las entregas, los costes asociados y la evolución del inventario. Si este modelo es suficientemente preciso, entonces la programación dinámica puede emplearse para encontrar políticas de reaprovisionamiento o rutas de transporte que minimicen costes o maximicen eficiencia.

En resumen, los algoritmos de programación dinámica son aplicables en contextos donde el entorno es **completamente especificable** y **computacionalmente tratable**, ya sea por construcción manual del modelo o porque puede derivarse de simulaciones controladas. En estos casos, la planificación basada en el modelo permite resolver las ecuaciones de Bellman con precisión y obtener decisiones óptimas sin necesidad de exploración directa.

Sin embargo, en muchos entornos reales —como la interacción con usuarios, la robótica en entornos no estructurados o el control de procesos inciertos— el modelo del entorno **no está disponible** o **no puede construirse de forma fiable**. Esto limita la aplicabilidad directa de los métodos de programación dinámica y motiva la necesidad de enfoques alternativos basados en aprendizaje a partir de la experiencia, como los algoritmos **model-free**.

Además, incluso cuando el modelo es conocido, la programación dinámica puede resultar computacionalmente ineficiente en espacios de estado y acción muy grandes, lo que justifica el uso posterior de aproximaciones funcionales y técnicas de RL profundo.

#### **Evaluación de políticas**

Una vez que el entorno se ha modelado como un proceso de decisión de Markov y se dispone de una política concreta —es decir, de una regla que especifica qué acción tomar en cada estado—, el siguiente paso natural consiste en cuantificar su rendimiento. Esta tarea, conocida como **evaluación de políticas**, permite responder a una pregunta fundamental: _¿cuánto valor puede esperar obtener un agente si actúa conforme a dicha política desde un estado determinado?_

La noción clave aquí es la de **función de valor**, que mide el retorno esperado bajo una política fija. Al evaluar una política no se busca modificarla ni mejorarla, sino **entender su comportamiento medio a largo plazo**. Esta información es valiosa porque proporciona una base cuantitativa sobre la cual construir mejoras posteriores, y también porque permite comparar diferentes políticas entre sí de forma objetiva.

Formalmente, se desea calcular la función $v_\pi(s)$, que representa el valor esperado del retorno cuando el agente comienza en el estado $s$ y sigue la política $\pi$:

$$
v_{\pi}(s) = \mathbb{E} \left[ G_t \mid s_t = s \right] = \mathbb{E} \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t = s \right]
$$

Esta expectativa se toma sobre las posibles trayectorias generadas por la política $\pi$, incluyendo la estocasticidad del entorno y de la propia política si no es determinista. Para calcular esta función de valor, no es necesario simular episodios ni observar interacciones: basta con **resolver el sistema de ecuaciones que se deriva de la ecuación de Bellman para $v_\pi$**, aprovechando que se dispone del modelo completo del entorno.

La **ecuación de Bellman para una política fija** describe el valor de un estado en función de las decisiones dictadas por la política y de las transiciones del entorno. Se expresa como:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma v_\pi(s') \right]
$$

Esta igualdad establece una relación de dependencia entre el valor del estado actual y el de sus sucesores, ponderados por la política y las probabilidades de transición. Cada término del sumatorio representa una posible evolución del sistema al tomar la acción $a$ en el estado $s$, transitar al estado $s'$, recibir una recompensa y continuar desde allí. El valor de un estado, por tanto, se define de manera **recursiva**: depende del valor de otros estados, lo que motiva el uso de métodos iterativos para su resolución.

En la práctica, para calcular esta función de valor se parte de una estimación inicial arbitraria de $v_\pi(s)$ (por ejemplo, asignando cero a todos los estados no terminales) y se aplica la ecuación anterior de forma repetida hasta que los valores convergen. Este procedimiento se conoce como **evaluación iterativa** de la política. En cada iteración se actualiza el valor de cada estado usando los valores actuales de sus sucesores, lo que produce una mejora progresiva de la aproximación.

La convergencia está garantizada bajo ciertas condiciones —por ejemplo, si el número de estados es finito y el factor de descuento $\gamma < 1$—, y el proceso puede detenerse cuando las variaciones entre iteraciones sean inferiores a una tolerancia prefijada. El resultado final es una función $v_\pi(s)$ que representa, con precisión arbitraria, la expectativa de retorno bajo la política considerada.

Un acuestión importante a tener en cuenta tiene que ver con el enfoque usado a la hora de realizar la evaluación de la política. En efecto, cuando aplicamos el método iterativo de evaluación de políticas, el objetivo es aproximar la función de valor $v_\pi(s)$ asociada a una política fija $\pi$, actualizando progresivamente su estimación en todos los estados del entorno.

Ya hemos visto como la **ecuación de Bellman para una política dada** proporciona una definición recursiva del valor esperado y para resolverla, se parte de una estimación inicial (por ejemplo, $v^{(0)}(s) = 0$ en todos los estados), y se aplica un procedimiento iterativo. En este contexto, existen dos variantes importantes del método iterativo. En la denominada **actualización síncrona (por lotes)**, todos los valores $v^{(k+1)}(s)$ en la iteración $k+1$ se calculan **utilizando exclusivamente** los valores de la iteración anterior $v^{(k)}(s')$. Esto asegura una separación clara entre la iteración actual y la anterior. Es el enfoque más común en y garantiza una interpretación directa de la propagación del valor. Sin embargo, también puede usarse la denominada **actualización asíncrona (en línea)**, en la que en cuanto se calcula el nuevo valor $v^{(k+1)}(s)$ para un estado, se actualiza **en el momento**, y ese nuevo valor se puede usar inmediatamente para calcular los siguientes estados. Aunque más eficiente en la práctica, puede resultar confuso si se quieren entender los fundamentos teóricos.

##### Ejemplo: Tablero bidimensional y evaluación de políticas

Recordemos el problema del tablero bidimensional 4x4 que ya tratamos como ejemplo en el módulo anterior. El espacio de estados se representaba del siguiente modo:

![Frozen Lake: Beginners Guide To Reinforcement Learning With OpenAI Gym](./assets/Frozen-Lake.png)

Donde:

- `S = 0` (estado de inicio)
- `G = 15` (estado final)
- Agujeros en 5, 7, 11 y 12 (estados absorbentes)
- Solo el estado 15 da recompensa 1, el resto 0
- Las transiciones son **estocásticas**, de modo que $\mathcal{P}(s' \mid s, a) = 0.33$ para el destino deseado (si es válido) y 0.33 para cada dirección ortogonal

Proponemos una política **determinista** que elige moverse **siempre a la derecha**.

Para evaluar el valor de los estados usaremos la ecuación de Bellman:

$$
v_\pi(s) = \sum_{s'} \mathcal{P}(s' \mid s, \pi(s)) \left[ \mathcal{R}(s, \pi(s), s') + \gamma \cdot v_\pi(s') \right]
$$

Podemos trabajar con $\gamma = 1$ para simplificar algo más. También supondremos que **los valores iniciales de cada estado son cero**.

La ecuación de Bellman así planteada nos permite usar un método iterativo para calcular los valores de los estados. Empezaremos con todos los valores a cero e iremos iterando para todo el tablero hasta los valores converjan en relación a un parámetro $\theta$ previamente definido. En este caso, por ejemplo, vamos a suponer que $\theta=10^{-4}$  

Ahora vamos a calcular el valor de la ecuación para el estado 14. Para ello tendremos en cuenta que desde ese estado puede se puede transicionar al estado 15, al 9 o quedarse en el mismo estado 14. El sumatorio anterior tendrá tres sumandos y quedará como sigue:

$$
v_\pi(14) = \sum_{a' \in \{\text{right, up, down}\}} \mathcal{P}(s' \mid 14, a') \left[ \mathcal{R}(14, a', s') + \gamma \cdot v_\pi(s') \right]
$$

Desglosando cada término:

- $0{,}33 \cdot (1 + v_\pi(15))$  
- $0{,}33 \cdot (0 + v_\pi(10))$  
- $0{,}33 \cdot (0 + v_\pi(14))$

Sustituyendo:

$$
v^{(1)}(14) = 0{,}33 \cdot (1 + v_\pi(15)) + 0{,}33 \cdot v_\pi(10) + 0{,}33 \cdot v^{(0)}_\pi(14)
$$

En la primera configuración de estados todos los valores están a cero, así que:

$$
v^{(0)}(10) = v^{(0)}(14) = v^{(0)}(15) = 0
$$

Por tanto:

$$
v^{(1)}(14) = 0{,}33 + 0 + 0 = 0,33
$$

En esta iteración ($k=1$), podemos intentar el calculo para otros estados, aunque debido a los valores nulos el resultado será cero.

Vamos a hora con la siguiente iteración ($k=2$). Volvemos a hacer el cálculo para $s_{14}$ y $s_{10}$. Si planteamos la ecuación de Bellman para $s_{14}$ y sustituimos valores nos quedaría que:

$$
v^{(2)}(14) = 0{,}33 \cdot (1 + 0) + 0{,}33 \cdot 0 + 0{,}33 \cdot 0,33 = 0,42
$$

Y para $s_{10}$ tendremos que:

$$
v^{(1)}(10) = 0{,}33 \cdot 0 + 0{,}33 \cdot 0 + 0{,}33 \cdot 0,33 = 0,1089
$$

Del mismo modo podríamos seguir calculando valores e iterando.

Una cuestión importante es cuándo parar de iterar. Para ello tendremos que calcular tras cada iteración, para cada estado la cantidad

$$
|v^{(k+1)}(s) - v^{(k)}(s)|
$$

Calcularemos el máximo de esa diferencia en todo el espacio de estados. Si ese valor es menor que la tolerancia elegida al principio del algoritmo ($\theta$) ya habremos llegado al final del algorimo y la política $v_\pi$ estará evaluada.

> [!important]
>
> Este proceso constituye el núcleo de la evaluación de políticas. Sin él, no sería posible determinar si una política es buena o no, ni habría forma de mejorarla de manera informada. De hecho, todos los algoritmos de programación dinámica se apoyan sobre esta fase de evaluación como paso esencial, ya sea ejecutado de forma explícita o implícita.

#### Exploración y explotación: un dilema central en el aprendizaje por refuerzo

Uno de los desafíos más importantes a los que se enfrenta un agente en aprendizaje por refuerzo es decidir en cada momento si debe explotar el conocimiento que ya posee sobre el entorno o si, por el contrario, debe explorar nuevas acciones que podrían conducir a soluciones mejores a largo plazo. Esta tensión se conoce como el dilema de exploración y explotación.

Explotar significa utilizar la información aprendida hasta el momento para seleccionar la acción que se estima más valiosa. Es decir, el agente toma decisiones basadas en sus estimaciones actuales de las funciones de valor, eligiendo sistemáticamente la mejor opción disponible. Esta actitud es eficiente cuando el conocimiento adquirido es suficientemente fiable, ya que permite maximizar el rendimiento inmediato.

Explorar, en cambio, implica tomar acciones que no necesariamente parecen las mejores en el presente, con el objetivo de adquirir información adicional sobre el entorno. Gracias a la exploración, el agente puede descubrir transiciones desconocidas, recompensas inesperadas o caminos más eficientes hacia su objetivo. Aunque explorar puede resultar costoso en el corto plazo, es esencial para mejorar la política y alcanzar un comportamiento verdaderamente óptimo.

El equilibrio entre ambas estrategias es especialmente delicado durante la fase de aprendizaje. Si el agente explota demasiado pronto, puede converger a una política subóptima basada en información incompleta. Si explora en exceso, puede dilatar innecesariamente el proceso de aprendizaje, desperdiciando episodios sin consolidar decisiones útiles. Por esto es por lo que el diseño de mecanismos que regulen este compromiso resulta fundamental en el desarrollo de algoritmos de refuerzo efectivos.

En este contexto pueden diferenciarse distintos tipos de políticas en RL. La primera de ellas, y de las más intuitivas, es la que se denomina una ***política greedy.*** decimos que una política es **greedy** cuando, en cada estado $s$, selecciona la acción $a$ que maximiza el valor estimado:

$$
\pi(s) = \arg\max_a q(s, a)
$$

Es decir, **la política greedy elige siempre la mejor acción conocida**, sin considerar la posibilidad de explorar alternativas. Desde el punto de vista del agente, adoptar una política greedy significa que, en todo momento, **explotará al máximo el conocimiento actual que posee** sobre el entorno para obtener la mayor recompensa posible. Este enfoque resulta adecuado cuando se dispone de una **estimación precisa** de los valores de acción, es decir, cuando $q(s,a)$ refleja con fidelidad la dinámica real del entorno. Esta estrategia puede ser adecuada **al final del entrenamiento**, cuando el agente ha adquirido una representación precisa del entorno. Sin embargo, si se aplica desde el principio o en fases intermedias, limita drásticamente la posibilidad de mejora. No se experimentan nuevas acciones, no se descubren recompensas ocultas, y no se corrigen errores de valoración.

Por este motivo, en la mayoría de los algoritmos prácticos se utilizan políticas que combinan una componente greedy **con un mecanismo de exploración controlada**. La **política $\epsilon$-greedy** es el ejemplo más sencillo: selecciona la acción greedy con alta probabilidad (por ejemplo, $1 - \epsilon$), pero con una pequeña probabilidad $\epsilon$ elige una acción aleatoria. Esta pequeña perturbación permite al agente mantener la posibilidad de explorar, incluso cuando ha comenzado a explotar. A lo largo del tiempo, $\epsilon$ puede reducirse gradualmente, permitiendo una transición progresiva hacia una política completamente determinista.

Este equilibrio dinámico entre exploración y explotación no solo mejora la eficiencia del aprendizaje, sino que también refleja una estrategia adaptativa inteligente. En fases iniciales, donde todo es incierto, es preferible explorar ampliamente. A medida que se acumula experiencia, es razonable explotar cada vez más el conocimiento adquirido. Comprender y gestionar este equilibrio es esencial para diseñar agentes de aprendizaje por refuerzo que sean eficaces y robustos.

Vamos ahora a ver como podemos mejorar una política que se ha evaluado previamente.

#### Algoritmo de mejora de la política

Una vez que el agente ha aprendido a evaluar una política, el siguiente paso natural consiste en utilizar esta información para **mejorar dicha política**. El objetivo es aprovechar las estimaciones de valor obtenidas durante la evaluación para modificar el comportamiento del agente en cada estado y así aumentar su rendimiento esperado.

La idea que subyace a este procedimiento es conceptualmente sencilla: si el agente conoce los valores $q_\pi(s, a)$ que reflejan el retorno esperado de cada acción en un estado dado bajo la política actual $\pi$, entonces puede comparar todas las acciones posibles en cada estado y seleccionar aquella que proporcione el mayor valor. De este modo, se obtiene una **nueva política** que es **greedy respecto a la función de valor actual**.

Este procedimiento se denomina **mejora de la política**, y puede formalizarse de la siguiente forma. Dada una política $\pi$ y su correspondiente función acción-valor $q_\pi(s, a)$, se define una nueva política $\pi'$ como:

$$
\pi'(s) = \arg\max_a q_\pi(s, a)
$$

Esta nueva política selecciona, en cada estado, la acción que maximiza el valor estimado. Si $\pi' = \pi$, es decir, si la política actual ya es greedy con respecto a sus propios valores, entonces $\pi$ es **óptima localmente**. En caso contrario, $\pi'$ representa una política **estrictamente mejor o igual** que la anterior.

Desde el punto de vista algorítmico, la mejora de la política se basa en un paso de actualización que puede expresarse así: en cada estado $s$, se recorren todas las acciones disponibles y se selecciona aquella que maximiza la expresión:

$$
\sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma v_\pi(s') \right]
$$

Esto equivale a evaluar el impacto esperado de ejecutar la acción $a$ desde $s$, teniendo en cuenta tanto la recompensa inmediata como el valor de los estados futuros a los que puede llevar. La política se modifica para elegir, en cada estado, la acción con mayor expectativa de retorno.

En la práctica, este proceso puede aplicarse tras una fase de evaluación de política. Se obtiene $v_\pi$ o $q_\pi$ y se construye una política mejorada $\pi'$. Posteriormente, esta nueva política puede evaluarse de nuevo, y el ciclo puede repetirse. Esta idea es la base de los algoritmos de iteración de la política que se estudiarán más adelante.

La mejora de política puede visualizarse como un mecanismo de ajuste progresivo. A partir de una política inicial arbitraria, se producen pequeñas modificaciones locales que aumentan el retorno esperado en cada estado. Repetido suficientes veces, este procedimiento puede conducir a políticas óptimas, siempre que se mantenga un equilibrio adecuado entre evaluación y mejora.

Este proceso también refleja el principio de explotación mencionado anteriormente. Al mejorar la política, el agente incrementa su preferencia por acciones que han demostrado un valor alto, y reduce la frecuencia de acciones menos prometedoras. Por esto es por lo que la mejora de política se convierte en un mecanismo natural de optimización dentro del ciclo de aprendizaje por refuerzo.

> [!note]
>
> **¿Es necesario conocer la función acción-valor para mejorar la política?**
>
> Una duda habitual al estudiar el algoritmo de mejora de la política es si es imprescindible disponer de la función acción-valor $q_\pi(s, a)$ para llevar a cabo el proceso de mejora. La respuesta es que **no es necesario conocer ni almacenar explícitamente esta función**, siempre que se disponga de dos elementos fundamentales: la función de valor de estados $v_\pi(s)$ y el modelo del entorno, es decir, las funciones de transición y de recompensa.
>
> En efecto, para mejorar una política en un estado dado, lo único que se necesita es poder comparar las distintas acciones disponibles en ese estado. Esta comparación se realiza evaluando el retorno esperado de cada acción, lo cual puede calcularse directamente mediante la siguiente expresión:
>
> $$
> q_\pi(s, a) = \sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma v_\pi(s') \right]
> $$
>
> Esta fórmula no requiere que $q_\pi(s, a)$ se conozca previamente, ya que puede evaluarse **de forma puntual** cada vez que se desea mejorar la política. De este modo, para cada estado $s$, se calcula el valor esperado de todas las acciones posibles $a \in \mathcal{A}(s)$ aplicando esta expresión. A continuación, se selecciona aquella acción que maximiza el resultado obtenido, generando así una nueva política mejorada:
>
> $$
> \pi'(s) = \arg\max_a \sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma v_\pi(s') \right]
> $$
>
> Por esto es por lo que se afirma que **la mejora de política puede realizarse directamente a partir de la función de valor** sin necesidad de disponer de la función acción-valor como objeto intermedio. El modelo del entorno proporciona toda la información necesaria para estimar el valor de las acciones en términos de los estados sucesores y sus respectivos valores. Esta capacidad de mejora basada únicamente en $v_\pi$ es una de las razones por las que la programación dinámica es computacionalmente eficiente cuando se dispone del modelo completo del problema.

##### Ejemplo de mejora de la política en un tablero unidimensional

Consideramos un entorno muy simple con tres estados dispuestos en línea: $s_0$, $s_1$ y $s_2$.

- $s_0$ representa un hueco. 
- $s_1$ es el estado inicial. 
- $s_2$ es la meta. 

Las acciones disponibles en todos los estados son:

- $a_0$: moverse a la izquierda. 
- $a_1$: moverse a la derecha.

La dinámica del entorno es determinista. Las transiciones posibles son:

- Desde $s_1$, la acción $a_0$ lleva a $s_0$ (sin recompensa). 
- Desde $s_1$, la acción $a_1$ lleva a $s_2$ (con recompensa $1$). 
- Desde $s_0$ o $s_2$, cualquier acción deja al agente en el mismo estado, sin recompensa. 
- Consideramos $\gamma = 1$.

Supongamos que el agente parte de una **política inicial** $\pi$ que no alcanza la meta:

- $\pi(s_0) = a_0$  
- $\pi(s_1) = a_0$  
- $\pi(s_2) = a_1$  

Tras aplicar el algoritmo de evaluación de política, obtenemos:

- $v_\pi(s_0) = 0$  
- $v_\pi(s_1) = 0$  
- $v_\pi(s_2) = 0$

A continuación, aplicamos un **paso de mejora de la política**, usando el modelo del entorno y los valores actuales.

En el estado $s_1$:

- Si el agente elige $a_0$, llega a $s_0$: 
- 
$$
q(s_1, a_0) = \mathcal{R}(s_1, a_0, s_0) + \gamma \cdot v_\pi(s_0) = 0 + 1 \cdot 0 = 0
$$

- Si elige $a_1$, llega a $s_2$ y recibe una recompensa:

$$
q(s_1, a_1) = \mathcal{R}(s_1, a_1, s_2) + \gamma \cdot v_\pi(s_2) = 1 + 1 \cdot 0 = 1
$$

Por tanto, la acción $a_1$ es mejor, y se mejora la política en $s_1$:

- $\pi'(s_1) = a_1$

En los estados $s_0$ y $s_2$ no hay recompensas ni cambios de estado, por lo que las acciones tienen valor cero:

- $q(s_0, a_0) = q(s_0, a_1) = 0$ 
- $q(s_2, a_0) = q(s_2, a_1) = 0$ 

Por tanto, no hay necesidad de cambiar la política en esos estados.

La **política mejorada** resultante es:

- $\pi'(s_0) = a_0$  
- $\pi'(s_1) = a_1$  
- $\pi'(s_2) = a_1$

Esta política permite alcanzar el estado objetivo $s_2$ desde $s_1$ con una recompensa de 1, lo cual mejorará el valor del estado $s_1$ en futuras evaluaciones. Esta sencilla mejora muestra cómo, utilizando la función de valor $v_\pi$ y el modelo del entorno, se pueden identificar acciones más prometedoras en cada estado y construir políticas cada vez más eficaces.

#### Iteración de políticas

Hemos visto cómo es posible evaluar una política fija, obteniendo su función de valor, y cómo puede mejorarse dicha política utilizando los valores estimados para elegir acciones más prometedoras. Estos dos procesos —evaluación y mejora— pueden combinarse en un esquema iterativo que constituye uno de los algoritmos fundamentales en programación dinámica: la **iteración de políticas**.

Este método parte de una política arbitraria y la va perfeccionando en sucesivas fases, alternando entre la evaluación de los valores esperados bajo dicha política y la mejora de las acciones elegidas en cada estado. El objetivo de este procedimiento es alcanzar una política estable que no pueda mejorarse más, lo que implica que se ha llegado a una política óptima.

La lógica del algoritmo es sencilla pero muy poderosa. Se comienza con una política inicial y se evalúan sus valores mediante la ecuación de Bellman. Con esos valores, se realiza una mejora de política seleccionando, en cada estado, la acción que maximiza la recompensa inmediata esperada más el valor estimado del estado siguiente. Si esta mejora da lugar a una política distinta, se repite el proceso. Pero si al aplicar el paso de mejora la política resultante es idéntica a la anterior, se ha alcanzado una política que ya es óptima, y el algoritmo se detiene.

Este punto es clave: **el criterio de parada no es que los valores de los estados hayan convergido a un conjunto concreto**, ni que coincidan con los valores óptimos. La condición que detiene el algoritmo es que **la política no cambie** tras el paso de mejora. Es decir, la política es **greedy respecto a su propia función de valor**. Cuando esto sucede, se garantiza que ya no existe ninguna acción alternativa que mejore el comportamiento del agente, y por tanto la política es óptima.

Esta propiedad hace que la iteración de políticas sea particularmente eficiente y segura. Siempre que los espacios de estados y acciones sean finitos, el algoritmo está garantizado a converger en un número finito de iteraciones. Además, en cada paso se obtiene una política no peor que la anterior, por lo que el proceso nunca retrocede. Esto permite interpretar el algoritmo como una sucesión de aproximaciones progresivas hacia una política óptima, en la que cada iteración refina las decisiones del agente a partir del conocimiento adquirido previamente.

Desde un punto de vista operativo, el algoritmo sigue una guionización sencilla: se inicializa una política arbitraria, se evalúan sus valores, se mejora la política, y se comprueba si ha cambiado. Si hay cambios, se repite el ciclo; si no los hay, se ha llegado al final. Esta dinámica alterna puede entenderse como un proceso de aprendizaje iterativo, en el que cada ciclo proporciona al agente una comprensión más ajustada de las consecuencias de sus decisiones.

Por tanto, la iteración de políticas ofrece una solución completa y sistemática al problema de encontrar una política óptima en entornos donde el modelo del entorno es conocido. Su carácter estructurado y su convergencia garantizada la convierten en una herramienta clave en la teoría del aprendizaje por refuerzo clásico.

Perfecto. A continuación tienes el desarrollo completo en **formato apuntes** para introducir la **iteración de valores** con una explicación clara, motivadora y técnicamente rigurosa. Esta sección pone el énfasis en **por qué** este algoritmo es necesario, **cómo** se relaciona con la programación dinámica y **qué ventajas tiene** respecto a la iteración de políticas, todo ello cuidando especialmente la comprensión por parte del estudiante.

#### Iteración de valores

Hasta ahora hemos trabajado con la **iteración de políticas**, un procedimiento que alterna dos fases bien diferenciadas: por un lado, la evaluación de una política fija, y por otro, su mejora sistemática a partir de los valores aprendidos. Esta estrategia es sólida y garantiza la convergencia hacia una política óptima, pero presenta una dificultad práctica: la **evaluación completa** de la política en cada ciclo puede ser costosa, sobre todo en espacios grandes de estados o cuando el número de iteraciones necesarias es elevado.

La **iteración de valores** surge como una alternativa más **eficiente** y directa. En lugar de separar evaluación y mejora, **fusiona ambos pasos en una sola operación**: en cada iteración se actualiza el valor de cada estado utilizando la mejor acción disponible en ese momento. De este modo, el proceso se basa directamente en la **ecuación de Bellman óptima**, sin necesidad de mantener una política explícita.

Este es el punto clave: mientras que la iteración de políticas sigue una lógica de "primero evalúo cómo de buena es mi política actual, luego decido cómo mejorarla", la iteración de valores adopta una estrategia más agresiva: "voy a suponer en todo momento que tomo la mejor acción posible y actualizo los valores directamente en base a ello".

El proceso se desarrolla de la siguiente manera:

Se parte de una estimación inicial de los valores de todos los estados (por ejemplo, todos ceros). En cada paso, para cada estado $s$, se calcula el nuevo valor como el máximo valor esperado que se puede obtener considerando todas las acciones posibles desde ese estado, aplicando la fórmula:

$$
v_{k+1}(s) = \max_a \sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \cdot v_k(s') \right]
$$

Esto significa que, en cada iteración, el agente **se pregunta qué acción es la mejor en este estado, según los valores actuales**, y **actualiza el valor del estado suponiendo que tomará esa acción**. No se evalúa una política concreta, sino que se actualizan directamente los valores suponiendo que se actúa de forma óptima.

En términos prácticos, se repite este proceso hasta que los valores dejan de cambiar significativamente (es decir, cuando el cambio entre una iteración y la siguiente es menor que un umbral prefijado). Este criterio de parada está basado en los **valores**, no en una política explícita. Una vez que los valores han convergido, se puede recuperar la política óptima seleccionando, para cada estado, la acción que maximizó la expresión anterior.

Esta diferencia con la iteración de políticas es fundamental. En iteración de políticas:

- Se mantiene una política explícita en cada paso.
- Se evalúa exactamente el valor de dicha política.
- Luego se mejora la política de forma greedily.

En iteración de valores:

- No se mantiene una política explícita durante el proceso.
- Se actualizan los valores directamente usando la mejor acción disponible en cada paso.
- Solo al final (si se desea) se extrae una política greedily a partir de los valores convergidos.

Este enfoque resulta ser más **computacionalmente compacto** y a menudo más **rápido**, ya que evita la evaluación completa de políticas intermedias. Además, permite **propagar información útil de estados con recompensa hacia atrás** en el espacio de estados desde las primeras iteraciones, haciendo que incluso sin convergencia total, los valores sean ya informativos.

Desde una perspectiva didáctica, la iteración de valores representa una **forma natural de propagación del conocimiento**: cada vez que un estado tiene un sucesor con alto valor, este "empuja" su utilidad hacia atrás, afectando a los valores de estados que llevan hasta él. Esta propagación es lo que finalmente construye un mapa de valores coherente con las mejores decisiones que se pueden tomar desde cada punto.

Por todo ello, la iteración de valores se convierte en uno de los algoritmos básicos de programación dinámica y constituye el núcleo conceptual de muchas técnicas modernas de aprendizaje por refuerzo.

#### Programación dinámica: Discusión final y comparativa

La programación dinámica ofrece una solución teóricamente sólida al problema de decisión secuencial en entornos donde el modelo del entorno es completamente conocido. Tanto la **iteración de políticas** como la **iteración de valores** permiten hallar una política óptima, pero lo hacen mediante estrategias distintas, lo que da lugar a diferencias relevantes desde el punto de vista computacional y conceptual.

Ambos algoritmos están respaldados por demostraciones matemáticas que garantizan su **convergencia** a la política óptima y a la función de valor correspondiente. Sin embargo, sus trayectorias hacia esa convergencia no son equivalentes. La iteración de políticas sigue un proceso estructurado de evaluación exacta y mejora sistemática, que puede implicar costes altos si se requiere precisión en la evaluación. En cambio, la iteración de valores realiza actualizaciones aproximadas pero inmediatas, propagando la información de forma más flexible y adaptativa desde los estados terminales hacia el resto del espacio.

Desde el punto de vista **computacional**, esta diferencia es significativa. La iteración de políticas requiere resolver un sistema de ecuaciones en cada paso de evaluación, lo que puede ser costoso si el número de estados es grande. La iteración de valores, en cambio, evita este coste mediante una actualización local basada en el máximo valor esperado entre las acciones posibles, lo que permite avanzar con mayor agilidad, aunque sacrificando exactitud intermedia. Este enfoque suele ser preferible cuando se requiere una implementación simple o cuando interesa disponer de políticas útiles aún antes de la convergencia completa.

Ambos métodos, no obstante, comparten una **limitación esencial**: exigen conocer por adelantado la función de transición $\mathcal{P}(s' \mid s, a)$ y la función de recompensa $\mathcal{R}(s, a, s')$. Esta dependencia del modelo restringe su aplicabilidad en la práctica, donde muchas veces el entorno es desconocido o solo parcialmente observable. Por eso, aunque estos algoritmos no suelen aplicarse directamente en entornos reales, su importancia es fundamental en la teoría del aprendizaje por refuerzo.

La programación dinámica proporciona la **base conceptual sobre la que se construyen los métodos de aprendizaje model-free**. Algoritmos como SARSA, Q-learning o los métodos basados en diferencia temporal pueden entenderse como intentos de aproximar las mismas ecuaciones de Bellman que aquí se han resuelto con modelo explícito, pero utilizando únicamente experiencia observada. Esta continuidad teórica entre programación dinámica y aprendizaje sin modelo es uno de los pilares del campo y ayuda a comprender por qué muchas ideas y expresiones se mantienen constantes a lo largo del temario.

Por tanto, más allá de su aplicabilidad directa, los métodos de programación dinámica ofrecen al estudiante una **visión estructurada y exacta** de los fundamentos del RL. Permiten entender cómo se define y evalúa una política, qué significa actuar de forma óptima y cómo se propaga el valor a través del espacio de estados. Estos conceptos serán cruciales en los bloques siguientes, donde el agente deberá aprender todo esto **sin conocer el entorno**, solo a partir de la experiencia de interacción.

### Métodos de Monte Carlo

#### Motivación: aprendizaje a partir de la experiencia

En los apartados anteriores hemos estudiado cómo resolver un problema de decisión secuencial mediante técnicas de programación dinámica. Estos métodos, como la iteración de políticas o la iteración de valores, parten de una hipótesis muy fuerte: **el conocimiento completo del entorno**, es decir, el acceso explícito a la función de transición $\mathcal{P}(s' \mid s, a)$ y a la función de recompensa $\mathcal{R}(s, a, s')$.

En muchos problemas del mundo real, esta suposición es irrealizable. El agente no conoce la dinámica del entorno ni cómo se generan las recompensas. Lo único que puede hacer es **interactuar con el entorno, registrar lo que ocurre, y aprender a partir de esa experiencia**. Es decir, el aprendizaje ya no se basa en planificación sobre un modelo, sino en **observación directa de episodios** completos: secuencias del tipo 

$$
(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

donde el agente actúa, observa las recompensas obtenidas y las consecuencias de sus acciones.

Este cambio de enfoque marca la entrada en una nueva clase de métodos de aprendizaje por refuerzo: los **métodos Monte Carlo**. Su característica fundamental es que aprenden a partir de **episodios simulados o reales**, sin necesidad de conocer ni estimar la estructura interna del entorno. En lugar de resolver las ecuaciones de Bellman directamente, lo que hacen es **aproximar el valor de un estado o una acción a partir del promedio de los retornos observados** tras múltiples ejecuciones.

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

#### Predicción con métodos Monte Carlo

La predicción es una de las tareas fundamentales en el aprendizaje por refuerzo: consiste en estimar **cuánto retorno cabe esperar** si el agente comienza en un determinado estado y sigue una política dada. Formalmente, el objetivo es aproximar la función de valor $V^\pi(s)$, definida como el valor esperado del retorno total acumulado al iniciar en el estado $s$ y actuar conforme a una política $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]
$$

En los métodos Monte Carlo esta predicción se realiza **únicamente a partir de la experiencia**, sin necesidad de conocer el modelo del entorno ni resolver sistemas de ecuaciones. La idea clave consiste en ejecutar múltiples **episodios completos**, registrar las recompensas obtenidas a lo largo de cada trayectoria, y calcular el retorno total desde cada estado visitado. A partir de esta información, puede estimarse el valor de cada estado como el **promedio de los retornos observados**.

Esto convierte a Monte Carlo en una técnica natural para entornos **episódicos**, donde se puede identificar un comienzo y un final claro. En cada episodio, el agente sigue la política $\pi$, y se recopilan las secuencias del tipo:

$$
(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

Una vez finalizado el episodio, se puede calcular el retorno total a partir de cualquier punto de la secuencia. Si este procedimiento se repite muchas veces y se agrupan los retornos obtenidos desde un mismo estado, la media de dichos retornos se convierte en una estimación del valor de ese estado bajo la política observada.

Este proceso es sencillo de implementar y permite una estimación directa de $V^\pi(s)$, sin requerir la función de transición $\mathcal{P}$ ni la función de recompensa $\mathcal{R}$. La única condición es que la política debe visitar todos los estados relevantes con suficiente frecuencia para garantizar una buena cobertura.

---

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
> V^\pi(s_0) \approx \frac{1 + 1 + 1}{3} = 1
> $$
>
> Este valor se ha estimado sin conocer las transiciones ni las recompensas a priori, simplemente observando los resultados de ejecutar la política.
>

---

##### First-visit y every-visit Monte Carlo

En los métodos Monte Carlo para la predicción del valor de una política, existen dos variantes según cómo se utilicen las visitas a los estados en un episodio: ***first-visit*** y ***every-visit***. Ambas buscan estimar la función de valor $V^\pi(s)$ como promedio de retornos observados tras visitar el estado $s$ siguiendo la política $\pi$, pero difieren en la cantidad de muestras que extraen de cada episodio.

First-visit utiliza exclusivamente **la primera aparición de cada estado** dentro de un episodio. Una vez detectada esa primera visita, se calcula el retorno total $G_t$ desde ese momento y se emplea como muestra única para ese episodio y ese estado. En cambio, Every-visit aprovecha **todas las apariciones del estado** en un mismo episodio. Cada visita genera su propio valor de retorno y contribuye de manera independiente a la estimación de $V^\pi(s)$. Aunque ambas variantes convergen teóricamente al mismo valor esperado, en la práctica pueden comportarse de manera muy distinta en cuanto a eficiencia estadística y velocidad de convergencia.

First-visit resulta especialmente útil cuando los episodios son largos o los estados aparecen muchas veces de forma repetida, ya que evita correlaciones estadísticas dentro del mismo episodio. Esto mejora la estabilidad de la estimación, aunque puede requerir una mayor cantidad de episodios, dado que se extrae menos información de cada uno. En este sentido, es más robusto estadísticamente, pero menos eficiente desde el punto de vista muestral.

Every-visit, por su parte, aprovecha todas las visitas posibles a un estado, lo que incrementa la velocidad de aprendizaje en términos de volumen de datos. Esto es ventajoso en episodios cortos o cuando se dispone de pocos datos, pero puede introducir varianza adicional si las visitas múltiples no aportan información verdaderamente independiente. En entornos donde los estados tienden a repetirse en bucles, las muestras pueden volverse redundantes o correlacionadas, lo que afecta a la calidad de la estimación.

Una analogía puede ayudar a comprender mejor esta diferencia. Supongamos que se desea estimar la temperatura media de una habitación. Aplicando el enfoque first-visit, se toma una única medida por día, quizás al entrar por primera vez. Esto produce un conjunto pequeño de datos, pero cada medida representa un contexto distinto. En cambio, siguiendo el enfoque every-visit, se toman múltiples medidas durante todo el día, incluso cada pocos minutos. El volumen de datos aumenta considerablemente, pero muchas observaciones pueden reflejar situaciones muy similares, añadiendo poco valor informativo. En este caso, aunque se acumulan más datos, la calidad marginal de cada muestra puede ser menor.

Ambos métodos son consistentes estadísticamente y conducen a estimaciones correctas si se acumulan suficientes muestras. No obstante, la decisión de utilizar uno u otro debe estar guiada por el tipo de entorno, la duración típica de los episodios y la frecuencia de visitas a los estados. First-visit tiende a ser más conservador y robusto, mientras que every-visit es más agresivo en términos de aprovechamiento de datos, a costa de asumir posibles correlaciones internas.

Este análisis justifica que ambas estrategias formen parte del conjunto de herramientas estándar para estimar funciones de valor cuando se dispone de muestras completas de episodios. La elección entre ellas representa un compromiso entre la calidad estadística de las muestras y la eficiencia en el uso de los datos disponibles.

---

> ##### Ejemplo: comparación entre first-visit y every-visit Monte Carlo
>
> Consideremos un episodio generado por una política $\pi$ en un entorno simple. El episodio consiste en la siguiente secuencia de estados y recompensas:
>
> $$
> (s_0) \to (s_1, r_1 = 0) \to (s_1, r_2 = 0) \to (s_1, r_3 = 1)
> $$
>
> Es decir, el agente comienza en $s_0$, luego visita tres veces el estado $s_1$ consecutivamente, y finalmente alcanza un estado terminal desde el que recibe una recompensa de 1. Asumimos que $\gamma = 1$, por lo que el retorno $G_t$ desde cada paso es simplemente la suma de las recompensas futuras.
>
> Vamos a calcular la estimación de $V^\pi(s_1)$ con ambos métodos:
>
> **First-visit Monte Carlo**
>
> En este enfoque, solo se tiene en cuenta la **primera visita** al estado $s_1$, que ocurre en el segundo paso del episodio (índice $t = 1$). Desde ese punto, las recompensas futuras son:
>
> $$
> G_1 = r_1 + r_2 + r_3 = 0 + 0 + 1 = 1
> $$
>
> El retorno desde la primera visita es 1. Por tanto, este episodio aporta un valor de 1 a la estimación de $V^\pi(s_1)$.
>
> **Every-visit Monte Carlo**
>
> En este caso, se tienen en cuenta **todas las visitas** al estado $s_1$, es decir, en los pasos $t = 1$, $t = 2$ y $t = 3$. El retorno desde cada uno de esos pasos es:
>
> - Desde $t = 1$: $G_1 = 0 + 0 + 1 = 1$
> - Desde $t = 2$: $G_2 = 0 + 1 = 1$
> - Desde $t = 3$: $G_3 = 1$
>
> La media de estos tres retornos es:
>
> $$
> V^\pi(s_1) \approx \frac{1 + 1 + 1}{3} = 1
> $$
>
> En este caso, **ambos métodos producen el mismo valor**, pero esto no siempre ocurre. La diferencia principal es que:
>
> - **First-visit** reduce la varianza al evitar múltiples actualizaciones por episodio, pero puede necesitar más episodios para converger.
> - **Every-visit** utiliza más datos por episodio, pero puede introducir correlación entre muestras.
>
> Este ejemplo permite visualizar que **ambos enfoques promedian retornos**, pero difieren en **cuántos datos** utilizan por episodio para cada estado. En situaciones reales, la elección entre uno u otro depende del balance deseado entre **eficiencia de muestra** y **robustez estadística**.
>

#### Control con métodos de Monte Carlo

Hasta ahora hemos visto cómo utilizar métodos Monte Carlo para estimar el valor de los estados bajo una política dada. Sin embargo, en el aprendizaje por refuerzo el objetivo habitual no es evaluar una política fija, sino **aprender una política que maximice el retorno esperado**. Este problema, conocido como *control*, requiere que el agente no solo cuantifique el valor de los estados o acciones, sino que también mejore progresivamente su comportamiento a partir de la experiencia.

Para abordar esta tarea con Monte Carlo, el primer paso es estimar la función **acción-valor** $Q^\pi(s, a)$, que representa el retorno esperado al ejecutar la acción $a$ en el estado $s$ y continuar siguiendo la política $\pi$ a partir de ahí. A diferencia de la función $V^\pi(s)$, que mide el valor del estado bajo una política, $Q^\pi(s, a)$ proporciona información más detallada sobre qué acción tomar en cada situación, y por tanto permite **comparar opciones**.

Una vez estimado $Q^\pi(s, a)$, el agente puede utilizar esa información para **mejorar su política**. La idea consiste en adoptar una política que, en cada estado, elija la acción con mayor valor esperado según las estimaciones actuales. A este proceso se le conoce como política **greedy** con respecto a $Q$.

No obstante, si el agente actúa siempre de forma greedy con respecto a sus valores actuales, puede dejar de explorar acciones menos conocidas que podrían ser mejores. Por esta razón, se utiliza una estrategia llamada **$\epsilon$-greedy**, que con probabilidad $1 - \epsilon$ selecciona la mejor acción conocida (explotación) y con probabilidad $\epsilon$ elige una acción al azar (exploración). Este enfoque permite **equilibrar exploración y explotación** mientras se sigue mejorando la política.

El ciclo completo del algoritmo de *control Monte Carlo* consiste, por tanto, en observar episodios generados por una política $\pi$, estimar los valores $Q^\pi(s, a)$ mediante promedios de retornos observados, y usar esa información para construir una nueva política mejorada $\pi'$. A lo largo de muchos episodios, este proceso permite **converger a una política óptima** bajo condiciones adecuadas.

Este tipo de aprendizaje se puede llevar a cabo de dos formas principales:

- En el enfoque **on-policy**, el agente estima el valor de la política que realmente ejecuta y mejora progresivamente esa misma política. Es el caso del algoritmo *Monte Carlo control on-policy* con $\epsilon$-greedy, donde se evalúa y mejora una política estocástica a medida que se interactúa con el entorno.
- En el enfoque **off-policy**, el agente estima el valor de una política objetivo $\pi$ mientras sigue una política de comportamiento diferente $\mu$, que le permite explorar más ampliamente el entorno. Este enfoque requiere técnicas adicionales como el *importance sampling*, que ajustan las estimaciones para corregir la diferencia entre ambas políticas.

A lo largo de los siguientes apartados se mostrará cómo implementar el algoritmo de control Monte Carlo paso a paso, aplicando estimaciones de $Q^\pi(s, a)$, estrategias $\epsilon$-greedy y actualizaciones iterativas de la política. Se analizarán también sus propiedades, limitaciones y se propondrá una práctica completa para su puesta en marcha.


##### Algoritmo de control Monte Carlo (on-policy)

Cuando se utiliza Monte Carlo para resolver un problema de control, el objetivo ya no es simplemente estimar el valor de los estados bajo una política fija, sino aprender directamente una **política óptima** que maximice el retorno esperado. Para lograrlo, el agente debe combinar la estimación de la función $Q^\pi(s,a)$ con un procedimiento de mejora de la política basado en dichas estimaciones.

El algoritmo se apoya en dos pilares fundamentales. Por un lado, se mantiene una política **estocástica $\epsilon$-greedy** que equilibra exploración y explotación. Por otro lado, se lleva a cabo una **estimación por promedio de retornos** de la función acción-valor $Q(s,a)$ a partir de múltiples episodios generados siguiendo esa política. A medida que se recogen más datos, la política se mejora utilizando los valores actuales de $Q$.

El ciclo de aprendizaje se articula en torno a la siguiente secuencia de pasos:

Al comenzar, se inicializan los valores $Q(s,a)$ para todos los pares estado-acción, normalmente a cero o con un pequeño valor aleatorio. Se define también una política inicial $\pi$ basada en esos valores, que suele ser $\epsilon$-greedy respecto a $Q$.

A continuación, se ejecuta un número determinado de episodios de interacción con el entorno. En cada episodio, el agente sigue su política actual, recogiendo una secuencia completa de transiciones de la forma $(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$. Una vez finalizado el episodio, se recorren las transiciones hacia atrás para calcular el retorno acumulado $G_t$ de cada par $(s_t, a_t)$.

Para cada par $(s,a)$ encontrado en el episodio, se acumulan los retornos observados y se actualiza la estimación de $Q(s,a)$ como el promedio de todos los retornos recogidos hasta ese momento. Esta estimación puede basarse en el método *first-visit* o *every-visit*, según se considere solo la primera aparición o todas las apariciones del par dentro del episodio.

Después de cada episodio, se reconstruye la política actual haciendo que, en cada estado $s$, se elija con mayor probabilidad la acción $a$ que maximiza el valor estimado $Q(s,a)$, pero permitiendo también con probabilidad $\epsilon$ elegir otras acciones. Este mecanismo garantiza que la política siga explorando incluso cuando tiene ya valores altos de retorno estimado.

Este proceso se repite durante muchos episodios. Si el entorno es finito, todos los estados y acciones son visitados con suficiente frecuencia, y el valor de $\epsilon$ es adecuado, entonces se garantiza la **convergencia a una política óptima** con probabilidad 1. En la práctica, el número de episodios necesarios depende de la variabilidad del entorno, de la calidad de la exploración y de la precisión deseada.

La ventaja principal de este enfoque es su **simplitud conceptual**: no requiere conocer el modelo del entorno (ni la función de transición ni la recompensa esperada), y se apoya únicamente en la observación de secuencias reales. Su desventaja, en cambio, es que requiere completar episodios completos para poder realizar actualizaciones, lo que puede ser costoso en dominios con episodios largos o poco frecuentes.

En resumen, el control on-policy mediante Monte Carlo permite aprender políticas óptimas a partir de la experiencia, manteniendo una política estocástica que se ajusta iterativamente según los valores observados de retorno. Este enfoque constituye una base sólida sobre la que se construyen métodos más avanzados, como aquellos que combinan Monte Carlo con aprendizaje por diferencia temporal.

###### Discusión y consideraciones prácticas

El enfoque de control on-policy mediante métodos Monte Carlo constituye una de las estrategias más accesibles y conceptualmente claras para aprender políticas óptimas en entornos desconocidos. Al no requerir un modelo explícito del entorno, resulta especialmente útil en escenarios donde la dinámica del sistema es compleja, no se dispone de ecuaciones formales de transición, o simplemente no se tiene acceso a información completa sobre los efectos de las acciones.

Una de sus principales virtudes es que se basa únicamente en la experiencia real generada al interactuar con el entorno. Esto lo convierte en un método completamente *model-free*, que no necesita estimar ni utilizar funciones de transición $\mathcal{P}$ ni funciones de recompensa $\mathcal{R}$. Toda la información relevante se obtiene directamente a partir de los episodios observados, lo cual simplifica considerablemente la implementación inicial y permite aplicar el método en entornos reales sin suposiciones fuertes.

Sin embargo, esta dependencia de episodios completos introduce una limitación estructural importante: para poder actualizar los valores de $Q(s,a)$, es necesario esperar a que el episodio finalice. Esto hace que el enfoque no sea fácilmente aplicable en tareas continuas o en dominios donde los episodios son muy largos o difíciles de completar. Además, el tiempo entre la acción y la retroalimentación puede introducir una alta varianza en las estimaciones, especialmente en dominios con resultados muy variables o recompensas retardadas.

El uso de políticas $\epsilon$-greedy garantiza la exploración suficiente del espacio de decisiones, pero también ralentiza la convergencia a la política óptima, ya que introduce aleatoriedad incluso cuando se ha identificado ya una buena acción. Reducir gradualmente el valor de $\epsilon$ a medida que mejora la política puede ser una solución eficaz para acelerar la convergencia sin perder capacidad exploratoria, aunque ello conlleva un ajuste fino de hiperparámetros y una cierta planificación del proceso de aprendizaje.

El proceso de actualización mediante promedios acumulados tiene la ventaja de ser estadísticamente consistente, pero puede resultar ineficiente en términos de velocidad de aprendizaje. En comparación con métodos basados en diferencias temporales, las actualizaciones de Monte Carlo no aprovechan el hecho de que muchas veces se puede estimar parcialmente el valor de un estado sin esperar al final del episodio. Este aspecto será clave cuando abordemos en el siguiente bloque los algoritmos TD, que combinan ventajas de Monte Carlo con aprendizaje incremental.

En resumen, el control on-policy con Monte Carlo ofrece un marco robusto para el aprendizaje de políticas óptimas, con la condición de que los episodios puedan observarse por completo y la exploración se mantenga activa. Su transparencia lo convierte en un excelente punto de partida para el estudio de algoritmos de refuerzo, pero también revela la necesidad de enfoques más eficientes cuando el entorno es complejo o el feedback es escaso. En estos casos, la extensión hacia métodos off-policy o algoritmos híbridos puede aportar mejoras sustanciales.

###### Ejemplo numérico: control Monte Carlo en un entorno simple

Consideramos un entorno muy reducido, representado como un tablero unidimensional con tres estados: $s_0$, $s_1$ y $s_2$. El estado $s_2$ es absorbente y representa una meta, mientras que $s_0$ es una casilla vacía y $s_1$ es la posición inicial. El agente puede realizar dos acciones: $a_0$ (izquierda) y $a_1$ (derecha). Las transiciones son deterministas.

Las reglas de transición son las siguientes:

- Desde $s_1$, si se toma $a_0$, se transita a $s_0$ con recompensa 0.
- Desde $s_1$, si se toma $a_1$, se transita a $s_2$ con recompensa 1.
- Desde $s_0$ y $s_2$, cualquier acción mantiene al agente en el mismo estado con recompensa 0.

Inicializamos $Q(s,a)$ con valores cero para todas las combinaciones de estado y acción. Usamos una política $\epsilon$-greedy con $\epsilon = 0.5$ (para facilitar la exploración en este entorno tan pequeño). 

Supongamos que el agente ejecuta un primer episodio desde $s_1$ siguiendo la política $\epsilon$-greedy. Elige la acción $a_1$ (ir a la derecha), transita a $s_2$ y termina el episodio con una recompensa de 1. Esta trayectoria puede representarse como:

$$(s_1, a_1, r = 1, s_2)$$

Calculamos el retorno total desde la única transición observada:

$$
G_0 = 1
$$

Dado que estamos utilizando el método **first-visit**, solo se actualiza $Q(s_1, a_1)$ con esta muestra. Como es la primera vez que se observa ese par $(s,a)$, el valor estimado se actualiza como:

$$
Q(s_1, a_1) \leftarrow \frac{1}{1} \cdot G_0 = 1
$$

Tras este primer episodio, la política $\epsilon$-greedy se reconstruye en base a los nuevos valores $Q$. En $s_1$, ahora la acción $a_1$ tiene valor 1, mientras que $a_0$ sigue valiendo 0. La política en $s_1$ se actualizaría de modo que $a_1$ se elija con probabilidad $1 - \epsilon + \epsilon/2 = 0.75$, y $a_0$ con probabilidad $0.25$.

En un segundo episodio, supongamos que el agente elige $a_0$ (izquierda) desde $s_1$, llega a $s_0$ y se queda ahí, terminando con una recompensa total de 0:

$$(s_1, a_0, r = 0, s_0)$$

Este episodio proporciona una muestra para actualizar $Q(s_1, a_0)$:

$$
G_0 = 0 \quad \Rightarrow \quad Q(s_1, a_0) \leftarrow \frac{1}{1} \cdot 0 = 0
$$

La política $\epsilon$-greedy sigue favoreciendo $a_1$ en $s_1$ con probabilidad 0.75, pero continúa explorando también $a_0$. A medida que se acumulan más episodios, se acumulan nuevas muestras para cada par $(s,a)$ y las estimaciones de $Q$ se refinan. Por ejemplo, si se observa una segunda vez la transición $(s_1, a_1) \to s_2$ con recompensa 1, la actualización se haría promediando los dos valores observados:

$$
Q(s_1, a_1) \leftarrow \frac{1 + 1}{2} = 1
$$

Los valores convergen progresivamente al valor real esperado bajo la política actual, y esta mejora en cada iteración.

Este ejemplo muestra de forma clara cómo los métodos Monte Carlo para control permiten ajustar las decisiones del agente sin necesidad de conocer el modelo del entorno, confiando únicamente en la información observada y en estrategias simples de exploración. La clave está en la iteración: cada episodio mejora las estimaciones, y con ello la política.

Perfecto. A continuación se desarrolla la sección dedicada al **control off-policy mediante métodos Monte Carlo**, manteniendo el estilo expositivo del formato apuntes, con claridad conceptual, precisión matemática y orientación didáctica. El objetivo es introducir el aprendizaje a partir de una política distinta a la que se desea evaluar, junto con el uso del *importance sampling* como herramienta para corregir el sesgo.

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

Una vez calculado este peso, puede utilizarse para ajustar las estimaciones de retorno. Por ejemplo, si se desea estimar el valor $V^\pi(s)$ mediante first-visit Monte Carlo, se acumulan los retornos $G_t$ ponderados por $\rho(\tau)$, de modo que:

$$
V^\pi(s) = \frac{\sum_{\tau \in \mathcal{E}} \rho(\tau) \cdot G_t(\tau)}{\sum_{\tau \in \mathcal{E}} \rho(\tau)}
$$

donde $\mathcal{E}_s$ es el conjunto de episodios en los que se visitó el estado $s$ por primera vez. De forma análoga, puede calcularse $Q^\pi(s, a)$ ponderando solo aquellos episodios donde se observó el par $(s, a)$.

Este enfoque es estadísticamente consistente: si se observa un número suficiente de episodios generados por $\mu$ y se aplican correctamente los pesos, las estimaciones convergen a los valores reales de la política $\pi$. Sin embargo, la principal dificultad es la **alta varianza** que puede producirse. Si $\pi$ y $\mu$ son muy distintas, los pesos de importancia pueden volverse inestables y generar estimaciones poco fiables. Este problema es especialmente agudo en secuencias largas, donde el producto de muchos cocientes puede crecer o decrecer exponencialmente.

Por esta razón, los métodos off-policy con Monte Carlo suelen utilizar variantes **con truncamiento o suavizado de pesos**, como el *importance sampling ponderado*, o bien recurrir a aproximaciones por *bootstrapping* en métodos TD, que ofrecen menor varianza a costa de introducir algo de sesgo.

El valor fundamental de estos métodos es que permiten **reutilizar experiencia** generada por otras políticas, agentes o simulaciones, lo que los hace extremadamente útiles en contextos donde la exploración activa es costosa, arriesgada o limitada. También permiten **evaluar múltiples políticas** a partir de un único conjunto de datos, lo que los convierte en una herramienta crucial en aprendizaje por refuerzo offline o en simulación basada en datos históricos.

###### Ejemplo: estimación de $V^\pi(s)$ con Monte Carlo off-policy

Supongamos un entorno muy simple con tres estados $s_0$, $s_1$ y $s_2$, y dos acciones posibles en cada estado: $a_0$ y $a_1$. El objetivo es estimar el valor $V^\pi(s_1)$, es decir, el retorno esperado al partir del estado $s_1$ y seguir la política $\pi$. La política objetivo $\pi$ es **determinista** y en $s_1$ siempre selecciona $a_1$.

Sin embargo, los episodios observados provienen de una política de comportamiento $\mu$ que actúa de forma **aleatoria uniforme**, seleccionando $a_0$ y $a_1$ con probabilidad 0.5 en cada estado. Esto hace necesario reponderar los episodios observados con un factor de *importance sampling* que corrija esta diferencia de comportamiento.

Supongamos que tenemos los siguientes tres episodios observados, todos iniciando en $s_1$:

- Episodio 1: $(s_1, a_1) \to s_2$, $G = 1$
- Episodio 2: $(s_1, a_0) \to s_0$, $G = 0$
- Episodio 3: $(s_1, a_1) \to s_2$, $G = 1$

Queremos estimar $V^\pi(s_1)$ mediante first-visit Monte Carlo off-policy. Solo consideraremos aquellos episodios en los que se haya seguido la acción que la política $\pi$ habría tomado, es decir, aquellos donde se ha ejecutado $a_1$ en $s_1$.

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

Aplicamos ahora la estimación de $V^\pi(s_1)$ como promedio ponderado:

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

En el siguiente módulo se abordarán los **métodos basados en diferencias temporales (TD)**, que introducen una idea clave: la posibilidad de actualizar los valores sin esperar al final del episodio. Este cambio de paradigma abre la puerta a algoritmos más eficientes y generalizables, que constituyen el núcleo de la mayoría de sistemas modernos de aprendizaje por refuerzo.

### Diferencias temporales (TD-learning)

#### Introducción

Los métodos de aprendizaje por refuerzo basados en Monte Carlo, que hemos estudiado previamente, parten de una premisa fundamental: la actualización del conocimiento del agente se produce únicamente al finalizar cada episodio. Esta característica implica que el agente debe esperar a alcanzar un estado terminal para poder estimar el retorno total obtenido y actualizar el valor del estado de partida o de las acciones ejecutadas.

Este enfoque, aunque conceptualmente sólido, presenta limitaciones importantes en la práctica. Por un lado, no todos los entornos tienen episodios claramente definidos o finitos. En muchos escenarios reales, como la navegación continua o la interacción sin fin con un entorno, el aprendizaje basado únicamente en episodios resulta inviable. Por otro lado, incluso en entornos episódicos, esperar al final del episodio puede suponer una **ineficiencia significativa** en términos de tiempo y capacidad de adaptación.

Frente a estas limitaciones surge un enfoque alternativo: **el aprendizaje por diferencias temporales** (Temporal-Difference Learning o TD-learning). La idea clave es que el agente no necesita esperar a conocer el retorno completo de un episodio para actualizar sus estimaciones, sino que puede hacerlo de manera **inmediata** tras cada transición observada. Este procedimiento se conoce como **bootstrapping**, y consiste en utilizar una **estimación actualizada del valor futuro** para corregir la estimación del valor presente.

> [!note]
>
> **¿Qué significa bootstrapping en TD-learning?**
>
> En el aprendizaje por diferencias temporales, el término **bootstrapping** hace referencia a la **forma en que el agente mejora su conocimiento sin necesidad de esperar información completa del entorno**. En lugar de calcular el valor de un estado basándose en la suma total de recompensas futuras observadas, como se hace en Monte Carlo, el agente **ajusta su estimación actual utilizando como guía su propia predicción del estado siguiente**.
>
> Imaginemos que el agente está en el estado $s_t$, realiza una acción y transiciona al estado $s_{t+1}$, obteniendo una recompensa inmediata. En ese momento, el agente ya tiene alguna idea de cuál podría ser el valor del nuevo estado (aunque esa idea sea aún imperfecta). Lo que hace entonces es **usar esa estimación parcial como punto de partida para actualizar su estimación del estado anterior**. No espera a recorrer todo el episodio. Aprende de manera local e incremental, corrigiendo su creencia paso a paso con cada transición observada.
>
> Este tipo de razonamiento es autorreferencial: el agente **se apoya en sus propias predicciones para aprender**, del mismo modo que alguien que quiere estimar la altitud de una montaña podría hacerlo comparando pequeñas diferencias entre puntos consecutivos, sin necesidad de conocer la altitud total desde la base hasta la cima.
>
> Desde un punto de vista práctico, esta idea de bootstrapping permite que el aprendizaje sea **mucho más eficiente y adaptativo**, ya que cada transición puede utilizarse inmediatamente para afinar las estimaciones. Pero al mismo tiempo, introduce nuevos desafíos: como el agente se apoya en sus propias predicciones, **puede propagar errores si estas no son suficientemente precisas**. Por ello, el proceso requiere mecanismos adecuados de control, como tasas de aprendizaje y exploración suficientes.
>
> Este es uno de los pilares que diferencian al TD-learning de los enfoques anteriores. En lugar de depender del modelo del entorno (como en programación dinámica) o de episodios completos (como en Monte Carlo), el TD-learning **construye el conocimiento en tiempo real y sobre la marcha**, confiando en su capacidad de corregirse con la experiencia.

Insistamos en la idea clave: En lugar de acumular todas las recompensas futuras para calcular el retorno, el aprendizaje por TD ajusta la estimación del valor de un estado en función de la recompensa inmediata obtenida y del valor estimado del siguiente estado. Esta capacidad de aprendizaje paso a paso convierte a los métodos TD en **algoritmos online**, capaces de aprender en tiempo real, adaptarse dinámicamente y operar en entornos no estacionarios.

Al final la clave estaría en el denominado **error de predicción**, también conocido como **TD-error**, que refleja la diferencia entre lo que el agente creía que iba a ocurrir y lo que efectivamente observa. Esta señal de error es la base de todas las actualizaciones y constituye uno de los pilares teóricos y prácticos del aprendizaje por refuerzo.

> **TD-Learning: Una analogía**
>
> Imaginemos que un excursionista atraviesa un valle poco conocido y debe estimar la altitud del terreno en cada punto del camino, con el objetivo de encontrar el camino más llano o la ruta de menor pendiente. No dispone de un mapa ni puede ver todo el trayecto por adelantado. Tampoco puede esperar a terminar toda la excursión para sacar conclusiones, porque necesita ir ajustando su estrategia de marcha en tiempo real. 
>
> En cada paso que da, el excursionista siente si ha subido o bajado respecto al punto anterior, y a partir de esa información ajusta su percepción de cuán elevada era la zona por la que acaba de pasar. No necesita llegar al final del trayecto para tener una idea aproximada de cómo varía la altitud, sino que puede **estimar la altitud relativa de un punto basándose en la diferencia con el siguiente**. Su conocimiento se construye de forma progresiva, haciendo pequeñas correcciones locales a medida que avanza.
>
> Este comportamiento refleja con bastante precisión la lógica del TD-learning. El valor de un estado no se ajusta observando todo el retorno futuro hasta el final del episodio, sino utilizando únicamente la **recompensa inmediata** y la **estimación actual del siguiente estado** como una predicción de lo que está por venir. Se aprende **de forma local y continua**, en lugar de global y diferida.
>
> Por esto se dice que el aprendizaje por TD combina la experiencia directa del entorno (como el descenso real percibido por el excursionista) con sus propias predicciones (la estimación previa de altitud), y ajusta su conocimiento en función del **error entre lo esperado y lo observado**.Esta capacidad de adaptación progresiva es lo que convierte al aprendizaje por diferencias temporales en una herramienta eficaz para entornos dinámicos, extensos o parcialmente desconocidos.
>

##### TD-Learning: Los mejor de dos mundos

Para situar adecuadamente el aprendizaje por diferencias temporales (TD-learning) en el panorama del aprendizaje por refuerzo, conviene compararlo con los dos enfoques clásicos que lo flanquean: la programación dinámica (PD) y los métodos Monte Carlo (MC). Cada uno de estos paradigmas representa una estrategia distinta para aprender el valor de los estados y acciones, con sus propias fortalezas y limitaciones.

La programación dinámica asume un conocimiento completo del modelo del entorno. Es decir, el agente conoce perfectamente la función de transición $\mathcal{P}(s' \mid s, a)$ y la función de recompensa $\mathcal{R}(s, a)$, lo que le permite planificar su comportamiento sin necesidad de interacción directa con el entorno. Sin embargo, este enfoque no es viable cuando no se dispone de un modelo explícito o cuando el entorno es demasiado complejo para modelarlo de forma precisa. Además, la PD requiere evaluar políticas mediante iteraciones globales sobre todos los estados, lo que impide una actualización en tiempo real.

En el extremo opuesto se sitúan los métodos Monte Carlo, que no requieren conocimiento del modelo y aprenden directamente de la experiencia. El agente interactúa con el entorno, genera episodios completos y calcula el retorno acumulado a partir de estos. Este enfoque es muy flexible y aplicable a una gran variedad de contextos, pero tiene la limitación de que necesita esperar a que el episodio finalice para realizar cualquier actualización. Esto lo convierte en un método poco eficiente en entornos con episodios largos o indefinidos.

En este contexto, TD-learning se sitúa en un punto intermedio entre ambos enfoques, y por eso se afirma que toma lo mejor de los dos mundos.

Por un lado, como ocurre en los métodos Monte Carlo, no requiere conocimiento del modelo del entorno: las transiciones $(s_t, a_t, r_{t+1}, s_{t+1})$ se obtienen directamente de la experiencia. Esto hace que sea aplicable en entornos reales, incluso cuando la dinámica del entorno es desconocida o no se puede modelar de forma explícita.

Por otro lado, como en la programación dinámica, aprovecha las propias estimaciones actuales para mejorar el conocimiento del agente. Es decir, no necesita esperar al final del episodio: puede actualizar el valor del estado actual basándose en la recompensa inmediata y en su estimación del valor del siguiente estado, utilizando la ecuación de Bellman como principio de actualización. Este proceso se conoce como bootstrapping y permite una actualización incremental y online del conocimiento.

Gracias a esta combinación, el TD-learning puede aprender en entornos continuos o no episodicos, adaptarse a cambios en la dinámica del entorno, y hacerlo de forma eficiente desde las primeras interacciones. Esto lo convierte en un paradigma fundamental en aprendizaje por refuerzo, y en la base de muchos de los algoritmos más potentes y generalizables, tanto en control clásico como en aprendizaje profundo.

En las secciones siguientes exploraremos cómo se formaliza esta idea a través del algoritmo TD(0), cómo se extiende al control mediante SARSA y Q-learning, y qué ventajas ofrece frente a otras estrategias. Comenzaremos por analizar en detalle el caso más simple: el aprendizaje del valor de una política fija mediante actualizaciones por TD.

#### Algoritmo TD(0)

El algoritmo TD(0) es el ejemplo más simple y representativo de los métodos de aprendizaje por diferencias temporales. Su objetivo es estimar la función de valor de una política fija $\pi$, es decir, aproximar $V^\pi(s)$ para cada estado $s$ del entorno, utilizando exclusivamente las transiciones observadas al interactuar con dicho entorno.

Lo que distingue a TD(0) de los métodos Monte Carlo es que **no espera a que termine un episodio completo para actualizar la estimación de valor**. En su lugar, realiza una actualización inmediata tras cada transición $(s_t, a_t, r_{t+1}, s_{t+1})$ observada. Esta actualización se basa en una estimación del retorno futuro a partir de la recompensa inmediata y el valor estimado del estado siguiente, sin necesidad de conocer toda la secuencia posterior. 

Este enfoque se apoya en la **ecuación de Bellman para una política fija**, que establece una relación recursiva entre el valor de un estado y el valor esperado de sus sucesores. TD(0) aprovecha esta estructura para aplicar **actualizaciones paso a paso**, corrigiendo la estimación de $V(s_t)$ mediante una fórmula basada en la experiencia más reciente.

Para llegar a una expresión algorítmica que nos permita hacer cálculos no podremos usar la ecuación de Bellman tal y como lo hicimos en la sección correspondiente a la Programación Dinámica. Tendremos que partir de la función estado-valor inicial

$$
V^\pi(s) = \mathbb{E} \left[ G_t \mid s_t = s \right] = \mathbb{E} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \,\bigg|\, s_t = s \right]
$$

Que ya sabemos que representa la suma esperada de recompensas futuras descontadas, comenzando desde el estado $s$ en el tiempo $t$, bajo la política $\pi$.

Podemos descomponer el retorno de este modo

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

Esta expresión es simplemente una **reescritura recursiva del retorno**. La recompensa total que espera recibir el agente a partir del instante $t$ es la recompensa inmediata $r_{t+1}$ más el retorno futuro desde el estado siguiente, descontado por el factor $\gamma$. A través de una serie de transformaciones puede demostrarse que en este caso la **ecuación de Bellman para $V^\pi$** Sería

$$
V^\pi(s_t) = \mathbb{E} \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t \right]
$$

Pero como en este tipo de problemas en la práctica no conocemos ni la distribución de recompensas ni las probabilidades de transición, **no podemos calcular directamente el valor esperado**. Así, lo que hace lo que hace TD(0) es **usar una sola muestra de la experiencia** para aproximar esta expectativa. Es decir, en lugar de tomar el valor esperado, toma directamente la observación puntual de la transición. Este es el **paso de actualización de TD(0)**. No calcula la expectativa, pero se **apoya en una estimación puntual**, y utiliza una tasa de aprendizaje $\alpha$ para ajustar gradualmente el valor de $s_t$ hacia una mejor aproximación. Este mecanismo es el que se denomina **actualización por diferencias temporales** y constituye el núcleo del algoritmo.

Formalmente, la regla de actualización de TD(0) se expresaría como:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
$$

Aquí:

- $\alpha \in (0,1]$ es la tasa de aprendizaje, que determina cuánto se ajusta la estimación actual en función de la nueva información.
- $r_{t+1}$ es la recompensa inmediata obtenida tras ejecutar $a_t$ en $s_t$.
- $V(s_{t+1})$ es la estimación actual del valor del siguiente estado.
- $\gamma \in [0,1)$ es el factor de descuento.

El término entre corchetes se denomina **TD-error** o error de diferencia temporal:

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

Este error representa la diferencia entre lo que el agente esperaba (su valor actual de $s_t$) y la recompensa más la estimación del estado siguiente. Si el error es positivo, indica que el estado $s_t$ resultó ser más valioso de lo previsto; si es negativo, lo contrario. El agente ajusta entonces su estimación de $V(s_t)$ en esa dirección, pero sin reemplazarla completamente, sino mezclándola con la nueva información.

Esta actualización ocurre **tras cada paso individual**, lo que convierte a TD(0) en un algoritmo **online** e **incremental**. En entornos estocásticos, el proceso se repite a lo largo de múltiples episodios o trayectorias, y con una elección adecuada de $\alpha$, las estimaciones de valor convergen a los verdaderos valores esperados bajo la política $\pi$.

Desde el punto de vista práctico, TD(0) es más eficiente que Monte Carlo, ya que puede comenzar a aprender desde la primera transición observada. Además, permite el aprendizaje continuo en tareas no episódicas o de duración indefinida.

COmo ya se avanzó en la introducción, este proceso de ajuste se denomina bootstrapping. El agente actualiza su conocimiento basándose no en datos reales del futuro, sino en su propia estimación del valor del siguiente estado. En lugar de esperar a conocer el retorno completo, como ocurre en Monte Carlo, el agente utiliza su predicción actual como punto de apoyo para corregirse progresivamente con cada nueva transición. Esta idea de autoajuste permite que el aprendizaje sea mucho más eficiente y continuo.

TD(0) es, por tanto, un algoritmo de **predicción**. Su propósito **no es aprender una política óptima,** sino cuantificar con precisión el retorno esperado bajo una política fija. Esta evaluación resulta especialmente útil cuando se desea analizar o comparar distintas estrategias antes de tomar decisiones, o cuando se quiere utilizar la evaluación como paso intermedio en métodos de control.

A diferencia de la programación dinámica, TD(0) no requiere conocer la función de transición del entorno ni la función de recompensa exacta. Tampoco necesita esperar al final del episodio, como Monte Carlo. Esto lo convierte en una herramienta versátil para tareas donde el entorno es desconocido o continuo, y donde las actualizaciones online e incrementales son necesarias.

##### Ejemplo comparativo: TD(0) con política determinista y estocástica

Vamos a trabajar con un entorno lineal sencillo compuesto por tres estados: $s_0$, $s_1$ y $s_2$, donde $s_2$ es un estado terminal o absorbente.

Desde cada estado no terminal, el agente puede elegir entre dos acciones: `'avanzar'` (pasar al siguiente estado) o `'quedarse'` (permanecer en el mismo estado). La dinámica del entorno es determinista: la acción `'avanzar'` lleva al siguiente estado y `'quedarse'` mantiene al agente en el mismo.

Las recompensas están definidas del siguiente modo:

- Transición $s_0 \to s_1$: recompensa 0  
- Transición $s_1 \to s_2$: recompensa 1  
- Cualquier otra transición: recompensa 0

El objetivo es estimar la función de valor $V^\pi(s)$ para cada política, usando TD(0) con $\alpha = 0{,}5$ y $\gamma = 1$.

---

###### Caso 1: política determinista

Definimos la política $\pi_d$ como:

- En $s_0$: siempre `'avanzar'`
- En $s_1$: siempre `'avanzar'`

Simulamos un episodio siguiendo esta política:

1. $s_0 \xrightarrow{\text{avanzar}} s_1$, $r = 0$
2. $s_1 \xrightarrow{\text{avanzar}} s_2$, $r = 1$

Valores iniciales: 
$V(s_0) = 0$, $V(s_1) = 0$, $V(s_2) = 0$

**Actualizaciones**:

Paso 1: $s_0 \to s_1$ 

$$
V(s_0) \leftarrow 0 + 0{,}5 \cdot (0 + 0 - 0) = 0
$$

Paso 2: $s_1 \to s_2$ 

$$
V(s_1) \leftarrow 0 + 0{,}5 \cdot (1 + 0 - 0) = 0{,}5
$$

Después de un episodio: 

$$
V(s_0) = 0, V(s_1) = 0{,}5, V(s_2) = 0
$$

El valor de $s_1$ refleja el hecho de que si el agente lo alcanza, el retorno esperado es 1 (porque llega a $s_2$ con recompensa 1). El valor de $s_0$ permanece en cero, pero comenzará a crecer en siguientes episodios, a medida que se propague el valor desde $s_1$.

---

###### Caso 2: política estocástica

Definimos ahora una política estocástica $\pi_s$ como:

- En $s_0$: 
  - $\pi_s(\text{avanzar} \mid s_0) = 0{,}7$ 
  - $\pi_s(\text{quedarse} \mid s_0) = 0{,}3$
- En $s_1$: 
  - $\pi_s(\text{avanzar} \mid s_1) = 0{,}5$ 
  - $\pi_s(\text{quedarse} \mid s_1) = 0{,}5$

Simulamos un episodio generado según esta política. Supongamos que en este caso se producen las siguientes transiciones:

1. $s_0 \xrightarrow{\text{quedarse}} s_0$, $r = 0$
2. $s_0 \xrightarrow{\text{avanzar}} s_1$, $r = 0$
3. $s_1 \xrightarrow{\text{avanzar}} s_2$, $r = 1$

Valores iniciales:

$$
V(s_0) = 0, V(s_1) = 0, V(s_2) = 0
$$

**Actualizaciones**:

Paso 1: $s_0 \to s_0$ 

$$
V(s_0) \leftarrow 0 + 0{,}5 \cdot (0 + 0 - 0) = 0
$$

Paso 2: $s_0 \to s_1$ 

$$
V(s_0) \leftarrow 0 + 0{,}5 \cdot (0 + 0 - 0) = 0
$$

Paso 3: $s_1 \to s_2$ 

$$
V(s_1) \leftarrow 0 + 0{,}5 \cdot (1 + 0 - 0) = 0{,}5
$$

Resultado tras un episodio: 

$$
V(s_0) = 0, V(s_1) = 0{,}5, V(s_2) = 0
$$

Aunque el valor final es el mismo que en el caso determinista, hay una diferencia importante: bajo una política estocástica, **no todas las trayectorias conducen siempre a $s_2$**. Por tanto, si repetimos muchos episodios generados con $\pi_s$, observaremos que $V(s_0)$ crecerá **más lentamente** que en el caso determinista, ya que no todas las trayectorias propagan recompensa.

Este ejemplo permite comparar claramente dos situaciones:

- Con política **determinista**, el valor de cada estado se aprende rápidamente porque el comportamiento es predecible.
- Con política **estocástica**, el valor aprendido refleja un **promedio ponderado** sobre todas las trayectorias posibles, y puede estabilizarse en valores menores dependiendo de la frecuencia con la que se alcanzan los estados recompensados.

Ambos casos son válidos para TD(0), pero ilustran cómo el tipo de política afecta al ritmo de aprendizaje y al valor final estimado.

#### Métodos TD para control

##### Introducción

En los apartados anteriores hemos visto cómo es posible utilizar métodos de tipo TD para estimar el valor de una política fija. Sin embargo, el verdadero objetivo en la mayoría de problemas de aprendizaje por refuerzo no es simplemente evaluar políticas, sino aprender una que sea óptima, es decir, que maximice el retorno esperado del agente en cada estado.

Para abordar este objetivo, los algoritmos de control basados en diferencias temporales introducen un componente esencial: la mejora de la política a lo largo del tiempo. Esto se logra manteniendo una estimación de la utilidad de cada acción en cada estado y actualizándola progresivamente con la experiencia.

A diferencia de los métodos de predicción, que solo estiman el valor de los estados, los métodos de control trabajan directamente con la función acción-valor. Este cambio permite tomar decisiones sobre qué acción ejecutar en un estado dado, sin depender de un modelo del entorno ni de cálculos de planificación.

El proceso general en estos métodos es el siguiente: el agente observa una transición en la interacción con el entorno, que consiste en un estado actual, una acción tomada, una recompensa obtenida y un nuevo estado alcanzado. Con esta información, el agente actualiza su estimación del valor de la acción que ha ejecutado. Posteriormente, la política puede modificarse favoreciendo aquellas acciones que se han mostrado más prometedoras.

Existen distintas variantes de este esquema, dependiendo de cómo se utilicen las acciones futuras en el proceso de actualización. Dos de los métodos más representativos son SARSA y Q-learning. Ambos permiten al agente aprender políticas cada vez mejores a partir de su experiencia directa, pero difieren en la manera en que se enfrentan al dilema exploración-explotación y en el tipo de política que efectivamente aprenden.

En el caso de SARSA, el aprendizaje se ajusta a la política que el agente está siguiendo realmente. Esto significa que si el agente incluye exploración en su comportamiento, como en las estrategias $\epsilon$-greedy, el conocimiento aprendido reflejará también esa misma estrategia. Este enfoque se conoce como aprendizaje on-policy, y tiende a ser más conservador y estable en entornos ruidosos o donde las decisiones arriesgadas pueden tener consecuencias negativas.

Por el contrario, Q-learning busca aprender directamente el valor de la política óptima, independientemente de cómo el agente actúe durante la fase de aprendizaje. En este caso, el agente puede explorar libremente, pero siempre estima el valor suponiendo que en el futuro actuará de forma óptima. Este enfoque se conoce como aprendizaje off-policy, y tiene la ventaja de converger más rápidamente hacia políticas de alto rendimiento en muchos contextos, aunque puede ser más sensible a una exploración excesiva o mal controlada.

Ambos métodos ilustran el principio fundamental del control mediante aprendizaje por refuerzo: la mejora continua de la política a partir de la experiencia directa, sin necesidad de conocer el modelo del entorno. A partir de la próxima sección, estudiaremos en detalle estas dos aproximaciones, sus fundamentos matemáticos y su comportamiento en distintos escenarios.

###### De nuevo el balance Explotación-Exploración

Ya hemos visto en secciones anteriores como uno de los retos fundamentales en los algoritmos de control del aprendizaje por refuerzo es decidir **qué acción debe ejecutarse en cada momento**, no solo para obtener buenas recompensas inmediatas, sino para **aprender la mejor política posible a largo plazo**. Esta situación nos obliga a retomar una idea clave que ya hemos discutido anteriormente: el equilibrio entre **explotación** y **exploración**.

Recordemos: La **explotación** consiste en seleccionar las acciones que actualmente se consideran mejores, es decir, aquellas para las que el agente ha estimado un alto valor esperado. Esta estrategia favorece decisiones seguras y rentables a corto plazo. Sin embargo, si el agente se limita a explotar lo que ya conoce, corre el riesgo de no descubrir alternativas potencialmente mejores que aún no han sido exploradas. La **exploración**, por el contrario, implica elegir acciones que pueden parecer subóptimas según el conocimiento actual, pero que ofrecen la posibilidad de obtener nueva información. Es precisamente esta información la que permite refinar las estimaciones de valor y, a la larga, encontrar políticas más efectivas.

En los métodos de predicción esta tensión ya estaba presente, pero su efecto era limitado: el objetivo era estimar los valores bajo una política dada, y por tanto el impacto de las acciones elegidas se acotaba a la calidad de esa estimación. Sin embargo, **en los algoritmos de control el agente no solo evalúa, sino que también actúa para mejorar**, y por tanto el dilema exploración-explotación adquiere una dimensión decisiva.

La necesidad de explorar se traduce en la práctica en el uso de políticas **estocásticas** o **deterministas con exploración**, como por ejemplo las políticas $\epsilon$-greedy, donde el agente elige con probabilidad $1 - \epsilon$ la acción con mayor valor estimado, y con probabilidad $\epsilon$ una acción aleatoria. Este mecanismo permite controlar la exploración de forma gradual: un valor alto de $\epsilon$ favorece la recolección de información en las primeras fases del entrenamiento, mientras que valores decrecientes permiten consolidar el conocimiento adquirido y optimizar la política en fases posteriores.

En los algoritmos que estudiaremos a continuación, como **SARSA** y **Q-learning**, este balance será gestionado de forma distinta. En SARSA, el agente aprende sobre la política que ejecuta realmente (incluida su exploración), mientras que en Q-learning se estima el valor de la política óptima incluso cuando se comporta de otro modo.

Por esto es por lo que el dilema entre exploración y explotación debe mantenerse presente como **marco conceptual esencial** durante todo el estudio de los métodos de control. No es solo una cuestión de estrategia de comportamiento, sino una dimensión crítica que condiciona cómo y qué aprende el agente en cada interacción.

##### Control on-policy: el algoritmo SARSA

El aprendizaje por refuerzo no solo permite evaluar políticas, sino también aprender políticas óptimas directamente desde la experiencia. Para lograrlo, los algoritmos de control permiten actualizar estimaciones de valor y mejorar decisiones de forma iterativa. En este contexto, el algoritmo SARSA representa una estrategia de aprendizaje *on-policy*, es decir, que aprende sobre la política que el propio agente está ejecutando.

La motivación detrás de SARSA parte de una idea fundamental: el agente debe adaptar su comportamiento no solo en función de los resultados que observa, sino también teniendo en cuenta la política real que sigue, incluida la exploración. Esto implica que la política no se asume óptima durante el aprendizaje, sino que evoluciona gradualmente conforme mejora la estimación de los valores de las acciones.

SARSA toma su nombre de los cinco elementos que intervienen en cada transición observada por el agente: *State–Action–Reward–State–Action*, es decir, $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$. A partir de esta secuencia, el algoritmo actualiza el valor de $Q(s_t, a_t)$, es decir, la utilidad estimada de ejecutar la acción $a_t$ en el estado $s_t$.

Este valor se ajusta con base en el retorno observado al seguir la política vigente, lo que incluye la posibilidad de que el agente explore en el siguiente estado. Esta es la principal diferencia respecto a enfoques off-policy: en SARSA, el aprendizaje refleja exactamente el comportamiento real del agente, con todas las implicaciones que conlleva la exploración.

Desde un punto de vista formal, SARSA se basa en la ecuación de Bellman para políticas fijas, aplicada a funciones acción-valor:

$$
Q^\pi(s_t, a_t) = \mathbb{E} \left[ r_{t+1} + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t, a_t \right]
$$

Esta ecuación establece que el valor de una acción es igual a la recompensa inmediata más el valor esperado de las acciones futuras, suponiendo que se sigue la misma política $\pi$.

Para llevar esto al terreno del aprendizaje, el agente calcula un **error de predicción** (o error de TD), que mide la diferencia entre el valor actual estimado y el retorno observado:

$$
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
$$

Con este error, se realiza una actualización incremental sobre el valor de la acción observada:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \, \delta_t
$$

donde $\alpha$ es la tasa de aprendizaje que regula cuánto se ajusta la nueva estimación.

El proceso general de SARSA se repite en cada paso de la interacción con el entorno:

1. El agente observa el estado actual $s_t$.
2. Elige una acción $a_t$ siguiendo una política $\epsilon$-greedy sobre la función $Q$.
3. Ejecuta la acción y recibe una recompensa $r_{t+1}$ junto con un nuevo estado $s_{t+1}$.
4. Elige una acción $a_{t+1}$ en el nuevo estado, de nuevo según la política vigente.
5. Actualiza $Q(s_t, a_t)$ utilizando la fórmula anterior.

Este procedimiento garantiza que el agente no solo actualiza sus estimaciones de forma coherente con su experiencia, sino también que la política aprendida será consistente con la forma en que se ha comportado durante el entrenamiento.

Una ventaja clave de SARSA es que refleja fielmente los efectos de la exploración. Si el agente actúa con cautela en un entorno incierto, el valor que aprenderá para cada acción tendrá en cuenta las consecuencias de esa cautela. Esto lo convierte en un método especialmente robusto en entornos con transiciones estocásticas o recompensas ruidosas.

Para garantizar la convergencia hacia una política óptima, es necesario reducir gradualmente el grado de exploración durante el entrenamiento. Una política $\epsilon$-greedy con $\epsilon \to 0$ permite que el comportamiento converja hacia el aprovechamiento del conocimiento aprendido, asegurando un equilibrio adecuado entre exploración inicial y explotación final.

SARSA constituye así un puente entre evaluación y mejora de políticas, permitiendo que el agente aprenda **paso a paso** y de forma **coherente con su propia estrategia de actuación**, sin necesidad de conocer el modelo del entorno ni planificar a largo plazo. Esta característica lo convierte en una herramienta eficaz y versátil en una amplia gama de problemas reales.

Perfecto. Vamos a construir un **ejemplo práctico y didáctico de aplicación del algoritmo SARSA**, basado en un entorno sencillo y completamente interpretable. Este ejemplo está diseñado para ser comprensible desde el punto de vista del alumno, permitiendo seguir **paso a paso** el ciclo de actualización SARSA y visualizar cómo evoluciona el aprendizaje del agente.

---

###### **Un ejemplo práctico de uso del algoritmo SARSA**  
Vamos a construir un **ejemplo práctico y didáctico de aplicación del algoritmo SARSA**, basado en un entorno sencillo y completamente interpretable. El entorno será el ya habitual del tablero unidimensional de tres casillas

Como sabemos, el entorno consiste en un tablero lineal con tres posiciones:

- $s_0$: hueco (estado no deseado)
- $s_1$: posición inicial del agente
- $s_2$: estado meta (absorbe el agente y da recompensa)

El agente puede moverse:

- a la izquierda: $a_0$
- a la derecha: $a_1$

El objetivo es alcanzar la meta $s_2$ desde el estado inicial $s_1$.

La dinámica del entorno es la siguiente:

- Desde $s_1$:
  - acción $a_0$ → transita a $s_0$ con recompensa 0
  - acción $a_1$ → transita a $s_2$ con recompensa 1
- Desde $s_0$ y $s_2$, el entorno es **absorbente** (el agente no se mueve y la recompensa es siempre 0)

Y la configuración inicial puede resumirse así:

- Inicializamos $Q(s, a) = 0$ para todo $(s, a)$
- Política $\epsilon$-greedy con $\epsilon = 0.1$
- Tasa de aprendizaje $\alpha = 0.5$
- Factor de descuento $\gamma = 1$

Procedamos ahora a la simulación de un episodio con SARSA. Para ello supongamos que el agente comienza en $s_1$ y elige acción $a_1$ (derecha) siguiendo su política.

1. El agente está en $s_1$, elige $a_1$.
2. El entorno responde con:
   - nuevo estado $s_2$
   - recompensa $r = 1$
3. El agente selecciona siguiente acción $a'$ en $s_2$ (por política), por ejemplo, $a_0$.
4. Se actualiza:

$$
Q(s_1, a_1) \leftarrow Q(s_1, a_1) + \alpha \left[ r + \gamma Q(s_2, a_0) - Q(s_1, a_1) \right]
$$

Con los valores iniciales $Q = 0$, se tiene:

$$
Q(s_1, a_1) \leftarrow 0 + 0.5 \cdot (1 + 1 \cdot 0 - 0) = 0.5
$$

Ahora, supongamos otro episodio donde el agente en $s_1$ elige acción $a_0$:

1. Estado $s_1$, acción $a_0$
2. Transición a $s_0$, recompensa $r = 0$
3. Siguiente acción desde $s_0$ es $a_1$
4. Actualización:

$$
Q(s_1, a_0) \leftarrow Q(s_1, a_0) + \alpha \left[ 0 + \gamma \cdot Q(s_0, a_1) - Q(s_1, a_0) \right]
$$

Como todos los valores siguen en 0:

$$
Q(s_1, a_0) \leftarrow 0
$$

Con suficientes episodios, el valor de $Q(s_1, a_1)$ se irá incrementando, reflejando que esa acción conduce a la recompensa. A su vez, $Q(s_1, a_0)$ permanecerá bajo, ya que nunca lleva al objetivo.

Al final del algoritmo el agente aprende a:

- preferir $a_1$ desde $s_1$, pues es la acción que lleva al estado objetivo
- evitar $a_0$, que lleva al estado no deseado

SARSA, al estar condicionado por la política que efectivamente ejecuta, puede ajustarse al comportamiento real del agente, incluyendo la exploración inducida por $\epsilon$.

##### Control off-policy: el algoritmo Q-learning

En el aprendizaje por refuerzo, el objetivo del control es aprender una política óptima, aquella que maximiza el retorno esperado a largo plazo. Mientras que los métodos *on-policy*, como SARSA, aprenden sobre la política que el agente ejecuta (incluyendo sus componentes exploratorios), los algoritmos *off-policy* separan **la política que se aprende** de **la política que se ejecuta para explorar**. Esta es la esencia del control off-policy, y representa un enfoque más general y, en muchos casos, más potente.

El algoritmo **Q-learning** es el ejemplo paradigmático de control off-policy. En él, el agente puede actuar siguiendo una política $\epsilon$-greedy, o incluso completamente aleatoria, pero **el aprendizaje no se ajusta a esa política ejecutada**, sino que se orienta hacia la estimación de la **política óptima**, definida como aquella que siempre escoge la acción con mayor valor esperado. Es decir, Q-learning aprende como si el agente siempre actuara de forma greedy, aunque en la práctica explore el entorno de otro modo.

Esta disociación entre comportamiento y objetivo permite que Q-learning busque **el mejor comportamiento posible** sin limitarse a las consecuencias inmediatas de las decisiones reales. Por este motivo se dice que es un algoritmo **off-policy**: el valor aprendido no representa la política seguida por el agente, sino aquella que se obtendría al elegir siempre la mejor acción estimada.

Desde un punto de vista formal, Q-learning se basa en una versión particular de la ecuación de Bellman para funciones acción-valor, donde se maximiza explícitamente sobre las acciones disponibles en el siguiente estado:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \cdot \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

Esta actualización no requiere conocer qué acción se ejecutará realmente en el estado $s_{t+1}$, sino que **toma el mejor valor estimado posible**, en consonancia con la política greedy.

Los pasos del algoritmo son los siguientes:

1. El agente observa el estado actual $s_t$.
2. Selecciona una acción $a_t$ según una política de comportamiento, por ejemplo $\epsilon$-greedy.
3. Ejecuta la acción $a_t$, recibe la recompensa $r_{t+1}$ y observa el nuevo estado $s_{t+1}$.
4. Calcula $\max_{a'} Q(s_{t+1}, a')$, el mejor valor posible según la estimación actual.
5. Aplica la actualización sobre $Q(s_t, a_t)$ según la fórmula anterior.

Este proceso se repite en cada paso del entorno, y con el tiempo, Q-learning converge al valor óptimo $Q^*$ bajo ciertas condiciones (visita suficiente de todos los pares $(s,a)$ y tasas de aprendizaje adecuadas).

Una característica importante de Q-learning es que **la acción $a_{t+1}$ no es necesaria** para el cálculo de la actualización. Esto lo diferencia claramente de SARSA, donde sí se requiere conocer cuál será la siguiente acción bajo la política actual. En Q-learning, el agente **simula el comportamiento de una política óptima** incluso si no actúa todavía conforme a ella.

###### Relevancia de Q-learning en el aprendizaje por refuerzo

Q-learning es, históricamente, uno de los algoritmos más influyentes en el desarrollo del aprendizaje por refuerzo moderno. Su capacidad para **aprender políticas óptimas directamente desde la interacción**, sin necesidad de conocer el modelo del entorno, y sin ajustarse estrictamente al comportamiento real del agente, lo ha convertido en el punto de partida de muchas extensiones y desarrollos posteriores.

Entre los motivos de su importancia destacan:

- Su aplicabilidad en entornos complejos, estocásticos y parcialmente observables.
- Su papel como base conceptual de métodos más avanzados, como **Deep Q-Networks (DQN)**.
- Su convergencia probada bajo condiciones razonables y su robustez ante exploración $\epsilon$-greedy.

En resumen, Q-learning proporciona al agente un mecanismo para aprender a actuar **como si ya supiera lo que es óptimo**, permitiéndole mejorar continuamente su política sin depender directamente de su comportamiento actual. Esta característica lo convierte en una herramienta central en cualquier curso avanzado de aprendizaje por refuerzo.

---

###### **Ejemplo práctico de Q-learning**  
Seguimos trabajando con el tablero unidimensional con tres estados. Conservamos la estructura del ejemplo anterior (SARSA), para facilitar la comparación:

- $s_0$: hueco (absorbe sin recompensa)
- $s_1$: estado inicial
- $s_2$: meta (absorbe con recompensa 1)

El agente tiene dos acciones:

- $a_0$: moverse a la izquierda
- $a_1$: moverse a la derecha

Los elementos del MDP serían los siguientes:

- $\mathcal{S} = \{s_0, s_1, s_2\}$
- $\mathcal{A} = \{a_0, a_1\}$
- Función de transición:
  - $T(s_1, a_0) = s_0$, recompensa = 0
  - $T(s_1, a_1) = s_2$, recompensa = 1
  - $s_0$ y $s_2$ son absorbentes
- $\gamma = 1$, $\alpha = 0.5$, política $\epsilon$-greedy con $\epsilon = 0.1$

---

**Inicialización**

Supongamos que todas las entradas de $Q(s, a)$ se inicializan a 0:

| Estado | Acción izquierda ($a_0$) | Acción derecha ($a_1$) |
| ------ | ------------------------ | ---------------------- |
| $s_0$  | 0                        | 0                      |
| $s_1$  | 0                        | 0                      |
| $s_2$  | 0                        | 0                      |

---

**Episodio 1: el agente ejecuta acción $a_1$ desde $s_1$**

1. Estado $s_t = s_1$, acción $a_t = a_1$
2. Transición a $s_{t+1} = s_2$, recompensa $r = 1$
3. $\max_{a'} Q(s_2, a') = \max(0, 0) = 0$
4. Actualización Q-learning:

$$
Q(s_1, a_1) \leftarrow Q(s_1, a_1) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s_2, a') - Q(s_1, a_1) \right]
$$

Sustituyendo:

$$
Q(s_1, a_1) \leftarrow 0 + 0.5 \cdot (1 + 0 - 0) = 0.5
$$

| Estado | $a_0$ | $a_1$ |
| ------ | ----- | ----- |
| $s_1$  | 0     | 0.5   |

---

**Episodio 2: el agente ejecuta acción $a_0$ desde $s_1$**

1. Estado $s_1$, acción $a_0$
2. Transición a $s_0$, recompensa = 0
3. $\max_{a'} Q(s_0, a') = 0$
4. Actualización:

$$
Q(s_1, a_0) \leftarrow 0 + 0.5 \cdot (0 + 0 - 0) = 0
$$

| Estado | $a_0$ | $a_1$ |
| ------ | ----- | ----- |
| $s_1$  | 0     | 0.5   |

---

**Episodio 3: el agente vuelve a ejecutar $a_1$ en $s_1$**

1. Estado $s_1$, acción $a_1$
2. Transición a $s_2$, recompensa = 1
3. $\max_{a'} Q(s_2, a') = 0$
4. Actualización:

$$
Q(s_1, a_1) \leftarrow 0.5 + 0.5 \cdot (1 + 0 - 0.5) = 0.75
$$

| Estado | $a_0$ | $a_1$ |
| ------ | ----- | ----- |
| $s_1$  | 0     | 0.75  |



Con sucesivos episodios, el valor de $Q(s_1, a_1)$ seguirá incrementándose hasta aproximarse a 1, mientras que $Q(s_1, a_0)$ permanecerá en torno a 0. La política greedy asociada a estos valores tenderá a elegir siempre $a_1$ desde $s_1$, es decir, moverse hacia la meta.

Esto ejemplifica cómo Q-learning, incluso actuando de forma exploratoria, **aprende la mejor política posible**, ya que las actualizaciones siempre se hacen respecto a la acción óptima en el siguiente estado.

##### Comparación entre SARSA y Q-learning

Aunque SARSA y Q-learning comparten la misma estructura general como algoritmos de control basados en diferencias temporales, existen diferencias fundamentales en la **naturaleza de la política que aprenden** y en cómo estas diferencias se reflejan en su comportamiento.

SARSA actualiza los valores acción-estado siguiendo estrictamente las decisiones reales del agente. Esto implica que la política aprendida incorpora explícitamente los efectos de la exploración, lo que lo convierte en un enfoque *on-policy*. El valor estimado refleja no solo las decisiones óptimas, sino también aquellas tomadas por la política de comportamiento, incluyendo los pasos exploratorios. Esta característica hace que SARSA tienda a adoptar **comportamientos más cautelosos**, especialmente en entornos donde la exploración puede conducir a estados peligrosos o no deseados.

Por el contrario, Q-learning actúa *off-policy*: aprende como si el agente siempre eligiera la mejor acción posible, incluso cuando en la práctica está explorando. Esto le permite **converger hacia la política óptima más agresivamente**, ya que las actualizaciones no dependen de las acciones realmente ejecutadas, sino de las mejores posibles según las estimaciones actuales. Esto puede resultar en un aprendizaje más rápido o más efectivo en entornos donde la política óptima es claramente definible y la exploración no conlleva riesgos significativos.

En términos de estabilidad, SARSA puede resultar más robusto cuando se combinan exploración y aprendizaje en paralelo, especialmente en entornos no deterministas. En cambio, Q-learning tiende a ser más eficiente a largo plazo cuando las condiciones permiten converger hacia una solución global óptima.

En la práctica, SARSA se utiliza a menudo en entornos donde es importante tener en cuenta el efecto real de las decisiones, como por ejemplo en tareas de navegación con estados peligrosos. Q-learning, en cambio, se ha convertido en el algoritmo de referencia en muchos dominios donde se busca rendimiento óptimo, desde juegos hasta control robótico, y ha sido la base conceptual de desarrollos posteriores como Deep Q-Networks (DQN).

#### Ventajas del aprendizaje incremental

Una de las fortalezas distintivas de los métodos basados en diferencias temporales es su naturaleza **online e incremental**, lo que permite al agente **aprender directamente de la experiencia a medida que actúa**, sin esperar a que finalicen los episodios ni requerir grandes volúmenes de memoria.

Este enfoque resulta especialmente valioso en escenarios donde no es posible almacenar toda la información sobre trayectorias completas o donde los episodios son largos, indefinidos o incluso inexistentes. El hecho de que cada paso pueda proporcionar una actualización útil convierte a estos métodos en herramientas **eficientes y adaptables**, tanto en términos de memoria como de cómputo.

Además, el aprendizaje incremental permite al agente **adaptarse de forma continua a cambios en el entorno**, lo que resulta esencial en entornos **no estacionarios**, donde las dinámicas, las recompensas o los objetivos pueden evolucionar con el tiempo. Métodos como SARSA o Q-learning pueden ajustarse sobre la marcha sin necesidad de reiniciar el entrenamiento.

Por supuesto, esta flexibilidad también conlleva algunas limitaciones. Al depender de estimaciones locales y actualizaciones paso a paso, estos métodos pueden ser más sensibles al ruido o a errores de exploración si no se configuran adecuadamente. Además, pueden requerir técnicas adicionales para estabilizar el aprendizaje cuando se utilizan representaciones complejas, como funciones aproximadoras.

A pesar de estas limitaciones, los algoritmos TD constituyen la **base operativa de los enfoques más avanzados de aprendizaje por refuerzo**, incluyendo aquellos que combinan redes neuronales con Q-learning, los métodos Actor-Critic o las variantes modernas de aprendizaje profundo. Su capacidad para actualizar eficientemente desde experiencia parcial es una propiedad fundamental que ha marcado el desarrollo contemporáneo del RL.

Aquí tienes la **sección final del módulo**, titulada _Comparativa de los métodos clásicos_, elaborada en formato apuntes, sin enumeraciones innecesarias ni líneas divisorias, y con un enfoque claro y didáctico para facilitar la comprensión del alumno.

### Comparativa de los métodos clásicos

Una vez analizados en detalle los tres enfoques fundamentales de aprendizaje por refuerzo —**programación dinámica**, **métodos Monte Carlo** y **métodos basados en diferencias temporales (TD)**— es conveniente establecer una comparación sistemática que permita al estudiante entender cuándo utilizar cada uno, cuáles son sus ventajas y limitaciones, y cómo se relacionan entre sí.

Un primer criterio clave para distinguir estos métodos es el tipo de información que requieren. La programación dinámica se apoya de forma explícita en el conocimiento completo del modelo del entorno, es decir, de la función de transición y de la recompensa esperada. Esto la hace inaplicable en situaciones reales donde dichos elementos no se conocen de antemano. En cambio, tanto Monte Carlo como TD aprenden directamente a partir de la experiencia, sin necesidad de conocer el modelo, lo que los convierte en enfoques model-free.

Otro criterio importante es la necesidad de episodios completos. Los métodos Monte Carlo requieren observar la totalidad de un episodio para poder calcular el retorno asociado a un estado o acción. Esto puede ser una limitación cuando los episodios son largos o no tienen una duración clara. Por el contrario, TD puede actualizar sus estimaciones paso a paso, en línea, sin esperar al final del episodio, lo que le confiere mayor flexibilidad y eficiencia computacional.

Desde el punto de vista de la convergencia, los métodos de programación dinámica proporcionan soluciones exactas bajo condiciones ideales, al resolver directamente las ecuaciones de Bellman. Monte Carlo también converge con suficiente número de episodios, aunque puede ser más lento y sensible al diseño de la política de comportamiento. TD, por su parte, combina rapidez de convergencia con una formulación compatible con entornos continuos, aunque su aproximación puede introducir sesgos si no se gestiona adecuadamente la exploración.

En cuanto al tipo de entornos a los que mejor se adapta cada enfoque, la programación dinámica es más adecuada en simulaciones donde el modelo es accesible y se pueden evaluar múltiples trayectorias sin coste. Monte Carlo encuentra su mayor utilidad en problemas episodios bien definidos, como juegos o simulaciones por trayectorias. Los métodos TD destacan especialmente en **entornos no estacionarios**, o cuando se requiere una **adaptación progresiva** a medida que el agente actúa.

Todo esto se resume de forma sintética en la siguiente tabla:

| Criterio                     | Programación Dinámica         | Monte Carlo                     | Diferencias Temporales (TD)      |
| ---------------------------- | ----------------------------- | ------------------------------- | -------------------------------- |
| Requiere modelo              | Sí                            | No                              | No                               |
| Necesita episodios completos | No                            | Sí                              | No                               |
| Tipo de feedback             | Basado en simulación exacta   | Retorno completo                | Recompensa inmediata + bootstrap |
| Convergencia                 | Exacta (con modelo)           | Estocástica (media de retornos) | Estocástica (valor estimado)     |
| Velocidad de aprendizaje     | Lenta (por barrido total)     | Lenta en tareas largas          | Rápida y online                  |
| Aplicabilidad práctica       | Limitada a entornos conocidos | Buena en juegos y simulaciones  | Alta en entornos reales          |

Esta comparativa permite visualizar cómo cada enfoque ofrece un equilibrio distinto entre realismo, eficiencia y precisión. En la práctica, muchos algoritmos modernos combinan elementos de estos tres paradigmas, aprovechando sus fortalezas para resolver problemas complejos de toma de decisiones en entornos inciertos. Por esto es por lo que resulta esencial dominar sus fundamentos antes de abordar técnicas más avanzadas.

