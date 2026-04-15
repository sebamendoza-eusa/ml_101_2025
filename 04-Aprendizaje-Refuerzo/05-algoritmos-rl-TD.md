# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Algoritmos de Aprendizaje por Refuerzo: Métodos de diferencias temporales

### Introducción

Los métodos de aprendizaje por refuerzo basados en Monte Carlo, que hemos estudiado previamente, parten de una premisa fundamental: la actualización del conocimiento del agente se produce únicamente **al finalizar cada episodio**. Esta característica implica que el agente debe esperar a alcanzar un estado terminal para poder estimar el retorno total obtenido y actualizar el valor del estado de partida o de las acciones ejecutadas.

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

Insistamos en la idea principal: En lugar de acumular todas las recompensas futuras para calcular el retorno, el aprendizaje por TD ajusta la estimación del valor de un estado en función de la recompensa inmediata obtenida y del valor estimado del siguiente estado. Esta capacidad de aprendizaje paso a paso convierte a los métodos TD en **algoritmos online**, capaces de aprender en tiempo real, adaptarse dinámicamente y operar en entornos no estacionarios.

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

#### TD-Learning: Los mejor de dos mundos

Para situar adecuadamente el aprendizaje por diferencias temporales (TD-learning) en el panorama del aprendizaje por refuerzo, conviene compararlo con los dos enfoques clásicos que lo flanquean: la programación dinámica (PD) y los métodos Monte Carlo (MC). Cada uno de estos paradigmas representa una estrategia distinta para aprender el valor de los estados y acciones, con sus propias fortalezas y limitaciones.

La programación dinámica asumía un conocimiento completo del modelo del entorno. Es decir, el agente conocía perfectamente la función de transición $P(s' \mid s, a)$ y la función de recompensa $R(s, a)$, lo que le permitía planificar su comportamiento sin necesidad de interacción directa con el entorno. Sin embargo, este enfoque no era viable cuando no se disponía de un modelo explícito o cuando el entorno era demasiado complejo para modelarlo de forma precisa. Además, la PD requeria evaluar políticas mediante iteraciones globales sobre todos los estados, lo que impedia una actualización en tiempo real.

En el extremo opuesto se situaban los métodos Monte Carlo, que no requerían conocimiento del modelo y aprendian directamente de la experiencia. El agente interactuaba con el entorno, generaba episodios completos y calculaba el retorno acumulado a partir de estos. Este enfoque era muy flexible y aplicable a una gran variedad de contextos, pero tenía la limitación de que necesitaba esperar a que el episodio finalizase para realizar cualquier actualización. Esto lo conviertía en un método poco eficiente en entornos con episodios largos o indefinidos.

En este contexto, TD-learning se sitúa en un punto intermedio entre ambos enfoques, y por eso se afirma que "toma lo mejor de los dos mundos".

Por un lado, como ocurre en los métodos Monte Carlo, no requiere conocimiento del modelo del entorno: las transiciones $(s_t, a_t, r_{t+1}, s_{t+1})$ se obtienen directamente de la experiencia. Esto hace que sea aplicable en entornos reales, incluso cuando la dinámica del entorno es desconocida o no se puede modelar de forma explícita.

Por otro lado, como en la programación dinámica, aprovecha las propias estimaciones actuales para mejorar el conocimiento del agente. Es decir, no necesita esperar al final del episodio: puede actualizar el valor del estado actual basándose en la recompensa inmediata y en su estimación del valor del siguiente estado, utilizando la ecuación de Bellman como principio de actualización. Este proceso se conoce como bootstrapping y permite una actualización incremental y online del conocimiento.

Gracias a esta combinación, el TD-learning puede aprender en entornos continuos o no episodicos, adaptarse a cambios en la dinámica del entorno, y hacerlo de forma eficiente desde las primeras interacciones. Esto lo convierte en un paradigma fundamental en aprendizaje por refuerzo, y en la base de muchos de los algoritmos más potentes y generalizables, tanto en control clásico como en aprendizaje profundo.

En las secciones siguientes exploraremos cómo se formaliza esta idea a través del algoritmo TD(0), cómo se extiende al control mediante SARSA y Q-learning, y qué ventajas ofrece frente a otras estrategias. Comenzaremos por analizar en detalle el caso más simple: el aprendizaje del valor de una política fija mediante actualizaciones por TD.

### Algoritmo TD(0)

El algoritmo TD(0) es el ejemplo más simple y representativo de los métodos de aprendizaje por diferencias temporales. Su objetivo es estimar la función de valor de una política fija $\pi$, es decir, aproximar $v_{\pi}(s)$ para cada estado $s$ del entorno, utilizando exclusivamente las transiciones observadas al interactuar con dicho entorno.

Lo que distingue a TD(0) de los métodos Monte Carlo es que **no espera a que termine un episodio completo para actualizar la estimación de valor**. En su lugar, realiza una actualización inmediata tras cada transición $(s_t, a_t, r_{t+1}, s_{t+1})$ observada. Esta actualización se basa en una estimación del retorno futuro a partir de la recompensa inmediata y el valor estimado del estado siguiente, sin necesidad de conocer toda la secuencia posterior. 

Este enfoque se apoya en la **ecuación de Bellman para una política fija**, que establece una relación recursiva entre el valor de un estado y el valor esperado de sus sucesores. TD(0) aprovecha esta estructura para aplicar **actualizaciones paso a paso**, corrigiendo la estimación de $v(s_t)$ mediante una fórmula basada en la experiencia más reciente.

Para llegar a una expresión algorítmica que nos permita hacer cálculos no podremos usar la ecuación de Bellman tal y como lo hicimos en la sección correspondiente a la Programación Dinámica. Tendremos que partir de la función estado-valor inicial

$$
v_\pi(s) = \mathbb{E} \left[ G_t \mid s_t = s \right] = \mathbb{E} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \,\bigg|\, s_t = s \right]
$$

Que ya sabemos que representa la suma esperada de recompensas futuras descontadas, comenzando desde el estado $s$ en el tiempo $t$, bajo la política $\pi$.



Recordemos ahora también la definición recursiva del retorno:

$$
G_t = r_{t+1} + \gamma G_{t+1}.
$$

Esta igualdad es exacta para una trayectoria concreta. Si tomamos esperanzas condicionadas al estado actual $s_t$, y suponemos que a partir de $t+1$ se sigue la política $\pi$, obteníamos la **ecuación de Bellman para la función de valor**:

$$
v_\pi(s_t) = \mathbb{E}\bigl[ r_{t+1} + \gamma v_\pi(s_{t+1}) \bigr].
$$

Esta ecuación es teórica e involucra una esperanza sobre todas las posibles transiciones y recompensas. Pero estamos en situaciones prácticas en las que el agente no conoce la distribución de probabilidad del entorno, por lo que **no puede calcular esa esperanza de forma exacta**.

La idea clave del aprendizaje por diferencias temporales es **aproximar la esperanza por una sola muestra observada** de la transición real. Es decir, en lugar de promediar sobre todos los posibles resultados, utilizamos el resultado concreto que acaba de ocurrir: $r_{t+1}$ en $s_{t+1}$. Esta aproximación puntual se conoce como **muestreo estocástico**.

Si simplemente igualáramos $v_\pi(s_t)$ a la muestra $r_{t+1} + \gamma v_\pi(s_{t+1})$, estaríamos reemplazando la estimación anterior por el valor observado, lo que sería muy ruidoso y provocaría grandes oscilaciones. Para suavizar el aprendizaje, se introduce un **promedio ponderado** entre la estimación actual y la nueva muestra. Esta es una técnica estándar en algoritmos iterativos: la nueva estimación se mueve una pequeña fracción hacia la muestra observada.

Denotando por $V(s_t)$ la estimación actual de $v_\pi(s_t)$, y por $V(s_{t+1})$ la estimación actual del siguiente estado, la **muestra objetivo** es:

$$
\text{objetivo} = r_{t+1} + \gamma V(s_{t+1}).
$$

La actualización se realiza entonces como:

$$
V(s_t) \leftarrow V(s_t) + \alpha \bigl[ \text{objetivo} - V(s_t) \bigr],
$$

donde $\alpha \in (0,1]$ es un **parámetro de tasa de aprendizaje** que determina cuánto nos movemos hacia la nueva muestra. Si $\alpha = 1$, la estimación se reemplaza completamente por el objetivo; si $\alpha$ es pequeño, el cambio es gradual y se promedian muchas muestras. Esta estructura es idéntica a la que ya vimos en el contexto de los promedios incrementales (con $\alpha = 1/n$), pero aquí $\alpha$ puede ser constante para permitir adaptación en entornos no estacionarios.

El término entre corchetes,

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t),
$$

Este error representa la diferencia entre lo que el agente esperaba (su valor actual de $s_t$) y la recompensa más la estimación del estado siguiente. Si el error es positivo, indica que el estado $s_t$ resultó ser más valioso de lo previsto; si es negativo, lo contrario. El agente ajusta entonces su estimación de $V(s_t)$ en esa dirección, pero sin reemplazarla completamente, sino mezclándola con la nueva información.

De este modo, la regla de actualización de TD(0) quedaría como

$$
V(s_t) \leftarrow V(s_t) + \alpha \bigl[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \bigr],
$$

y representa una **aproximación estocástica** de la ecuación de Bellman, combinando el muestreo de una única transición con un suavizado mediante tasa de aprendizaje. Esta actualización ocurre **tras cada paso individual**, lo que convierte a TD(0) en un algoritmo **online** e **incremental**. En entornos estocásticos, el proceso se repite a lo largo de múltiples episodios o trayectorias, y con una elección adecuada de $\alpha$, las estimaciones de valor convergen a los verdaderos valores esperados bajo la política $\pi$.

Para reflexionar…

> **¿Por qué no podemos simplemente igualar $V(s_t)$ a $r_{t+1} + \gamma V(s_{t+1})$ y prescindir de $\alpha$?**  
>
> **Clave**: Porque $r_{t+1} + \gamma V(s_{t+1})$ es una muestra ruidosa; usarla directamente provocaría cambios bruscos y falta de convergencia. $\alpha$ permite promediar implícitamente muchas muestras.

> **¿En qué se diferencia esta actualización de la de Monte Carlo, que usa el retorno completo $G_t$ en lugar de $r_{t+1} + \gamma V(s_{t+1})$?** 
>
> **Clave**: Monte Carlo espera a que termine el episodio y usa el retorno real $G_t$ (sin bootstrapping). TD(0) usa una estimación del futuro, $V(s_{t+1})$, lo que permite aprender antes pero introduce un posible sesgo si las estimaciones no son precisas.



Desde el punto de vista práctico, TD(0) es más eficiente que Monte Carlo, ya que puede comenzar a aprender desde la primera transición observada. Además, permite el aprendizaje continuo en tareas no episódicas o de duración indefinida.

Como ya se avanzó en la introducción, este proceso de ajuste se denomina **bootstrapping**. El agente actualiza su conocimiento basándose no en datos reales del futuro, sino en su propia estimación del valor del siguiente estado. En lugar de esperar a conocer el retorno completo, como ocurre en Monte Carlo, el agente utiliza su predicción actual como punto de apoyo para corregirse progresivamente con cada nueva transición. Esta idea de autoajuste permite que el aprendizaje sea mucho más eficiente y continuo.

TD(0) es, por tanto, un algoritmo de **predicción**. Su propósito **no es aprender una política óptima,** sino cuantificar con precisión el retorno esperado bajo una política fija. Esta evaluación resulta especialmente útil cuando se desea analizar o comparar distintas estrategias antes de tomar decisiones, o cuando se quiere utilizar la evaluación como paso intermedio en métodos de control.

A diferencia de la programación dinámica, TD(0) no requiere conocer la función de transición del entorno ni la función de recompensa exacta. Tampoco necesita esperar al final del episodio, como Monte Carlo. Esto lo convierte en una herramienta versátil para tareas donde el entorno es desconocido o continuo, y donde las actualizaciones online e incrementales son necesarias.

#### Ejemplo comparativo: TD(0) con política determinista y estocástica

En este ejemplo aplicamos el algoritmo **TD(0)** para la **predicción** (estimación de la función de valor $v_\pi(s)$) bajo una política y estocástica. A diferencia de los métodos de Monte Carlo, TD(0) actualiza las estimaciones **tras cada transición**, sin necesidad de esperar a que termine el episodio. Esto permite un aprendizaje más gradual y en tiempo real.

Recordemos el entorno del agente saltarín en su versión estocástica. 

![image-20260412104231969](./assets/image-20260412104231969.png)

Los estados relevantes son:

- $0$: estado inicial (no terminal)
- $1$: agujero (terminal, recompensa $-10$)
- $2$: casilla segura (no terminal)
- $3$: meta (terminal, recompensa neta $+19$, pues ya se ha descontado el coste de paso)
- Además, desde el estado $2$ con la acción **saltar** se puede ir a un estado terminal 4 con recompensa $-5$.


Las transiciones estocásticas (conocidas solo por el entorno, no por el agente) son las mismas que se utilizaron en la iteración de valores en los algoritmos anteriores. El agente no dispone de estas probabilidades; solo observa las consecuencias de sus acciones. En cualquier caso la siguiente tabla recoge dichas transiciones es la siguiente:

Desde la casilla**0**:

| Acción  | Prob | s'   | r    | Terminal |
| ------- | ---- | ---- | ---- | -------- |
| avanzar | 0.8  | 1    | -10  | Sí       |
| avanzar | 0.2  | 0    | -1   | No       |
| saltar  | 0.9  | 2    | -1   | No       |
| saltar  | 0.1  | 1    | -10  | Sí       |

Desde la casilla **2**:

| Acción  | Prob | s'   | r    | Terminal |
| ------- | ---- | ---- | ---- | -------- |
| avanzar | 0.9  | 3    | +19  | Sí       |
| avanzar | 0.1  | 2    | -1   | No       |
| saltar  | 0.8  | 4    | -5   | Sí       |
| saltar  | 0.2  | 2    | -1   | No       |

Desde el resto de casillas (1, 3 y 4) no tenía sentido modelar ninguna transición ya que eran estados terminales

Recordemos también el grafo que correponde a este problema. Sería este:

![image-20260412101744254](./assets/image-20260412101744254.png)



Ahora se trata de definir una **política estocástica fija** $\pi$ que el agente seguirá durante todo el aprendizaje. No se trata de una política óptima, sino de una estrategia de comportamiento que nos permitirá estudiar cómo TD(0) estima los valores de los estados. La política en nuestro caso sería:

- En estado $0$:
  - $\pi(\text{saltar} \mid 0) = 0.6$
  - $\pi(\text{avanzar} \mid 0) = 0.4$
- En estado $2$:
  - $\pi(\text{avanzar} \mid 2) = 0.7$
  - $\pi(\text{saltar} \mid 2) = 0.3$


Los estados terminales ($1$, $3$ y "fuera") tienen valor $0$ por definición y no se actualizan.

Los hiperparámetros del algoritmo serían: Un factor de descuento $\gamma = 0.9$ y una tasa de aprendizaje $\alpha = 0.1$ (constante). Inicializamos las estimaciones de los valores de los estados no terminales a cero: $V(0)=0$, $V(2)=0$.

A continuación, simulamos varios episodios generados según la política $\pi$. En cada paso de cada episodio, aplicamos la regla de actualización de TD(0):

$$
V(s_t) \leftarrow V(s_t) + \alpha \bigl[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \bigr].
$$

**Episodio 1**

Supongamos que la aleatoriedad de la política y del entorno produce la siguiente trayectoria:

1. Desde el estado $0$, la política elige **saltar** (probabilidad $0.6$). El entorno, con probabilidad $0.9$, lleva al estado $2$ con recompensa $-1$; con probabilidad $0.1$ llevaría al agujero ($1$) con $-10$. En este caso, el resultado es favorable: transición a $2$ con $r = -1$.
2. Desde el estado $2$, la política elige **avanzar** (probabilidad $0.7$). El entorno, con probabilidad $0.9$, lleva a la meta ($3$) con recompensa $+19$; con probabilidad $0.1$ se quedaría en $2$ con $-1$. Aquí el resultado es llegar a la meta: transición a $3$ con $r = +19$. El episodio termina.

  La secuencia de transiciones es:

  - $t=0$: $s_0=0$, acción saltar → $s_1=2$, $r=-1$

  - $t=1$: $s_1=2$, acción avanzar → $s_2=3$, $r=+19$


Valores iniciales: $V(0)=0$, $V(2)=0$, $V(3)=0$.

**Actualizaciones paso a paso** (aplicamos la regla inmediatamente después de cada transición):

- **Paso 1** (transición de $0$ a $2$):

  $$
  \delta = r + \gamma V(2) - V(0) = -1 + 0.9 \cdot 0 - 0 = -1
  $$
  
  $$
  V(0) \leftarrow 0 + 0.1 \cdot (-1) = -0.1
  $$

- **Paso 2** (transición de $2$ a $3$):

  $$
  \delta = r + \gamma V(3) - V(2) = 19 + 0.9 \cdot 0 - 0 = 19
  $$
  
  $$
  V(2) \leftarrow 0 + 0.1 \cdot 19 = 1.9
  $$

  Al final del episodio: $V(0) = -0.1$, $V(2) = 1.9$. Observamos que $V(0)$ ha bajado ligeramente (debido a la recompensa negativa del primer paso) y $V(2)$ ha subido notablemente al recibir la recompensa positiva de la meta.

**Episodio 2**

Continuamos con la misma política. Supongamos ahora una trayectoria desafortunada:

1. Desde el estado $0$, la política elige **avanzar** (probabilidad $0.4$). El entorno, con probabilidad $0.8$, lleva al agujero ($1$) con recompensa $-10$; con probabilidad $0.2$ se quedaría en $0$ con $-1$. En este caso, el resultado es caer en el agujero: transición a $1$ con $r = -10$. El episodio termina inmediatamente (solo un paso).

Valores actuales antes del episodio: $V(0) = -0.1$, $V(2)=1.9$, $V(1)=0$.

**Actualización** (único paso):
- Transición de $0$ a $1$:

  $$
  \delta = r + \gamma V(1) - V(0) = -10 + 0.9 \cdot 0 - (-0.1) = -10 + 0.1 = -9.9
  $$
  
  $$
  V(0) \leftarrow -0.1 + 0.1 \cdot (-9.9) = -0.1 - 0.99 = -1.09
  $$

  Tras el episodio: $V(0) = -1.09$, $V(2)$ se mantiene en $1.9$. La mala experiencia ha hecho que la estimación de $V(0)$ baje considerablemente.

**Episodio 3**

Generamos otra trayectoria, ahora con una combinación mixta:

1. Desde $0$, la política elige **saltar** (probabilidad $0.6$). El entorno, esta vez, lleva a $2$ con recompensa $-1$ (caso favorable).
2. Desde $2$, la política elige **saltar** (probabilidad $0.3$). El entorno, con probabilidad $0.8$, saca al agente del tablero (estado "fuera") con recompensa $-5$; con probabilidad $0.2$ se quedaría en $2$ con $-1$. Supongamos que ocurre lo primero: transición a "fuera" con $r = -5$. El episodio termina.

Valores antes del episodio: $V(0) = -1.09$, $V(2) = 1.9$, $V(\text{fuera})=0$.

**Actualizaciones**:

- **Paso 1** (transición de $0$ a $2$):

  $$
  \delta = -1 + 0.9 \cdot V(2) - V(0) = -1 + 0.9 \cdot 1.9 - (-1.09) = -1 + 1.71 + 1.09 = 1.80
  $$
  
  $$
  V(0) \leftarrow -1.09 + 0.1 \cdot 1.80 = -1.09 + 0.18 = -0.91
  $$

- **Paso 2** (transición de $2$ a "fuera"):

  $$
  \delta = -5 + 0.9 \cdot 0 - V(2) = -5 - 1.9 = -6.9
  $$
  
  $$
  V(2) \leftarrow 1.9 + 0.1 \cdot (-6.9) = 1.9 - 0.69 = 1.21
  $$

  Resultados tras el episodio 3: $V(0) \approx -0.91$, $V(2) \approx 1.21$.



En los tres episodios observamos fluctuaciones significativas en las estimaciones:

- $V(0)$ comenzó en $0$, bajó a $-0.1$, luego a $-1.09$, y finalmente subió a $-0.91$.
- $V(2)$ pasó de $0$ a $1.9$, y luego descendió a $1.21$.


Estas oscilaciones son propias de la alta varianza inherente a las trayectorias generadas por una política estocástica. Algunas veces se alcanza la meta (episodio 1), otras se cae en el agujero (episodio 2) o se sale del tablero (episodio 3). Con un número elevado de episodios, las estimaciones convergerían a los valores verdaderos bajo la política $\pi$, que se obtendrían resolviendo las ecuaciones de Bellman. Esos valores serían inferiores a los de la política óptima, porque esta política estocástica no siempre elige las mejores acciones.

La principal ventaja de TD(0) frente a Monte Carlo es que **no necesita esperar a que termine el episodio** para actualizar. En este ejemplo los episodios son muy cortos y la diferencia no se aprecia, pero en entornos con episodios largos o continuos, TD(0) puede aprender en tiempo real, actualizando tras cada transición. Además, el **bootstrapping** (usar la estimación actual $V(s_{t+1})$ como parte del objetivo) permite que la información se propague hacia atrás incluso antes de conocer el retorno final.

> Si hubiésemos aplicado Monte Carlo (first‑visit) a los mismos episodios, en el episodio 1 habríamos calculado el retorno completo desde el primer estado: $G_0 = -1 + 0.9 \cdot 19 = 16.1$, y habríamos actualizado $V(0)$ como promedio (con $\alpha = 1/n$ o con una constante). En TD(0), en cambio, actualizamos después de cada paso, lo que puede acelerar el aprendizaje, aunque introduce un posible sesgo si las estimaciones $V(s_{t+1})$ no son precisas al principio.
>

Este caso práctico muestra cómo TD(0) aprende a estimar los valores de los estados a partir de la experiencia directa, sin necesidad de conocer el modelo del entorno. La política estocástica introduce variabilidad en las trayectorias, y el algoritmo va ajustando las estimaciones paso a paso. Tras muchos episodios, los valores convergerían a los esperados bajo dicha política. El alumno puede experimentar variando $\alpha$ y $\gamma$, o cambiando la política, para observar cómo afectan a la velocidad de convergencia y a la estabilidad de las estimaciones.

**Para reflexionar…**

> **¿Qué ventaja tiene TD(0) frente a Monte Carlo en entornos con episodios muy largos o continuos?** 
>
> **Clave**: TD(0) no necesita esperar a que termine el episodio; puede actualizar tras cada transición, lo que permite aprender en tiempo real y adaptarse a cambios en la política o en el entorno.

> **En el episodio 2, ¿por qué $V(0)$ se vuelve más negativo a pesar de que la política a veces lleva a la meta?** 
>
> **Clave**: Porque la política tiene una probabilidad significativa de elegir avanzar (0.4), y al hacerlo, el entorno con alta probabilidad (0.8) lleva al agujero, produciendo recompensas muy negativas. El valor de $V(0)$ refleja el promedio ponderado de todas las consecuencias posibles, no solo las favorables.

### Métodos TD para control

#### Introducción

En los apartados anteriores hemos visto cómo es posible utilizar métodos de tipo TD para estimar el valor de una política fija. Sin embargo, el verdadero objetivo en la mayoría de problemas de aprendizaje por refuerzo no es simplemente evaluar políticas, sino aprender una que sea óptima, es decir, que maximice el retorno esperado del agente en cada estado.

Para abordar este objetivo, los algoritmos de control basados en diferencias temporales introducen un componente esencial: la mejora de la política a lo largo del tiempo. Esto se logra manteniendo una estimación de la utilidad de cada acción en cada estado y actualizándola progresivamente con la experiencia.

A diferencia de los métodos de predicción, que solo estiman el valor de los estados, **los métodos de control trabajan directamente con la función acción-valor**. Este cambio permite tomar decisiones sobre qué acción ejecutar en un estado dado, sin depender de un modelo del entorno ni de cálculos de planificación.

El proceso general en estos métodos es el siguiente: el agente observa una transición en la interacción con el entorno, que consiste en un estado actual, una acción tomada, una recompensa obtenida y un nuevo estado alcanzado. Con esta información, el agente actualiza su estimación del valor de la acción que ha ejecutado. Posteriormente, la política puede modificarse favoreciendo aquellas acciones que se han mostrado más prometedoras.

Existen distintas variantes de este esquema, dependiendo de cómo se utilicen las acciones futuras en el proceso de actualización. Dos de los métodos más representativos son SARSA y Q-learning. Ambos permiten al agente aprender políticas cada vez mejores a partir de su experiencia directa, pero difieren en la manera en que se enfrentan al dilema exploración-explotación y en el tipo de política que efectivamente aprenden.

En el caso de SARSA, el aprendizaje se ajusta a la política que el agente está siguiendo realmente. Esto significa que si el agente incluye exploración en su comportamiento, como en las estrategias $\epsilon$-greedy, el conocimiento aprendido reflejará también esa misma estrategia. Este enfoque se conoce como aprendizaje on-policy, y tiende a ser más conservador y estable en entornos ruidosos o donde las decisiones arriesgadas pueden tener consecuencias negativas.

Por el contrario, Q-learning busca aprender directamente el valor de la política óptima, independientemente de cómo el agente actúe durante la fase de aprendizaje. En este caso, el agente puede explorar libremente, pero siempre estima el valor suponiendo que en el futuro actuará de forma óptima. Este enfoque se conoce como aprendizaje off-policy, y tiene la ventaja de converger más rápidamente hacia políticas de alto rendimiento en muchos contextos, aunque puede ser más sensible a una exploración excesiva o mal controlada.

Ambos métodos ilustran el principio fundamental del control mediante aprendizaje por refuerzo: la mejora continua de la política a partir de la experiencia directa, sin necesidad de conocer el modelo del entorno. A partir de la próxima sección, estudiaremos en detalle estas dos aproximaciones, sus fundamentos matemáticos y su comportamiento en distintos escenarios.

#### De nuevo el balance Explotación-Exploración

Ya hemos visto en secciones anteriores como uno de los retos fundamentales en los algoritmos de control del aprendizaje por refuerzo es decidir **qué acción debe ejecutarse en cada momento**, no solo para obtener buenas recompensas inmediatas, sino para **aprender la mejor política posible a largo plazo**. Esta situación nos obliga a retomar una idea clave que ya hemos discutido anteriormente: el equilibrio entre **explotación** y **exploración**.

Recordemos: La **explotación** consiste en seleccionar las acciones que actualmente se consideran mejores, es decir, aquellas para las que el agente ha estimado un alto valor esperado. Esta estrategia favorece decisiones seguras y rentables a corto plazo. Sin embargo, si el agente se limita a explotar lo que ya conoce, corre el riesgo de no descubrir alternativas potencialmente mejores que aún no han sido exploradas. La **exploración**, por el contrario, implica elegir acciones que pueden parecer subóptimas según el conocimiento actual, pero que ofrecen la posibilidad de obtener nueva información. Es precisamente esta información la que permite refinar las estimaciones de valor y, a la larga, encontrar políticas más efectivas.

En los métodos de predicción esta tensión ya estaba presente, pero su efecto era limitado: el objetivo era estimar los valores bajo una política dada, y por tanto el impacto de las acciones elegidas se acotaba a la calidad de esa estimación. Sin embargo, **en los algoritmos de control el agente no solo evalúa, sino que también actúa para mejorar**, y por tanto el dilema exploración-explotación adquiere una dimensión decisiva.

La necesidad de explorar se traduce en la práctica en el uso de políticas **estocásticas** o **deterministas con exploración**, como por ejemplo las políticas $\epsilon$-greedy, donde el agente elige con probabilidad $1 - \epsilon$ la acción con mayor valor estimado, y con probabilidad $\epsilon$ una acción aleatoria. Este mecanismo permite controlar la exploración de forma gradual: un valor alto de $\epsilon$ favorece la recolección de información en las primeras fases del entrenamiento, mientras que valores decrecientes permiten consolidar el conocimiento adquirido y optimizar la política en fases posteriores.

En los algoritmos que estudiaremos a continuación, como **SARSA** y **Q-learning**, este balance será gestionado de forma distinta. En SARSA, el agente aprende sobre la política que ejecuta realmente (incluida su exploración), mientras que en Q-learning se estima el valor de la política óptima incluso cuando se comporta de otro modo.

Por esto es por lo que el dilema entre exploración y explotación debe mantenerse presente como **marco conceptual esencial** durante todo el estudio de los métodos de control. No es solo una cuestión de estrategia de comportamiento, sino una dimensión crítica que condiciona cómo y qué aprende el agente en cada interacción.

#### Control on-policy: el algoritmo SARSA

Hemos dicho que el aprendizaje por refuerzo no solo permite evaluar políticas, sino también aprender políticas óptimas directamente desde la experiencia. Para lograrlo, los algoritmos de control permiten actualizar estimaciones de valor y mejorar decisiones de forma iterativa. En este contexto, el algoritmo SARSA representa una estrategia de aprendizaje *on-policy*, es decir, que aprende sobre la política que el propio agente está ejecutando.

La motivación detrás de SARSA parte de una idea fundamental: el agente debe adaptar su comportamiento no solo en función de los resultados que observa, sino también teniendo en cuenta la política real que sigue, incluida la exploración. Esto implica que la política no se asume óptima durante el aprendizaje, sino que evoluciona gradualmente conforme mejora la estimación de los valores de las acciones.

SARSA toma su nombre de los cinco elementos que intervienen en cada transición observada por el agente: *State–Action–Reward–State–Action*, es decir, $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$. A partir de esta secuencia, el algoritmo actualiza el valor de $Q(s_t, a_t)$, es decir, la utilidad estimada de ejecutar la acción $a_t$ en el estado $s_t$.

Este valor se ajusta con base en el retorno observado al seguir la política vigente, lo que incluye la posibilidad de que el agente explore en el siguiente estado. Esta es la principal diferencia respecto a enfoques off-policy: en SARSA, el aprendizaje refleja exactamente el comportamiento real del agente, con todas las implicaciones que conlleva la exploración.

Desde un punto de vista formal, SARSA se basa en la ecuación de Bellman para políticas fijas, aplicada a funciones acción-valor:

$$
q^\pi(s_t, a_t) = \mathbb{E} \left[ r_{t+1} + \gamma q^\pi(s_{t+1}, a_{t+1}) \mid s_t, a_t \right]
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

##### **Un ejemplo práctico de uso del algoritmo SARSA**  
Vamos a aplicar **SARSA** (State‑Action‑Reward‑State‑Action) para el **control on‑policy** del agente saltarín. A diferencia de los métodos de predicción (como TD(0) aplicado a una política fija), SARSA aprende directamente la función acción‑valor $Q(s,a)$ mientras interactúa con el entorno siguiendo una política $\epsilon$-greedy que se va mejorando. 

Recordemos el entorno estocástico del agente saltarín:

- **Estados**: $0$ (inicio), $1$ (agujero, terminal), $2$ (seguro), $3$ (meta, terminal), y un estado terminal adicional 4 (fuera) al saltar desde $2$.
- **Acciones**: $0$ = avanzar (+1 casilla), $1$ = saltar (+2 casillas).
- **Recompensas** (ya incluyen el coste de paso de $-1$):
  - Transición $0 \xrightarrow{\text{avanzar}} 1$ (agujero): $-10$ (terminal)
  - Transición $0 \xrightarrow{\text{avanzar}} 0$ (permanecer, con prob. 0.2): $-1$ (no terminal)
  - Transición $0 \xrightarrow{\text{saltar}} 2$: $-1$ (no terminal)
  - Transición $0 \xrightarrow{\text{saltar}} 1$ (agujero, con prob. 0.1): $-10$ (terminal)
  - Transición $2 \xrightarrow{\text{avanzar}} 3$ (meta): $+19$ (terminal)
  - Transición $2 \xrightarrow{\text{avanzar}} 2$ (permanecer, con prob. 0.1): $-1$ (no terminal)
  - Transición $2 \xrightarrow{\text{saltar}} \text{fuera}$: $-5$ (terminal)
  - Transición $2 \xrightarrow{\text{saltar}} 2$ (permanecer, con prob. 0.2): $-1$ (no terminal)

- **Factor de descuento**: $\gamma = 0.9$
- **Tasa de aprendizaje**: $\alpha = 0.1$ (constante)
- **Parámetro de exploración**: $\epsilon = 0.2$ (política $\epsilon$-greedy)


Inicializamos $Q(s,a)=0$ para todos los pares estado‑acción. Los estados terminales tienen valor $0$ por definición y no se actualizan.

La regla de actualización de SARSA es

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \bigr],
$$

donde $a_{t+1}$ es la acción que realmente se tomará en el siguiente estado según la política $\epsilon$-greedy actual.

A continuación, simulamos varios episodios.

**Episodio 1**

**Inicialización**: $Q(s,a)=0$ para todo $(s,a)$. La política $\epsilon$-greedy, al tener todos los valores iguales, elige acciones al azar con probabilidad $1/2$ (en cada estado, dos acciones). Supongamos que la aleatoriedad produce la siguiente trayectoria:

- $t=0$: estado $0$, se elige acción **saltar** (por azar). 
  
  El entorno, con probabilidad $0.9$, lleva a estado $2$ con recompensa $-1$ (asumimos que ocurre así). No es terminal.
  
- $t=1$: estado $2$, se elige acción **avanzar** (por azar). 
  
  El entorno, con probabilidad $0.9$, lleva a estado $3$ (meta) con recompensa $+19$. El episodio termina. La acción siguiente no existe (o se define $Q(3,\cdot)=0$).

Registramos la secuencia de transiciones:

- $(0, \text{saltar}) \to (2, -1)$
- $(2, \text{avanzar}) \to (3, +19)$


**Actualizaciones** (recorremos hacia atrás o en orden, pero SARSA actualiza en cada paso inmediatamente después de conocer $a_{t+1}$):

- **Paso 2** (última transición): desde $t=1$, $s_1=2$, $a_1=\text{avanzar}$, $r=19$, $s_2=3$, $a_2$ no existe (estado terminal). En la práctica, para estados terminales se define $Q(s_{\text{terminal}}, \cdot)=0$, y la actualización se realiza sin el término $Q(s_{t+1},a_{t+1})$ (o se toma como $0$). La fórmula queda:

  $$
  Q(2,\text{avanzar}) \leftarrow 0 + 0.1 \bigl[ 19 + 0.9 \cdot 0 - 0 \bigr] = 1.9
  $$

- **Paso 1** (primera transición): $s_0=0$, $a_0=\text{saltar}$, $r=-1$, $s_1=2$, $a_1=\text{avanzar}$ (ya conocido). Aplicamos:
- 
  $$
  \delta = r + \gamma Q(2,\text{avanzar}) - Q(0,\text{saltar}) = -1 + 0.9 \cdot 1.9 - 0 = -1 + 1.71 = 0.71
  $$
  
  $$
  Q(0,\text{saltar}) \leftarrow 0 + 0.1 \cdot 0.71 = 0.071
  $$


Tras el episodio 1, la tabla $Q$ es:

| (s,a)        | Valor |
| ------------ | ----- |
| (0, avanzar) | 0     |
| (0, saltar)  | 0.071 |
| (2, avanzar) | 1.9   |
| (2, saltar)  | 0     |

**Episodio 2**

Ahora la política $\epsilon$-greedy se basa en estos valores. En el estado $0$, $Q(0,\text{saltar})(=0.071) > Q(0,\text{avanzar})(=0)$, por lo que la acción greedy es saltar. Con probabilidad $1-\epsilon=0.8$ se elige saltar; con probabilidad $\epsilon=0.2$ se elige una acción aleatoria (avanzar o saltar con igual probabilidad). Supongamos que en este episodio la política elige **saltar** (explotación).

- $t=0$: estado $0$, acción saltar. El entorno, esta vez, con probabilidad $0.1$ lleva al agujero (estado $1$) con recompensa $-10$ (asumimos que ocurre así). El episodio termina inmediatamente. Solo hay una transición.


**Actualización**:

- $s_0=0$, $a_0=\text{saltar}$, $r=-10$, $s_1=1$ (terminal). No hay $a_1$.

  $$
  Q(0,\text{saltar}) \leftarrow 0.071 + 0.1 \bigl[ -10 + 0.9 \cdot 0 - 0.071 \bigr] = 0.071 + 0.1 \cdot (-10.071) = 0.071 - 1.0071 = -0.9361
  $$


Tabla tras episodio 2:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | 0      |
| (0, saltar)  | -0.936 |
| (2, avanzar) | 1.9    |
| (2, saltar)  | 0      |

**Episodio 3**

Ahora en estado $0$, la acción greedy es avanzar (porque $Q(0,\text{avanzar})=0 > -0.936$). Con probabilidad $0.8$ se elige avanzar. Supongamos que se elige **avanzar**.

- $t=0$: estado $0$, acción avanzar. El entorno, con probabilidad $0.8$, lleva a agujero ($1$) con recompensa $-10$ (asumimos que ocurre). Episodio termina.

  **Actualización**:

- $Q(0,\text{avanzar})$:

  $$
  Q(0,\text{avanzar}) \leftarrow 0 + 0.1 \bigl[ -10 + 0 - 0 \bigr] = -1
  $$


Tabla tras episodio 3:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | -1     |
| (0, saltar)  | -0.936 |
| (2, avanzar) | 1.9    |
| (2, saltar)  | 0      |

En estado $0$, la mejor acción es ahora saltar ( $-1 < -0.936$ ), aunque ambas son negativas. En estado $2$, la mejor acción es avanzar con $1.9$.

**Episodio 4**

En estado $0$, la acción greedy es saltar ($-0,936 > -1$). Con probabilidad $1-\epsilon = 0,8$ la política elegirá saltar; con probabilidad $\epsilon = 0,2$ elegirá una acción aleatoria (avanzar o saltar con igual probabilidad). Supongamos que en este episodio se elige **saltar** (explotación).

**Desarrollo del episodio**:

- $t=0$: estado $0$, acción saltar. El entorno tiene dos posibles resultados:
  - Con probabilidad $0,9$: transición a estado $2$ con recompensa $-1$ (no terminal).
  - Con probabilidad $0,1$: transición a agujero ($1$) con recompensa $-10$ (terminal).
  Asumimos que ocurre el resultado favorable: se transita a $2$ con $r = -1$.

- $t=1$: estado $2$. La política $\epsilon$-greedy compara $Q(2,\text{avanzar})=1,9$ y $Q(2,\text{saltar})=0$, por lo que la acción greedy es avanzar. Con probabilidad $0,8$ se elige avanzar (supongamos que así ocurre). El entorno, desde $2$ con acción avanzar:
  - Con probabilidad $0,9$: transición a meta ($3$) con recompensa $+19$ (terminal).
  - Con probabilidad $0,1$: permanece en $2$ con recompensa $-1$ (no terminal).
  Asumimos que ocurre lo primero: se alcanza la meta con $r = +19$. El episodio termina.

  La secuencia de transiciones es:

1. $(0, \text{saltar}) \to (2, -1)$
2. $(2, \text{avanzar}) \to (3, +19)$


**Actualizaciones SARSA** (paso a paso, usando los valores previos al episodio):

- **Paso 2** (última transición): $s=2$, $a=\text{avanzar}$, $r=19$, $s'=3$ (terminal). No hay $a'$ (se toma $Q(3,\cdot)=0$).

  $$
  Q(2,\text{avanzar}) \leftarrow 1,9 + 0,1 \bigl[ 19 + 0,9 \cdot 0 - 1,9 \bigr] = 1,9 + 0,1 \cdot (19 - 1,9) = 1,9 + 0,1 \cdot 17,1 = 1,9 + 1,71 = 3,61
  $$

- **Paso 1** (primera transición): $s=0$, $a=\text{saltar}$, $r=-1$, $s'=2$, $a'=\text{avanzar}$ (la acción que se eligió en el siguiente estado).

$$
  \delta = r + \gamma Q(2,\text{avanzar}) - Q(0,\text{saltar}) = -1 + 0,9 \cdot 3,61 - (-0,936)
$$

  Calculamos: $0,9 \cdot 3,61 = 3,249$; luego $\delta = -1 + 3,249 + 0,936 = 3,185$.

$$
  Q(0,\text{saltar}) \leftarrow -0,936 + 0,1 \cdot 3,185 = -0,936 + 0,3185 = -0,6175
$$


**Tabla final tras el episodio 4**:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | -1     |
| (0, saltar)  | -0.618 |
| (2, avanzar) | 3.61   |
| (2, saltar)  | 0      |

La acción greedy en estado $0$ sigue siendo saltar ($-0,618 > -1$), y su valor ha mejorado notablemente (de $-0,936$ a $-0,618$) gracias a la recompensa obtenida al final del episodio. En estado $2$, avanzar sigue siendo la mejor acción con un valor de $3,61$.



Los cuatro episodios simulados muestran la evolución típica de SARSA en un entorno estocástico con política $\epsilon$-greedy. Partiendo de valores nulos, el agente explora inicialmente y actualiza sus estimaciones. Tras los primeros episodios, ya identifica que saltar desde $0$ es preferible a avanzar, aunque ambas acciones tengan valores negativos. La experiencia positiva (alcanzar la meta) refuerza las acciones que llevan a ella, mientras que las caídas en el agujero las penalizan. Con suficientes episodios, los valores convergerían a los que se obtendrían resolviendo las ecuaciones de Bellman para la política óptima.

**Para reflexionar…**

> **¿Por qué en el episodio 4, a pesar de que la acción saltar llevó a una trayectoria exitosa, su valor sigue siendo negativo ($-0,618$)?**  
>
> **Clave**: Porque la acción saltar tiene una probabilidad del 10% de caer directamente en el agujero ($-10$). El valor estimado es el promedio ponderado de todas las posibles consecuencias, no solo de las favorables. Aunque en este episodio concreto fue buena, la expectativa a largo plazo aún es ligeramente negativa, pero mucho mejor que la de avanzar ($-1$). Con más episodios exitosos, el valor podría volverse positivo.

> **¿Qué habría ocurrido si en el episodio 4, después de permanecer en $0$, la política hubiera elegido saltar en lugar de avanzar? ¿Cómo afectaría eso a la actualización?** 
>
> **Clave**: El valor de $Q(0,\text{avanzar})$ se habría actualizado usando $Q(0,\text{saltar})$ como estimación del siguiente estado, lo que podría haber cambiado la dirección de la actualización. Esto muestra la importancia de la acción real $a_{t+1}$ en SARSA.

> **¿Por qué SARSA es considerado un algoritmo on‑policy mientras que Q‑learning es off‑policy?**
>
> **Clave**: SARSA utiliza la acción $a_{t+1}$ que realmente se va a ejecutar según la política actual. Q‑learning, en cambio, usa $\max_{a'} Q(s_{t+1}, a')$, independientemente de la acción que luego tome el agente. Eso permite a Q‑learning aprender la política óptima incluso mientras se comporta de forma exploratoria.

#### Control off-policy: el algoritmo Q-learning

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

##### Relevancia de Q-learning en el aprendizaje por refuerzo

Q-learning es, históricamente, uno de los algoritmos más influyentes en el desarrollo del aprendizaje por refuerzo moderno. Su capacidad para **aprender políticas óptimas directamente desde la interacción**, sin necesidad de conocer el modelo del entorno, y sin ajustarse estrictamente al comportamiento real del agente, lo ha convertido en el punto de partida de muchas extensiones y desarrollos posteriores.

Entre los motivos de su importancia destacan:

- Su aplicabilidad en entornos complejos, estocásticos y parcialmente observables.
- Su papel como base conceptual de métodos más avanzados, como **Deep Q-Networks (DQN)**.
- Su convergencia probada bajo condiciones razonables y su robustez ante exploración $\epsilon$-greedy.


En resumen, Q-learning proporciona al agente un mecanismo para aprender a actuar **como si ya supiera lo que es óptimo**, permitiéndole mejorar continuamente su política sin depender directamente de su comportamiento actual. Esta característica lo convierte en una herramienta central en cualquier curso avanzado de aprendizaje por refuerzo.

##### **Ejemplo práctico de Q-learning**  
Seguimos trabajando con el problema del agente saltarín. Ahora la idea es aplicar el algoritmo **Q‑learning** para el **control off‑policy** del agente. A diferencia de SARSA, Q‑learning actualiza los valores $Q(s,a)$ utilizando la **mejor acción posible** en el siguiente estado, independientemente de la acción que realmente se vaya a tomar. Esto permite aprender la política óptima mientras se sigue una política de comportamiento exploratoria (por ejemplo, $\epsilon$-greedy). La regla de actualización es entonces:
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right],
$$

donde $\max_{a'} Q(s_{t+1}, a')$ es el valor de la mejor acción en el siguiente estado, independientemente de la acción que luego se elija para actuar.

Recordemos el entorno estocástico del agente saltarín:

- **Estados**: $0$ (inicio), $1$ (agujero, terminal), $2$ (seguro), $3$ (meta, terminal), y 4 (terminal al saltar desde $2$).
- **Acciones**: $0$ = avanzar (+1), $1$ = saltar (+2).
- **Recompensas** (ya incluyen coste de paso $-1$):
  - $0 \xrightarrow{\text{avanzar}} 1$: $-10$ (terminal)
  - $0 \xrightarrow{\text{avanzar}} 0$ (permanecer, prob. 0.2): $-1$
  - $0 \xrightarrow{\text{saltar}} 2$: $-1$ (no terminal)
  - $0 \xrightarrow{\text{saltar}} 1$ (agujero, prob. 0.1): $-10$ (terminal)
  - $2 \xrightarrow{\text{avanzar}} 3$ (meta): $+19$ (terminal)
  - $2 \xrightarrow{\text{avanzar}} 2$ (permanecer, prob. 0.1): $-1$
  - $2 \xrightarrow{\text{saltar}} \text{fuera}$: $-5$ (terminal)
  - $2 \xrightarrow{\text{saltar}} 2$ (permanecer, prob. 0.2): $-1$

- **Parámetros**: $\gamma = 0.9$, $\alpha = 0.1$, $\epsilon = 0.2$ (política $\epsilon$-greedy).
- **Inicialización**: $Q(s,a)=0$ para todos los pares estado‑acción.


A diferencia de SARSA, Q‑learning **no necesita conocer la acción $a_{t+1}$** para actualizar; utiliza el máximo sobre acciones en $s_{t+1}$.

A continuación, simulamos varios episodios.

**Episodio 1**

**Inicialización**: todos los $Q=0$. La política $\epsilon$-greedy elige acciones al azar (con igual probabilidad). Supongamos que se genera la siguiente trayectoria:

- $t=0$: estado $0$, se elige **saltar** (por azar).
  
  El entorno, con probabilidad $0.9$, lleva a estado $2$ con recompensa $-1$ (asumimos que ocurre así). No es terminal.
  
- $t=1$: estado $2$, se elige **avanzar** (por azar). 
  
  El entorno, con probabilidad $0.9$, lleva a estado $3$ (meta) con recompensa $+19$. Episodio termina.

**Actualizaciones Q‑learning** (se realizan después de cada transición, usando el máximo sobre acciones del siguiente estado):

- **Paso 2** (última transición): $s=2$, $a=\text{avanzar}$, $r=19$, $s'=3$ (terminal). Para estados terminales, $\max_{a'} Q(3,a') = 0$

  $$
  Q(2,\text{avanzar}) \leftarrow 0 + 0.1 \bigl[ 19 + 0.9 \cdot 0 - 0 \bigr] = 1.9
  $$

- **Paso 1** (primera transición): $s=0$, $a=\text{saltar}$, $r=-1$, $s'=2$. Calculamos $\max_{a'} Q(2,a') = \max(1.9, 0) = 1.9$ (el valor que acabamos de actualizar, aunque en la práctica se usa el valor anterior si la actualización es secuencial; aquí asumimos que primero actualizamos el paso 2 y luego el paso 1, lo cual es válido porque en Q‑learning el orden no afecta a la convergencia).

  $$
  \delta = -1 + 0.9 \cdot 1.9 - 0 = -1 + 1.71 = 0.71
  $$
  
  $$
  Q(0,\text{saltar}) \leftarrow 0 + 0.1 \cdot 0.71 = 0.071
  $$


Tabla tras episodio 1:

| (s,a)        | Valor |
| ------------ | ----- |
| (0, avanzar) | 0     |
| (0, saltar)  | 0.071 |
| (2, avanzar) | 1.9   |
| (2, saltar)  | 0     |



**Episodio 2**

Ahora la política $\epsilon$-greedy en estado $0$ prefiere saltar ($0.071 > 0$). Con probabilidad $0.8$ se elige saltar. Supongamos que se elige **saltar**, pero esta vez el entorno, con probabilidad $0.1$, lleva al agujero (estado $1$) con recompensa $-10$. Episodio termina inmediatamente.

**Actualización**:

- $s=0$, $a=\text{saltar}$, $r=-10$, $s'=1$ (terminal). $\max_{a'} Q(1,a') = 0$.
  $$
  Q(0,\text{saltar}) \leftarrow 0.071 + 0.1 \bigl[ -10 + 0 - 0.071 \bigr] = 0.071 + 0.1 \cdot (-10.071) = 0.071 - 1.0071 = -0.9361
  $$


Tabla tras episodio 2:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | 0      |
| (0, saltar)  | -0.936 |
| (2, avanzar) | 1.9    |
| (2, saltar)  | 0      |

En estado $0$, la acción greedy ahora es **avanzar** (porque $0 > -0.936$). La política ha cambiado.



**Episodio 3**

En estado $0$, la política elige avanzar con probabilidad $0.8$. Supongamos que se elige **avanzar**. El entorno, con probabilidad $0.8$, lleva a agujero ($1$) con recompensa $-10$ (asumimos que ocurre). Episodio termina.

**Actualización**:
- $s=0$, $a=\text{avanzar}$, $r=-10$, $s'=1$ terminal.

  $$
  Q(0,\text{avanzar}) \leftarrow 0 + 0.1 \bigl[ -10 + 0 - 0 \bigr] = -1
  $$


Tabla tras episodio 3:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | -1     |
| (0, saltar)  | -0.936 |
| (2, avanzar) | 1.9    |
| (2, saltar)  | 0      |

Ahora la acción greedy en estado $0$ es **saltar** ($-0.936 > -1$). La política vuelve a preferir saltar. Obsérvese que Q‑learning ha actualizado solo el valor de la acción ejecutada, pero la política puede cambiar en cada episodio.



**Episodio 4**

Supongamos una trayectoria más larga, donde la política elige saltar (greedy) y el entorno lleva a estado $2$ (favorable).

- $t=0$: $0 \xrightarrow{\text{saltar}} 2$, $r=-1$, no terminal.

- $t=1$: estado $2$. La política $\epsilon$-greedy compara $Q(2,\text{avanzar})=1.9$ y $Q(2,\text{saltar})=0$, por lo que elegirá avanzar con probabilidad $0.8$ (explotación). Supongamos que elige avanzar y el entorno lleva a la meta ($3$) con $r=+19$ (terminal).

**Actualizaciones** (primero el último paso, luego el primero, usando siempre el máximo en el siguiente estado):

- **Paso 2**: $s=2$, $a=\text{avanzar}$, $r=19$, $s'=3$ terminal.
  $$
  Q(2,\text{avanzar}) \leftarrow 1.9 + 0.1 \bigl[ 19 + 0 - 1.9 \bigr] = 1.9 + 0.1 \cdot 17.1 = 1.9 + 1.71 = 3.61
  $$

- **Paso 1**: $s=0$, $a=\text{saltar}$, $r=-1$, $s'=2$. Calculamos $\max_{a'} Q(2,a') = \max(3.61, 0) = 3.61$.

  $$
  \delta = -1 + 0.9 \cdot 3.61 - (-0.936) = -1 + 3.249 + 0.936 = 3.185
  $$
  
  $$
  Q(0,\text{saltar}) \leftarrow -0.936 + 0.1 \cdot 3.185 = -0.936 + 0.3185 = -0.6175
  $$


Tabla tras episodio 4:

| (s,a)        | Valor  |
| ------------ | ------ |
| (0, avanzar) | -1     |
| (0, saltar)  | -0.617 |
| (2, avanzar) | 3.61   |
| (2, saltar)  | 0      |

La acción greedy en estado $0$ sigue siendo saltar ($-0.618 > -1$), y su valor ha mejorado. En estado $2$, avanzar es claramente superior.

###### Comparación con SARSA

En los mismos episodios, SARSA habría actualizado $Q(2,\text{avanzar})$ usando la acción $a_{t+1}$ real (que en el episodio 4 era avanzar, igual que el máximo, por lo que coincidiría). Sin embargo, si en el episodio 4, desde el estado $2$, la política hubiera elegido **saltar** (por exploración), SARSA habría usado $Q(2,\text{saltar})$ en la actualización del paso 1, mientras que Q‑learning habría seguido usando $\max_a Q(2,a)=Q(2,\text{avanzar})$. Esta es la diferencia fundamental: Q‑learning aprende la política óptima incluso cuando se comporta de forma subóptima, mientras que SARSA aprende la política que realmente sigue (incluyendo la exploración). 

Por eso Q‑learning es más **optimista** (puede converger más rápido a la política óptima) pero también puede ser más inestable en entornos ruidosos.

**Para reflexionar…**

> **¿Qué habría ocurrido en el episodio 2 si, en lugar de caer en el agujero, el salto hubiera llevado a $2$ (resultado favorable)? ¿Cómo afectaría eso a $Q(0,\text{saltar})$?** 
>
> **Clave**: Habría actualizado $Q(0,\text{saltar})$ con un valor positivo, reforzando aún más la preferencia por saltar. El promedio ponderado con episodios futuros llevaría al valor esperado verdadero.

> **¿Por qué Q‑learning puede aprender la política óptima incluso cuando la política de comportamiento es $\epsilon$-greedy y a veces elige acciones subóptimas?** 
>
> **Clave**: Porque la actualización no depende de la acción realmente elegida en $s_{t+1}$, sino del máximo sobre todas las acciones. Así, aunque el agente explore, el objetivo de la actualización supone que a partir del siguiente estado se actuará de forma óptima. Esto permite que la información sobre la mejor acción se propague hacia atrás independientemente del comportamiento real.



El ejmplo muestra efectivamente cómo Q-learning, incluso actuando de forma exploratoria, **aprende la mejor política posible**, ya que las actualizaciones siempre se hacen respecto a la acción óptima en el siguiente estado.

#### Comparación entre SARSA y Q-learning

Aunque SARSA y Q-learning comparten la misma estructura general como algoritmos de control basados en diferencias temporales, existen diferencias fundamentales en la **naturaleza de la política que aprenden** y en cómo estas diferencias se reflejan en su comportamiento.

SARSA actualiza los valores acción-estado siguiendo estrictamente las decisiones reales del agente. Esto implica que la política aprendida incorpora explícitamente los efectos de la exploración, lo que lo convierte en un enfoque *on-policy*. El valor estimado refleja no solo las decisiones óptimas, sino también aquellas tomadas por la política de comportamiento, incluyendo los pasos exploratorios. Esta característica hace que SARSA tienda a adoptar **comportamientos más cautelosos**, especialmente en entornos donde la exploración puede conducir a estados peligrosos o no deseados.

Por el contrario, Q-learning actúa *off-policy*: aprende como si el agente siempre eligiera la mejor acción posible, incluso cuando en la práctica está explorando. Esto le permite **converger hacia la política óptima más agresivamente**, ya que las actualizaciones no dependen de las acciones realmente ejecutadas, sino de las mejores posibles según las estimaciones actuales. Esto puede resultar en un aprendizaje más rápido o más efectivo en entornos donde la política óptima es claramente definible y la exploración no conlleva riesgos significativos.

En términos de estabilidad, SARSA puede resultar más robusto cuando se combinan exploración y aprendizaje en paralelo, especialmente en entornos no deterministas. En cambio, Q-learning tiende a ser más eficiente a largo plazo cuando las condiciones permiten converger hacia una solución global óptima.

En la práctica, SARSA se utiliza a menudo en entornos donde es importante tener en cuenta el efecto real de las decisiones, como por ejemplo en tareas de navegación con estados peligrosos. Q-learning, en cambio, se ha convertido en el algoritmo de referencia en muchos dominios donde se busca rendimiento óptimo, desde juegos hasta control robótico, y ha sido la base conceptual de desarrollos posteriores como Deep Q-Networks (DQN).

### Ventajas del aprendizaje incremental

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

