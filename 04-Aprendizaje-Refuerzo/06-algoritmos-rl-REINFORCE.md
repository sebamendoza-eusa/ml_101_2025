# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Algoritmos de Aprendizaje por Refuerzo: Algoritmos basados en políticas 

### Introducción: Algoritmos basados en valores y basados en políticas

A lo largo de los capítulos anteriores hemos trabajado exclusivamente con **métodos basados en valor**. Estos algoritmos estiman una función de valor —ya sea la función estado-valor $v(s)$ o la función acción-valor $q(s,a)$— y a partir de ella derivan una política. En los métodos de control, la política suele ser $\epsilon$-greedy con respecto a la estimación actual de $q(s,a)$, como en SARSA o Q‑learning. En los métodos de predicción, la política se da fija y el objetivo es aprender sus valores.

Este enfoque ha demostrado ser muy eficaz en una amplia variedad de problemas, especialmente cuando el espacio de acciones es discreto y reducido. Los ejemplos que hemos estudiado —el agente saltarín, los laberintos bidimensionales, el entorno de FrozenLake— se resolvieron satisfactoriamente con estos algoritmos, siempre que se dispusiera de un número suficiente de episodios y de una adecuada parametrización de la función de valor (tabular en nuestros ejemplos).

Sin embargo, los métodos basados en valor presentan limitaciones importantes que conviene conocer antes de abordar enfoques alternativos.

Podemos identificar varias dificulutades o limitaciones. La primera y quizás más evidente aparece cuando el espacio de acciones es continuo. Pensemos, por ejemplo, en un brazo robótico que debe aplicar una fuerza que puede ser cualquier número real entre -1 y 1. Para determinar la acción greedy a partir de la función $q(s,a)$, necesitaríamos maximizar sobre un número infinito de acciones, lo que resulta sencillamente inviable. Una solución inmediata sería discretizar el espacio, convirtiendo el continuo en un conjunto finito de valores, pero esto conlleva una pérdida de resolución y, si se pretende mantener la precisión, un crecimiento exponencial del número de acciones a considerar. En la práctica, la discretización suele ser un mal compromiso.

Otra limitación, quizás más sutil, tiene que ver con la naturaleza determinista de las políticas que se obtienen de los métodos basados en valor. Salvo por la exploración artificial que introduce el parámetro $\epsilon$ en estrategias como $\epsilon$-greedy, la política final es determinista: ante un mismo estado, el agente siempre elegirá la misma acción. Sin embargo, en muchos problemas reales la política óptima es intrínsecamente estocástica. Un ejemplo paradigmático son los juegos de cartas con información imperfecta, como el póker. Allí, la mejor estrategia suele consistir en mezclar acciones con ciertas probabilidades: farolear un 30% de las veces, pasar un 70%, etc. Un método basado en valor no puede representar explícitamente esta aleatoriedad. En el mejor de los casos, podría aproximarla mediante una política $\epsilon$-greedy con un valor fijo de $\epsilon$, pero esa aproximación es muy burda, ya que no permite ajustar las probabilidades de forma diferenciada según el estado. En unos estados puede interesar farolear mucho y en otros poco, y un $\epsilon$ global no captura esa sutileza.

Las dificultades se agravan cuando combinamos métodos off‑policy, como Q‑learning, con aproximadores potentes, como las redes neuronales. Esta combinación puede generar inestabilidades y divergencias, un fenómeno conocido en la literatura como la **tríada mortal**: la interacción entre la aproximación de funciones, el bootstrapping y el entrenamiento off‑policy tiende a romper las garantías de convergencia. Aunque existen técnicas paliativas, como las redes objetivo (target networks) o la repetición de experiencia (experience replay), la convergencia no está asegurada y a menudo se requiere un delicado ajuste de hiperparámetros.

Por último, los métodos basados en valor ofrecen escasas vías para incorporar conocimiento previo sobre la estructura deseada de la política. Imaginemos que, por nuestra experiencia en un dominio, sabemos que en ciertos estados una acción concreta debería ser más probable que otras. En un método basado en valor, este conocimiento no se puede inyectar directamente, porque la política no es un objeto de diseño independiente, sino una consecuencia de la función de valor. Cualquier sesgo debe introducirse a través de la función de valor, lo que resulta indirecto y, a menudo, poco intuitivo.

> **Ejemplo**: Supongamos un robot que debe aprender a caminar. Las acciones son las fuerzas aplicadas a cada articulación, valores continuos en un rango. No podemos enumerar todas las fuerzas posibles, ni maximizar $q(s,a)$ sobre un continuo. Un método basado en política nos permitiría parametrizar directamente la fuerza media y la desviación típica, y aprenderlas mediante gradiente.

#### Enfoque alternativo: aprender directamente la política

Frente a las limitaciones que acabamos de describir, surge una idea tan natural como poderosa: ¿por qué no aprender directamente la política? En lugar de preocuparnos por estimar una función de valor y luego derivar una política a partir de ella, podemos parametrizar la propia política $\pi(a|s,\theta)$ como una función diferenciable que, para cada estado, nos devuelva una distribución de probabilidad sobre las acciones (o una densidad, si el espacio es continuo). El vector $\theta$ contiene los parámetros que ajustaremos para maximizar la recompensa esperada. Este planteamiento se conoce como **aprendizaje basado en política** o **métodos de gradiente de política**.

La principal virtud de este enfoque es su claridad de propósito: optimizamos directamente aquello que nos interesa, que es el rendimiento del agente a largo plazo. No necesitamos calcular una función de valor como paso intermedio, aunque, como veremos más adelante, a veces resulta conveniente incorporarla para reducir la varianza de las estimaciones. La política puede ser determinista o estocástica, y su parametrización debe ser diferenciable para poder aplicar las técnicas de ascenso por gradiente que ya hemos introducido.

Una de las ventajas más apreciables de estos métodos es su capacidad para manejar acciones continuas de forma natural. Recordemos el problema del brazo robótico que debe aplicar una fuerza real entre -1 y 1. Con un enfoque basado en valor, tendríamos que maximizar $q(s,a)$ sobre un continuo, lo que es inviable. En cambio, con un método basado en política podemos parametrizar la propia política como una distribución gaussiana: la acción no se elige maximizando nada, sino que se muestrea de una distribución cuya media y varianza son funciones del estado. El gradiente de la política se propaga a través de estos parámetros, ajustando la media para que genere acciones que conduzcan a mayores recompensas, y la varianza para controlar el grado de exploración.

Otra ventaja fundamental es que estos métodos manejan de manera explícita políticas estocásticas. La política $\pi(a|s,\theta)$ es, por construcción, una distribución de probabilidad. Esto significa que podemos representar y aprender estrategias óptimas que sean intrínsecamente aleatorias, como las que aparecen en juegos de cartas con información imperfecta (el póker, el bridge, el mus). En esos casos, la mejor estrategia no es siempre hacer lo mismo, sino mezclar acciones con ciertas probabilidades. Con un método basado en valor, la única forma de obtener aleatoriedad sería recurrir a trucos externos como $\epsilon$-greedy, que aplica la misma probabilidad de exploración a todos los estados y no aprende qué grado de aleatoriedad es óptimo en cada situación. En los métodos de gradiente de política, la exploración no es un añadido artificial, sino que está integrada en la propia política; el agente decide cuán incierto debe ser su comportamiento.

Los métodos basados en política también presentan mejores propiedades de convergencia cuando se utilizan aproximadores potentes, como redes neuronales. A diferencia de la tríada mortal que acecha a los métodos off‑policy con bootstrapping y aproximación, los algoritmos de gradiente de política suelen ser on‑policy (actualizan con los datos generados por la propia política) y, bajo condiciones bastante suaves, convergen a óptimos locales. Incluso cuando se extienden a variantes off‑policy, existen técnicas para estabilizarlos.

Por último, estos métodos permiten incorporar conocimiento previo de una manera mucho más directa. Si sabemos que en un determinado estado ciertas acciones son inadmisibles, podemos fijar su probabilidad a cero en la parametrización. Si tenemos una intuición sobre qué acción debería ser más probable, podemos inicializar los parámetros para que la política se comporte de forma razonable desde el principio, acelerando así el aprendizaje. En un método basado en valor, en cambio, cualquier sesgo debe introducirse a través de la función de valor, lo que resulta indirecto y a menudo poco intuitivo.

En resumen, los métodos de gradiente de política no vienen a sustituir por completo a los basados en valor, sino a complementarlos. Como veremos más adelante, los algoritmos más potentes combinan ambas filosofías en los llamados métodos actor‑crítico. Pero antes de llegar a esas sofisticaciones, conviene dominar el algoritmo más simple de esta familia: REINFORCE.

#### Estructura general de los métodos de gradiente de política

Una vez que hemos decidido aprender la política directamente, necesitamos una forma de medir su calidad y de mejorarla sistemáticamente. En los métodos basados en valor, el objetivo era que la función de valor se aproximara lo mejor posible a la verdadera, y la política era una consecuencia. Aquí, en cambio, definimos explícitamente una **medida de rendimiento** $J(\theta)$ que depende de los parámetros $\theta$ de la política $\pi_\theta$. En el caso más habitual de tareas episódicas, esta medida suele ser el valor esperado del retorno desde un estado inicial:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[ G(\tau) \right],
$$

donde $G_{\tau} = \sum_{t=0}^{T-1} \gamma^t r_{t+1}$ es el retorno acumulado del episodio. En tareas continuas (sin episodios), se suele emplear la recompensa media por paso. En esencia, $J(\theta)$ cuantifica lo bien que se comporta el agente cuando sigue la política $\pi_\theta$. Nuestro objetivo es encontrar el $\theta$ que haga máxima esta cantidad.

Para maximizar $J(\theta)$ recurrimos a una idea clásica en optimización: el **ascenso por gradiente**. Si conociéramos el gradiente $\nabla_\theta J(\theta)$, podríamos actualizar los parámetros en la dirección de máximo crecimiento:

$$
\theta_{\text{nuevo}} = \theta_{\text{viejo}} + \alpha \nabla_\theta J(\theta_{\text{viejo}}),
$$

donde $\alpha$ es la tasa de aprendizaje. Este proceso se repite iterativamente hasta alcanzar un máximo local. El problema, como es fácil imaginar, es que no disponemos de una expresión analítica de $J(\theta)$ ni de su gradiente, porque dependen de la dinámica del entorno (que desconocemos) y de la distribución de estados inducida por la propia política $\pi_\theta$. No podemos, por tanto, calcular el gradiente de forma exacta.

La solución característica de los métodos de gradiente de política consiste en **estimar** el gradiente a partir de muestras. Es decir, ejecutamos la política $\pi_\theta$ en el entorno (real o simulado), observamos las trayectorias resultantes y con ellas construimos una aproximación ruidosa pero insesgada de $\nabla_\theta J$. A continuación, actualizamos $\theta$ utilizando esa estimación, como en un **ascenso por gradiente estocástico**. Con el tiempo, y bajo condiciones adecuadas, los parámetros convergen a un óptimo local (aunque no necesariamente global).

El corazón de estos algoritmos es, por tanto, disponer de una fórmula que nos permita estimar el gradiente a partir de la experiencia. Esta fórmula la proporciona el **teorema del gradiente de política**, que es uno de los resultados teóricos más importantes del aprendizaje por refuerzo. El teorema establece que:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \, Q_{\pi_\theta}(s,a) \right].
$$

La expresión puede parecer críptica, pero su interpretación es notable. Nos dice que, para estimar el gradiente, no necesitamos conocer cómo varía la distribución de estados con $\theta$ (algo muy complejo), sino que basta con tomar muestras de la política actual y multiplicar el gradiente del logaritmo de la política por el valor de la acción tomada, $Q_{\pi_\theta}(s,a)$. En la práctica, este $Q_{\pi_\theta}(s,a)$ **se sustituye por una estimación muestral,** que en el algoritmo REINFORCE es simplemente el retorno $G(\tau)$ observado desde ese paso hasta el final del episodio.

De este modo, REINFORCE actualiza los parámetros tras cada episodio (o incluso por paso, si se acumulan gradientes) mediante la regla:

$$
\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

Observemos la elegancia de la idea: si el retorno fue alto, el gradiente del logaritmo de la política se multiplica por un factor grande, empujando los parámetros en la dirección que hace más probable la acción tomada. Si el retorno fue bajo o negativo, el empuje es pequeño o incluso en sentido contrario, haciendo menos probable esa acción. No es necesario maximizar sobre acciones, ni mantener una función de valor explícita (aunque luego veremos que incluir una puede reducir la varianza).

Esta estructura general —definir una política parametrizada, medir su rendimiento mediante $J(\theta)$, estimar el gradiente a partir de muestras y actualizar en la dirección del gradiente estocástico— es la base de todos los métodos de gradiente de política. REINFORCE es el ejemplo más simple, pero existen muchas variantes que introducen líneas de base, utilizan funciones de valor para reducir la varianza (actor‑crítico), o incluso combinan gradiente de política con aprendizaje off‑policy. En los siguientes apartados profundizaremos en cada una de estas ideas.

#### Relación entre ambos enfoques

Los métodos basados en valor y los basados en política no son excluyentes. De hecho, los algoritmos más potentes combinan ambos: son los llamados **métodos actor‑crítico**. En ellos, un **actor** (la política) se actualiza mediante gradiente de política, mientras que un **crítico** (una función de valor) estima la ventaja o el retorno esperado, proporcionando una señal de menor varianza para el actor.

En este capítulo nos centraremos primero en los métodos puramente basados en política, empezando por el algoritmo **REINFORCE**, que utiliza retornos muestreados de episodios completos. Posteriormente, introduciremos una línea de base para reducir la varianza, y finalmente presentaremos los fundamentos de los métodos actor‑crítico.

Para reflexionar…

> **¿Puede un método basado en valor representar una política estocástica óptima? ¿Cómo podría hacerse?**
>
> **Clave**: Una política $\epsilon$-greedy es estocástica, pero la aleatoriedad es uniforme (todas las acciones no greedy tienen la misma probabilidad) y no depende del estado. Para políticas estocásticas más generales (por ejemplo, “en este estado, la acción A con 0.7 y la B con 0.3”), los métodos basados en valor no tienen una forma natural de representarlas. Podría usarse una política softmax sobre los valores $q(s,a)$, pero entonces la política dependería de la escala de los valores, y no sería fácil ajustar las probabilidades a valores específicos.

> **En un problema con acciones continuas, ¿cómo se selecciona la acción si solo tenemos $q(s,a)$? ¿Y si tenemos una política parametrizada $\pi(a|s,\theta)$?**
>
> **Clave**: Con $q(s,a)$ habría que resolver un problema de optimización (maximizar $q$ sobre $a$) en cada paso, lo que es costoso y a menudo inviable. Con una política parametrizada, simplemente se muestrea $a \sim \pi(\cdot|s,\theta)$, lo que es inmediato si la distribución es fácil de muestrear (por ejemplo, una gaussiana).



### El ascenso del gradiente

En los métodos basados en valor, el objetivo era encontrar la función de valor óptima (resolviendo las ecuaciones de Bellman o aproximándolas mediante muestras) y luego derivar la política a partir de ella. En los métodos basados en política, en cambio, nos planteamos un problema más directo: **maximizar una medida de rendimiento $J(\theta)$** que depende de los parámetros $\theta$ de la política $\pi_\theta$.

Esta función $J(\theta)$ puede ser, por ejemplo, el valor esperado del retorno desde un estado inicial en tareas episódicas, o la recompensa media por paso en tareas continuas. Lo importante es que queremos encontrar el valor de $\theta$ que hace máxima a $J$.

En general, no disponemos de una expresión analítica de $J(\theta)$ ni de su gradiente. La única información que podemos obtener es a través de muestras: ejecutando la política $\pi_\theta$ en el entorno (real o simulado) y observando las recompensas obtenidas. Esto nos sitúa en el terreno de la **optimización estocástica**.

Una de las técnicas más utilizadas en este contexto es el **ascenso por gradiente** (gradient ascent), que actualiza iterativamente los parámetros en la dirección que, localmente, aumenta más rápidamente el valor de $J$.

#### Idea fundamental del ascenso por gradiente

Imaginemos que nuestra política $\pi_\theta$ es un montañero perdido en la niebla. Quiere alcanzar la cima más alta de una cordillera, pero no ve el camino. Lo único que puede hacer es sentir la inclinación del terreno bajo sus pies. Si el suelo asciende hacia el norte, dará un paso hacia el norte; si la pendiente es más pronunciada hacia el este, se irá hacia el este. Repitiendo este proceso, paso a paso, acabará llegando a alguna cumbre. Eso es, en esencia, el **ascenso por gradiente**: moverse en la dirección de máximo crecimiento de una función.

En nuestro caso, la función que queremos maximizar es el rendimiento esperado $J(\theta)$. Si conociéramos el gradiente $\nabla_\theta J(\theta)$ —es decir, el vector que apunta hacia donde más aumenta $J$—, podríamos actualizar los parámetros con la regla:

$$
\theta_{\text{nuevo}} = \theta_{\text{viejo}} + \alpha \nabla_\theta J(\theta_{\text{viejo}}),
$$

donde $\alpha$ es un pequeño paso que controla cuánto nos movemos en esa dirección. Si $\alpha$ es muy grande, podemos saltarnos la cima y caer al otro lado; si es muy pequeño, tardaremos mucho en llegar. Este esquema iterativo es el mismo que se usa en los problemas de optimización clásicos.

**Pero hay un problema fundamental**: no conocemos el gradiente exacto. La función $J(\theta)$ depende de la dinámica del entorno, que es desconocida, y también de la distribución de estados que genera la propia política. Calcularlo de forma analítica es imposible.

Aquí es donde entra la idea del **gradiente estocástico**. En lugar de calcular el gradiente verdadero, lo **estimamos** a partir de muestras. Por ejemplo, ejecutamos un episodio completo siguiendo la política actual y, con las recompensas observadas, construimos una aproximación $\widehat{\nabla_\theta J}$. Esta estimación será ruidosa, pero si su valor esperado coincide con el gradiente verdadero, podemos usarla para actualizar los parámetros:

$$
\theta_{t+1} = \theta_t + \alpha \widehat{\nabla_\theta J(\theta_t)}.
$$

Aunque cada paso pueda ser un poco errático, si el ruido tiene media cero y elegimos bien $\alpha$, el proceso convergerá a un máximo local. Este es el núcleo de todos los métodos de gradiente de política.

REINFORCE, que es uno de los algoritmos que veremos en este capítulo, utiliza una fórmula muy concreta para estimar el gradiente, que se deriva del **teorema del gradiente de política**. Dicho teorema afirma que:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q_{\pi_\theta}(s,a) \right].
$$

La belleza de esta expresión es que no necesita conocer cómo cambia la distribución de estados con $\theta$; solo requiere muestrear acciones y estimar $Q_{\pi_\theta}(s,a)$. En REINFORCE, hacemos la aproximación más sencilla: sustituimos $Q_{\pi_\theta}(s,a)$ por el retorno muestreado $G_{\tau}$ (la suma de recompensas desde el paso $t$ hasta el final del episodio). De este modo, la actualización queda:

$$
\theta \leftarrow \theta + \alpha \, G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

Interpretación: si el retorno $G_{\tau}$ es alto y positivo, empujamos los parámetros en la dirección que hace más probable la acción que tomamos; si $G_t$ es bajo o negativo, la empujamos en la dirección contraria, haciendo menos probable esa acción. Así, el agente aprende a repetir las acciones que llevan a buenas recompensas y a evitar las que conducen a malos resultados.

Para que todo esto funcione, debemos elegir una **tasa de aprendizaje $\alpha$** adecuada. Si es demasiado grande, las actualizaciones serán violentas y el algoritmo puede diverger; si es demasiado pequeña, el aprendizaje será muy lento. En la práctica, a menudo se prueba con varios valores (por ejemplo, $0.001$, $0.0001$) y se selecciona el que da mejor convergencia. También se puede ir reduciendo $\alpha$ con el tiempo (como $\alpha = 1/t$), lo que garantiza la convergencia en entornos estacionarios, aunque en entornos cambiantes puede ser mejor mantener un $\alpha$ constante para que el agente siga adaptándose.

> En resumen, el ascenso por gradiente estocástico es la brújula que guía a nuestros agentes hacia políticas cada vez mejores, y REINFORCE es el primer y más sencillo ejemplo de cómo aplicarlo en la práctica. En los siguientes apartados veremos cómo mejorar su eficiencia introduciendo una línea de base o combinándolo con una función de valor (actor‑crítico).

#### Por qué ascendemos en lugar de descender

En su día nos ocupamos de algoritmos de Machine Learning clásico, como la regresión lineal o de las redes neuronales. Parte de su estudio se basaba en el método de **descenso por gradiente**: aquel algoritmo que ajustaba los pesos de un modelo para minimizar el error cuadrático medio. En cada paso, se calculaba la pendiente de la función de pérdida y se movía el parámetro en dirección contraria (de ahí lo de "descenso"), porque lo que se buscaba era reducir el error, no aumentarlo.

En aprendizaje por refuerzo, el objetivo es exactamente el opuesto: **maximizar** la recompensa esperada, no minimizarla. Por eso hablamos de **ascenso por gradiente**. La idea matemática es la misma —actualizar parámetros en la dirección que más cambia la función objetivo— pero con el signo cambiado. Si definimos una función de pérdida como $-J(\theta)$, entonces minimizar esa pérdida equivale a maximizar $J(\theta)$. Es decir, el descenso por gradiente aplicado a $-J$ es equivalente al ascenso por gradiente aplicado a $J$.

En la práctica, muchos algoritmos de gradiente de política se implementan precisamente como descenso del negativo de la recompensa esperada. Podemos pensar en ello como si estuviésemos entrenando un modelo para que su error (el negativo de la recompensa) sea lo más pequeño posible. La única diferencia real es la dirección del ajuste: allí donde antes restábamos el gradiente, ahora sumamos. El resto —el cálculo de gradientes, la elección de la tasa de aprendizaje, la convergencia a óptimos locales— es el mismo mundo.

Esta conexión no es solo una curiosidad histórica. Refleja que, en el fondo, tanto la regresión como el refuerzo comparten un mismo andamiaje: el de la optimización iterativa basada en gradientes. Lo que cambia es la naturaleza de la función objetivo y la forma en que obtenemos las muestras para estimar el gradiente. En la regresión lineal, las muestras son pares (entrada, etiqueta) y la función de pérdida es conocida y diferenciable. En el aprendizaje por refuerzo, las muestras son trayectorias completas y la función $J(\theta)$ solo se puede estimar de forma ruidosa. Pero el principio rector es el mismo: moverse poco a poco en la dirección que mejora el comportamiento.

**Para reflexionar…**

> **Si ya conocemos el descenso por gradiente de los algoritmos de ML clásico, ¿qué nuevo desafío introduce el ascenso por gradiente en refuerzo?** 
>
> **Clave**: En ML supervisado, el gradiente se calcula directamente sobre una función de pérdida conocida. En refuerzo, tenemos que estimar el gradiente a partir de interacciones con el entorno, lo que introduce ruido y varianza. Además, la política afecta a la distribución de los datos que se recogen (exploración vs explotación), algo que no ocurre en el aprendizaje supervisado estático.

> **¿Qué sucede si la función $J(\theta)$ tiene múltiples máximos locales? ¿Podemos garantizar que el ascenso por gradiente encontrará el máximo global?**
> **Clave**: En general, no. El ascenso por gradiente (y sus variantes estocásticas) converge a un máximo local, que puede no ser el global. Sin embargo, en la práctica, para muchas tareas los máximos locales son suficientemente buenos. Además, la aleatoriedad de las estimaciones puede ayudar a escapar de máximos locales pobres.

> **¿Por qué necesitamos que la política sea diferenciable respecto a $\theta$?**
> **Clave**: Porque la regla de actualización utiliza el gradiente $\nabla_\theta \log \pi_\theta(a|s)$. Si la política no fuera diferenciable, no podríamos calcular esta dirección de mejora. Por eso las políticas basadas en tablas (como las $\epsilon$-greedy) no son adecuadas para el ascenso por gradiente directo; necesitamos una parametrización suave, como la softmax o redes neuronales.



### Parametrización de la política

En los métodos de gradiente de política, el elemento central es la propia política $\pi(a|s,\theta)$. A diferencia de los métodos basados en valor, donde la política se derivaba de forma indirecta (por ejemplo, mediante una selección $\epsilon$-greedy sobre $q(s,a)$), aquí la política se **parametriza directamente**. Esto significa que elegimos una familia de funciones dependientes de un vector de parámetros $\theta$, y el objetivo del aprendizaje es ajustar $\theta$ para maximizar el rendimiento esperado.

La condición fundamental que debe cumplir esta parametrización es que $\pi(a|s,\theta)$ sea **diferenciable** respecto a $\theta$. Esta propiedad es necesaria para poder calcular el gradiente $\nabla_\theta \log \pi(a|s,\theta)$, que aparece, en general, en todos los algoritmos de gradiente de política.

Además, la política debe definir una distribución de probabilidad válida sobre las acciones: para cada estado $s$ y cada $\theta$, se debe cumplir que $\sum_a \pi(a|s,\theta)=1$ (en el caso discreto) o que la integral sobre el espacio de acciones continuo sea $1$ (en el caso continuo). Asimismo, es habitual (aunque no estrictamente necesario) que la política asigne probabilidad positiva a todas las acciones, al menos durante el entrenamiento, para garantizar la exploración.

#### Parametrización para acciones discretas: softmax

Cuando el espacio de acciones es discreto y de tamaño moderado, la parametrización más común consiste en asignar a cada par $(s,a)$ una **preferencia numérica** $h(s,a,\theta)$, que puede ser una función lineal de características o la salida de una red neuronal. Las preferencias se convierten en probabilidades mediante la función **softmax** (también conocida como distribución de Boltzmann o Gibbs):

$$
\pi(a|s,\theta) = \frac{\exp\bigl(h(s,a,\theta)\bigr)}{\sum_{b} \exp\bigl(h(s,b,\theta)\bigr)}.
$$

El denominador asegura que las probabilidades sumen $1$. Esta parametrización tiene varias propiedades interesantes:

- Es **diferenciable** respecto a $\theta$ (si $h$ lo es). El gradiente del logaritmo resulta:
  $$
  \nabla_\theta \log \pi(a|s,\theta) = \nabla_\theta h(s,a,\theta) - \sum_{b} \pi(b|s,\theta) \nabla_\theta h(s,b,\theta),
  $$
  que puede interpretarse como la diferencia entre el gradiente de la preferencia de la acción tomada y el gradiente medio ponderado de todas las preferencias.

- La política es **estocástica** por naturaleza, salvo en el límite en que una preferencia domina infinitamente sobre las demás. Esto permite explorar de forma natural, sin necesidad de un mecanismo separado como $\epsilon$-greedy (aunque en la práctica a menudo se combina con exploración adicional).

- Las preferencias pueden inicializarse a cero (lo que da una política uniforme) o con valores que reflejen conocimiento previo. A diferencia de los métodos basados en valor, donde los valores $q(s,a)$ estaban acotados por las recompensas, las preferencias no tienen una interpretación directa en términos de retorno; solo importan sus diferencias relativas.

Un caso particular muy utilizado es la **parametrización lineal** de las preferencias:

$$
h(s,a,\theta) = \theta^\top x(s,a),
$$

donde $x(s,a)$ es un vector de características del par estado-acción. Esta elección, aunque simple, ya permite capturar interacciones a través de las características. En problemas más complejos, $h$ puede ser la salida de una red neuronal profunda que toma como entrada una representación del estado.

> **Ejemplo**: En el agente saltarín, el espacio de acciones tiene dos acciones: avanzar (0) y saltar (1). Podemos definir una preferencia lineal con una única característica para cada par: por ejemplo, $x(s=0,a=0)=1$, $x(s=0,a=1)=0$, $x(s=2,a=0)=0$, $x(s=2,a=1)=1$, y así sucesivamente. La política softmax asignará probabilidades suaves según los valores de $\theta$. A medida que aprendemos, los parámetros se ajustarán para favorecer saltar en el estado 0 y avanzar en el estado 2.

#### Caso particular: Parametrización para dos acciones. La función sigmoide

Cuando el espacio de acciones consta exactamente de dos posibilidades, existe una forma especialmente sencilla y natural de parametrizar una política estocástica. En lugar de asignar una preferencia a cada acción por separado, podemos definir una única función que determine la probabilidad de elegir una de ellas, mientras que la probabilidad de la otra se obtiene por complemento.

Supongamos que en un estado $s$ el agente puede ejecutar dos acciones, que llamaremos $a_0$ y $a_1$. Elegimos una de ellas como referencia, por ejemplo $a_1$, y definimos una función $f(s,\theta)$ que depende de un conjunto de parámetros $\theta$. La política se define entonces mediante la función sigmoide (o logística):

$$
\pi(a_1|s,\theta) = \frac{1}{1 + e^{-f(s,\theta)}}, \qquad
\pi(a_0|s,\theta) = 1 - \pi(a_1|s,\theta) = \frac{e^{-f(s,\theta)}}{1 + e^{-f(s,\theta)}}.
$$

La sigmoide transforma cualquier número real en el intervalo $(0,1)$. Cuando $f(s,\theta)$ es muy grande y positivo, $\pi(a_1|s,\theta)$ se acerca a $1$; cuando es muy negativo, se acerca a $0$; y cuando $f(s,\theta)=0$, ambas acciones son equiprobables con probabilidad $0.5$.

En el caso más simple, $f(s,\theta)$ puede ser una función lineal de características del estado. Por ejemplo, si el estado se representa mediante un vector de características $x(s)$, podemos escribir $f(s,\theta) = \theta^\top x(s)$. Si solo hay un parámetro (es decir, el estado tiene una única característica), la política queda determinada por un único número real $\theta$.

##### **Ejemplo con el agente saltarín (estado inicial)**

Consideremos el estado $0$ del problema del agente saltarín, donde las dos acciones son avanzar y saltar. Parametrizamos la política con un único parámetro $\theta$ (sin características adicionales), de modo que $f(0,\theta)=\theta$. La probabilidad de saltar es:

$$
\pi(\text{saltar}|\theta) = \frac{1}{1 + e^{-\theta}}.
$$

Veamos cómo varía esta probabilidad con distintos valores de $\theta$:

- Si $\theta = 0$, entonces $\pi(\text{saltar}) = 0.5$. El agente es completamente indiferente: elige saltar o avanzar con igual probabilidad.
- Si $\theta = 1$, tenemos $\pi(\text{saltar}) = \frac{1}{1+e^{-1}} \approx \frac{1}{1+0.3679} \approx 0.731$. El agente prefiere saltar en aproximadamente tres de cada cuatro ocasiones.
- Si $\theta = 2$, $\pi(\text{saltar}) = \frac{1}{1+e^{-2}} \approx \frac{1}{1+0.1353} \approx 0.881$. La preferencia por saltar es aún más acusada.
- Si $\theta = -1$, $\pi(\text{saltar}) = \frac{1}{1+e^{1}} \approx \frac{1}{1+2.718} \approx 0.269$. Ahora el agente prefiere avanzar (probabilidad de avanzar $1-0.269 = 0.731$).
- Si $\theta$ es muy grande, por ejemplo $\theta = 5$, $\pi(\text{saltar}) \approx 0.993$, prácticamente siempre saltará.
- Si $\theta$ es muy negativo, $\theta = -5$, $\pi(\text{saltar}) \approx 0.0067$, casi nunca saltará.

De este modo, el parámetro $\theta$ controla la tendencia del agente a elegir una acción u otra. Un valor positivo indica preferencia por saltar; un valor negativo, preferencia por avanzar. La magnitud de $\theta$ indica la intensidad de esa preferencia: cuanto mayor sea $|\theta|$, más determinista es la política.

##### **Generalización con características**

Si el estado dispone de varias características, podemos definir $f(s,\theta) = \theta_1 x_1(s) + \theta_2 x_2(s) + \dots$. Por ejemplo, en un problema unidimensional sin agujeros (un simple pasillo), podríamos usar como característica la posición del agente. Así, la preferencia por saltar (o por avanzar) dependería linealmente de la posición, permitiendo que la política sea diferente en cada casilla.

##### **Relación con la softmax**

Cuando hay dos acciones, la parametrización softmax con dos preferencias $h(s,a_0)$ y $h(s,a_1)$ es equivalente a la parametrización sigmoide con $f(s,\theta) = h(s,a_1) - h(s,a_0)$. En efecto:

$$
\frac{e^{h(s,a_1)}}{e^{h(s,a_0)}+e^{h(s,a_1)}} = \frac{1}{1 + e^{-(h(s,a_1)-h(s,a_0))}}.
$$

Ambas formas representan exactamente la misma familia de políticas, pero la sigmoide utiliza un parámetro menos (la diferencia de preferencias), lo que simplifica el modelo cuando el número de acciones es dos.

En resumen, la parametrización sigmoide es una herramienta simple y potente para definir políticas estocásticas en problemas con dos acciones. Su interpretación es directa: el parámetro $\theta$ (o combinación de parámetros) determina la probabilidad de elegir una acción frente a la otra, y su signo y magnitud indican la dirección e intensidad de la preferencia.

#### Parametrización para acciones continuas: distribuciones gaussianas

Cuando el espacio de acciones es continuo, no podemos enumerar las acciones. Una solución habitual es modelar la política como una **distribución gaussiana (normal)** multivariante (o univariante, según la dimensión). En este caso, la política emite la media y (a veces) la desviación típica de la distribución, y la acción se muestrea de dicha distribución.

Concretamente, supongamos que la acción es un escalar $a \in \mathbb{R}$. La política gaussiana se define como:

$$
\pi(a|s,\theta) = \frac{1}{\sigma(s,\theta)\sqrt{2\pi}} \exp\left(-\frac{\bigl(a - \mu(s,\theta)\bigr)^2}{2\sigma(s,\theta)^2}\right),
$$

donde $\mu(s,\theta)$ es la media y $\sigma(s,\theta) > 0$ es la desviación típica. Ambos pueden ser funciones parametrizadas (por ejemplo, redes neuronales) que toman el estado como entrada. Para asegurar que $\sigma$ sea positiva, a menudo se parametriza como $\sigma = \exp(\rho(s,\theta))$ o mediante una transformación softplus.

En este caso, la política es diferenciable respecto a $\theta$ (si $\mu$ y $\sigma$ lo son). El gradiente del logaritmo se puede calcular analíticamente. Para una muestra $a$, tenemos:

$$
\nabla_\theta \log \pi(a|s,\theta) = \frac{a - \mu(s,\theta)}{\sigma(s,\theta)^2} \nabla_\theta \mu(s,\theta) + \left( \frac{(a - \mu(s,\theta))^2}{\sigma(s,\theta)^3} - \frac{1}{\sigma(s,\theta)} \right) \nabla_\theta \sigma(s,\theta).
$$

Aunque la expresión parezca compleja, su implementación es directa si se dispone de bibliotecas de diferenciación automática.

La principal ventaja de las políticas gaussianas es que permiten explorar el espacio de acciones continuo de forma natural, ya que cada acción tiene una probabilidad (densidad) positiva. Además, la varianza puede ser aprendida (o fijada) para ajustar el grado de exploración.

> **Ejemplo**: Un brazo robótico debe aplicar un par de fuerzas entre $-10$ N y $+10$ N. Podemos parametrizar $\mu(s,\theta)$ como una red neuronal que toma la posición y velocidad actuales, y $\sigma$ como un parámetro fijo o aprendido. En cada paso, muestreamos $a \sim \mathcal{N}(\mu, \sigma^2)$ y la aplicamos. Tras observar la recompensa (por ejemplo, el acercamiento a un objetivo), actualizamos $\theta$ para que $\mu$ tienda a producir acciones que conduzcan a mayores recompensas.

#### Otras parametrizaciones y consideraciones prácticas

La elección de la parametrización no es única ni arbitraria; debe adaptarse a la naturaleza del problema que estamos resolviendo. La softmax para acciones discretas y la gaussiana para acciones continuas son solo dos ejemplos comunes, pero existen muchas otras posibilidades, cada una con sus propias ventajas según el dominio.

Cuando el espacio de acciones es discreto pero tiene muchas opciones (por ejemplo, elegir una carta de una baraja o una casilla en un tablero grande), la generalización natural de la softmax es la **distribución categórica**. En esencia, se trata de asignar una preferencia a cada acción y normalizar para que las probabilidades sumen uno. Es la misma idea que la softmax, pero aplicada a un número arbitrario de acciones. No hay nada conceptualmente nuevo; simplemente el denominador incluye tantos términos como acciones haya.

Si las acciones están acotadas en un intervalo, como por ejemplo ajustar la intensidad de un calentador entre 0 y 1, una distribución **beta** puede ser una elección elegante. La distribución beta tiene dos parámetros que controlan su forma y puede concentrarse en los extremos o en el centro, lo que la hace muy flexible para representar preferencias en un rango finito. Su densidad es cero fuera de $[0,1]$, por lo que nunca se generan acciones inválidas.

Cuando la política óptima es multimodal —es decir, hay varias regiones igualmente buenas en el espacio de acciones— una simple gaussiana no bastará, porque solo tiene un pico. En esos casos se recurre a **mixturas de gaussianas**, que combinan varias campanas con diferentes medias y varianzas. La política muestrea primero qué componente de la mezcla usar y luego genera una acción a partir de esa gaussiana. Esto permite representar, por ejemplo, que a veces convenga girar bruscamente a la izquierda y a veces suavemente a la derecha, pero nunca algo intermedio.

En el otro extremo, si sabemos que la política óptima es determinista (siempre la misma acción en cada estado), podemos parametrizar directamente una función determinista $a = \mu(s,\theta)$ sin componente aleatorio. Esto es lo que hacen los métodos de **gradiente de política determinista** (DDPG y variantes). Para explorar, se añade ruido externo a la acción, en lugar de depender de la varianza de la política. Este enfoque suele ser más eficiente en espacios de acción continuos de alta dimensión, porque se evita tener que estimar una distribución completa.

En la práctica, la parametrización no es un mero detalle técnico. Refleja el conocimiento que tenemos sobre el problema. Si sospechamos que la mejor política es casi determinista, podemos fijar la varianza de una gaussiana a un valor pequeño o directamente usar una política determinista con ruido de exploración. Si el problema requiere exploración continua y la acción óptima puede variar suavemente, una gaussiana con varianza aprendida suele funcionar bien. Si la acción está acotada de forma natural, usar una distribución beta o una gaussiana truncada evita tener que proyectar las acciones al rango válido.

Conviene recordar que, incluso con una parametrización suave y diferenciable, el ascenso por gradiente estocástico no garantiza encontrar el máximo global de $J(\theta)$. Puede quedarse atrapado en un óptimo local. Sin embargo, en la mayoría de los problemas prácticos de aprendizaje por refuerzo, estos óptimos locales son suficientemente buenos, y el ruido inherente a las estimaciones del gradiente ayuda a veces a escapar de los peores. Lo importante es elegir una parametrización que permita expresar la política que intuimos como razonable, y luego dejar que el algoritmo ajuste los parámetros.

#### Diferencias con la parametrización de la función de valor

En los métodos basados en valor, la parametrización (por ejemplo, redes neuronales para $q(s,a)$) no necesita cumplir la condición de suma de probabilidades, porque no es una política. En cambio, en la parametrización de la política debemos asegurar que la salida sea una distribución válida. Esto a menudo se consigue mediante capas finales apropiadas (softmax para acciones discretas, o parámetros de escala para continuas).

Otra diferencia importante es que la política parametrizada puede ser **más simple** que la función de valor. Por ejemplo, en un problema de navegación con obstáculos, la política óptima puede ser casi lineal (girar ligeramente para evitar colisiones), mientras que la función de valor puede ser muy compleja (dependiendo de la distancia exacta a los obstáculos). En estos casos, aprender la política directamente es más eficiente.

**Para reflexionar…**

> **¿Por qué la política softmax no es adecuada si el espacio de acciones es muy grande (por ejemplo, millones de acciones)?**  
> **Clave**: Calcular el denominador (la suma sobre todas las acciones) sería prohibitivo. En esos casos se recurre a aproximaciones como el muestreo de importancia o a políticas parametrizadas de forma diferente (por ejemplo, políticas que emiten directamente una acción sin normalizar, como una red que predice la mejor acción, aunque entonces se pierde la interpretación probabilística).

> **En una política gaussiana con $\sigma$ fijo, ¿cómo se controla la exploración?**  
> **Clave**: Si $\sigma$ es grande, las acciones muestreadas se desviarán mucho de la media, explorando más. Si $\sigma$ es pequeño, la política será casi determinista, explotando la media aprendida. A menudo se comienza con $\sigma$ grande y se reduce gradualmente (annealing) para pasar de exploración a explotación. También se puede aprender $\sigma$ para que se ajuste automáticamente.

### Introducción a los algoritmos del gradiente de la política

Hasta ahora hemos explorado la idea de aprender la política directamente mediante el ascenso por gradiente, y hemos visto las ventajas de este enfoque: un manejo natural de acciones continuas, políticas estocásticas explícitas y mejor comportamiento con aproximadores potentes.

Pero, ¿cómo se traduce esta idea en un algoritmo concreto? ¿Cómo podemos actualizar los parámetros $\theta$ de la política $\pi_\theta$ utilizando solo la experiencia del agente? La respuesta la proporcionan los **algoritmos del gradiente de la política**. Su característica común es que estiman el gradiente de la función de rendimiento $J(\theta)$ a partir de muestras y luego actualizan $\theta$ en la dirección de ese gradiente.

El más sencillo de todos es **REINFORCE**, que utiliza retornos completos de episodios (como en Monte Carlo) y no necesita una función de valor auxiliar. Aunque es conceptualmente claro, suele tener una varianza elevada. Para reducir esa varianza se introduce una **línea de base**, que puede ser una estimación del valor del estado. Finalmente, la combinación de una política parametrizada (el actor) con una función de valor aprendida (el crítico) da lugar a los **métodos actor-crítico**, que son la base de muchos sistemas modernos de aprendizaje por refuerzo profundo.

En este módulo recorreremos este camino paso a paso, desde REINFORCE hasta el actor-crítico, ilustrando cada paso con ejemplos y, al final, con una implementación en Python para el problema del péndulo invertido (una tarea clásica de control con acciones continuas).

El corazón de todos estos algoritmos es el **teorema del gradiente de política**. Para entenderlo, recordemos que queremos maximizar $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$, donde $G_0$ es el retorno desde el inicio de un episodio. El gradiente $\nabla_\theta J(\theta)$ nos indica cómo cambiar $\theta$ para aumentar el rendimiento.

El teorema establece una expresión sorprendentemente sencilla:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q_{\pi_\theta}(s,a) \right].
$$

No entraremos en la demostración completa, pero conviene destacar dos aspectos. Primero, el gradiente no depende de la derivada de la distribución de estados, que sería muy compleja de calcular. Segundo, la expresión es una esperanza que podemos aproximar mediante muestras: basta con ejecutar la política, registrar los pares $(s_t, a_t)$, calcular el retorno (o una estimación de $Q$) y multiplicar por el gradiente del logaritmo de la política.

En el caso de REINFORCE, aproximamos $Q_{\pi_\theta}(s_t, a_t)$ por el retorno muestreado $G_t$ (la suma de recompensas desde $t$ hasta el final del episodio). Así obtenemos una estimación del gradiente:

$$
\widehat{\nabla_\theta J} = G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

Esta estimación es insesgada (su valor esperado es el gradiente verdadero), pero tiene alta varianza, porque $G_t$ puede fluctuar mucho de un episodio a otro.

#### Algoritmo REINFORCE

REINFORCE es el algoritmo más sencillo dentro de los métodos de gradiente de política. Su nombre, acuñado por Ronald Williams en 1992, refleja la idea de que refuerza (reinforces) las acciones que conducen a buenos retornos. Se trata de un algoritmo de Monte Carlo para el control, lo que significa que **aprende a partir de episodios completos**, sin necesidad de conocer el modelo del entorno.

Ya hemos visto tambien la justificacion de sustituir la función accion-valor por el retorno muestreado

$$
\theta \leftarrow \theta + \alpha \, G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

Esta expresión tiene una interpretación intuitiva muy clara. Supongamos que después de tomar la acción $a_t$ en el estado $s_t$, el retorno $G_t$ es alto y positivo. Entonces multiplicamos el gradiente del logaritmo de la política por un número grande, y al sumarlo a $\theta$ hacemos que la política aumente la probabilidad de elegir $a_t$ en $s_t$ en el futuro. Si $G_t$ es bajo o negativo, el producto es pequeño o negativo, y la probabilidad de esa acción disminuye. De esta manera, el agente aprende a repetir las acciones que llevan a buenos resultados y a evitar las que conducen a malos resultados.

##### Descripción paso a paso del algoritmo

El algoritmo REINFORCE funciona de la siguiente manera.

**Inicialización**: Se elige una parametrización para la política $\pi_\theta(a|s)$ que sea diferenciable respecto a $\theta$. Por ejemplo, para acciones discretas se puede usar una softmax con preferencias lineales; para acciones continuas, una distribución gaussiana. Los parámetros $\theta$ se inicializan con valores pequeños, a menudo todos ceros, lo que suele dar una política uniforme o una distribución centrada. También se fija una tasa de aprendizaje $\alpha > 0$, normalmente un número pequeño como $0.001$ o $0.0001$.

**Generación de un episodio**: Se ejecuta la política actual $\pi_\theta$ en el entorno (real o simulado) hasta que se alcance un estado terminal. Durante este proceso, se registra en una lista cada paso: el estado $s_t$, la acción elegida $a_t$ y la recompensa recibida $r_{t+1}$. Al final del episodio se tiene una secuencia completa:

$$
s_0, a_0, r_1, s_1, a_1, r_2, s_2, \dots, s_T,
$$

donde $s_T$ es el estado terminal (o el último paso si se ha alcanzado un número máximo de pasos).

**Cálculo de los retornos $G_t$**: Una vez terminado el episodio, se recorren los pasos hacia atrás, comenzando desde el final, para calcular el retorno acumulado desde cada instante $t$. La fórmula recursiva es $G_t = r_{t+1} + \gamma G_{t+1}$, comenzando con $G_T = 0$ (el retorno después del estado terminal es cero). Este cálculo es eficiente y evita sumar repetidamente las mismas recompensas. Por ejemplo, si el episodio tiene $T$ pasos, primero se calcula $G_{T-1} = r_T$, luego $G_{T-2} = r_{T-1} + \gamma G_{T-1}$, y así sucesivamente hasta $G_0$.

**Actualización de los parámetros**: Para cada paso $t$ del episodio, se calcula el gradiente $\nabla_\theta \log \pi_\theta(a_t|s_t)$ con los parámetros actuales (los mismos que se usaron para generar el episodio, pues todavía no se han modificado). A continuación, se actualiza $\theta$ sumando $\alpha \, G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t)$. Esta actualización se puede hacer inmediatamente después de calcular cada $G_t$ (mientras se recorre hacia atrás) o al final, acumulando todas las contribuciones.

**Repetición**: Se repite todo el proceso (generar un episodio, calcular retornos, actualizar parámetros) durante muchos episodios. Con el tiempo, los parámetros $\theta$ convergen a un valor que (esperamos) maximiza el rendimiento esperado.

##### El gradiente del logaritmo de la política

Para poder aplicar la regla de actualización, necesitamos una expresión concreta de $\nabla_\theta \log \pi_\theta(a|s)$. Esta expresión depende de cómo hayamos parametrizado la política. Ilustremos con dos casos habituales.

**Caso 1: política softmax con preferencias lineales**. Supongamos que para cada acción $a$ tenemos una preferencia $h(s,a,\theta) = \theta_a^\top x(s,a)$ (donde $\theta_a$ es un vector de parámetros asociado a esa acción). Entonces:

$$
\pi(a|s,\theta) = \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}.
$$

El gradiente del logaritmo resulta ser:

$$
\nabla_{\theta_a} \log \pi(a|s,\theta) = x(s,a) - \sum_b \pi(b|s,\theta) x(s,b).
$$

Para la acción $a$, el gradiente respecto a sus propios parámetros es $x(s,a)$ menos la media ponderada de las características de todas las acciones. Para las demás acciones $b \neq a$, el gradiente respecto a $\theta_b$ es simplemente $-\pi(b|s,\theta) x(s,b)$ (pues la derivada solo afecta al denominador). Esta expresión es sencilla de implementar.

**Caso 2: política gaussiana para acciones continuas**. Si la acción es un escalar y la política es $\pi(a|s,\theta) = \mathcal{N}(\mu(s,\theta), \sigma^2)$ con $\sigma$ fijo, entonces:

$$
\log \pi(a|s,\theta) = -\frac{(a-\mu(s,\theta))^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2).
$$

Derivando respecto a $\theta$ obtenemos:

$$
\nabla_\theta \log \pi(a|s,\theta) = \frac{a-\mu(s,\theta)}{\sigma^2} \nabla_\theta \mu(s,\theta).
$$

Si además la varianza $\sigma^2$ es aprendida, aparecen términos adicionales, pero la idea central es la misma.

##### Ejemplo detallado con el agente saltarín

Para que el algoritmo quede claro, apliquemos REINFORCE al agente saltarín estocástico, pero centrándonos únicamente en el **estado 0**. Supondremos que el agente ya ha aprendido a comportarse correctamente en el estado 2 (por ejemplo, siempre avanza) y nos concentramos en aprender la política en el estado inicial. De hecho, para simplificar, consideraremos que el agente siempre realiza la misma secuencia después de salir del estado 0: si llega a 2, entonces avanza y alcanza la meta. De este modo, el retorno $G_0$ depende únicamente de la acción tomada en el estado 0 y de la estocasticidad del entorno.

Parametrizamos la política en el estado 0 mediante una sigmoide (pues es más sencillo cuando solo hay dos acciones). Definimos un único parámetro $\theta$, y la probabilidad de saltar es:

$$
\pi(\text{saltar} | \theta) = \frac{1}{1 + e^{-\theta}}, \qquad
\pi(\text{avanzar} | \theta) = 1 - \pi(\text{saltar} | \theta).
$$

Inicializamos $\theta = 0$, lo que da $\pi(\text{saltar}) = 0.5$. Fijamos la tasa de aprendizaje $\alpha = 0.1$ y el descuento $\gamma = 0.9$. Recordemos los retornos esperados (que no conoce el agente, pero nosotros los usamos para simular el entorno): si el agente salta, con probabilidad 0.9 va a 2 y luego a la meta, obteniendo $G_0 = -1 + 0.9 \times 19 = 16.1$; con probabilidad 0.1 cae en el agujero y obtiene $G_0 = -10$. Si avanza, con probabilidad 0.8 cae en el agujero ($G_0 = -10$) y con probabilidad 0.2 permanece en 0 ($G_0 = -1$), pero para simplificar supondremos que avanzar siempre lleva al agujero (el caso más desfavorable), es decir, $G_0 = -10$ con probabilidad 1 (esto es una simplificación, pero sirve para ilustrar).

**Episodio 1**: El agente, con $\theta=0$, elige saltar (por azar, ya que ambas acciones tienen probabilidad 0.5). El entorno, en este caso, produce el resultado favorable (90% de probabilidad): va a 2 y luego a la meta. El retorno observado es $G_0 = 16.1$. 

Calculamos el gradiente del logaritmo de la política para la acción saltar:
$$
\nabla_\theta \log \pi(\text{saltar}|\theta) = 1 - \pi(\text{saltar}|\theta) = 1 - 0.5 = 0.5.
$$

Actualizamos $\theta$:

$$
\theta \leftarrow 0 + 0.1 \times 16.1 \times 0.5 = 0.805.
$$

La nueva probabilidad de saltar es $\pi(\text{saltar}) = 1/(1+e^{-0.805}) \approx 0.691$. El agente ahora prefiere saltar.

**Episodio 2**: Con $\theta = 0.805$, la política elige saltar con probabilidad 0.691. Supongamos que vuelve a elegir saltar. Esta vez el entorno produce el resultado desfavorable (10% de probabilidad): cae en el agujero, obteniendo $G_0 = -10$. 

Calculamos el gradiente para la acción saltar (con el $\theta$ actual):
$$
\pi(\text{saltar}) = 0.691, \quad \nabla_\theta \log \pi(\text{saltar}) = 1 - 0.691 = 0.309.
$$

Actualizamos:

$$
\theta \leftarrow 0.805 + 0.1 \times (-10) \times 0.309 = 0.805 - 0.309 = 0.496.
$$

Ahora $\pi(\text{saltar}) = 1/(1+e^{-0.496}) \approx 0.622$. La preferencia por saltar ha disminuido, pero sigue siendo superior a 0.5.

**Episodio 3**: Con $\theta = 0.496$, el agente vuelve a elegir saltar (probabilidad 0.622). Supongamos que ahora el entorno es favorable: $G_0 = 16.1$. El gradiente es $1 - 0.622 = 0.378$. Actualización:

$$
\theta \leftarrow 0.496 + 0.1 \times 16.1 \times 0.378 = 0.496 + 0.608 = 1.104.
$$

Probabilidad de saltar: $\pi(\text{saltar}) = 1/(1+e^{-1.104}) \approx 0.751$.

Observamos que $\theta$ fluctúa hacia arriba y hacia abajo según los resultados de cada episodio. Sin embargo, la tendencia es a crecer porque los episodios favorables (con $G_0=16.1$) tienen un efecto positivo mayor en magnitud que los desfavorables ($-10$), aunque estos últimos también ocurren con cierta frecuencia. Si continuamos durante muchos episodios, $\theta$ se estabilizará en un valor positivo que refleja el equilibrio entre la probabilidad de éxito (0.9) y la de fracaso (0.1). Ese valor es el que hace que la dirección esperada del gradiente sea cero, y corresponde a la política que maximiza el rendimiento esperado (que, recordemos, es saltar con una probabilidad muy alta, pero no del 100% porque siempre existe el riesgo de caer en el agujero).

**¿Qué pasa si el agente elige avanzar?** Aunque no lo hemos mostrado en los episodios, si en algún momento la política eligiera avanzar, obtendría un retorno muy negativo (por ejemplo, $-10$). El gradiente para la acción avanzar es $\nabla_\theta \log \pi(\text{avanzar}) = -\pi(\text{saltar})$, que es negativo. Al multiplicar por $G_0$ (negativo) y por $\alpha$, el producto sería positivo, lo que aumentaría $\theta$ ligeramente (es decir, haría más probable saltar). De hecho, elegir avanzar refuerza la acción contraria, porque el retorno negativo indica que avanzar es malo, y la actualización disminuye la probabilidad de avanzar (aumenta la de saltar). Así, el algoritmo tiende naturalmente a favorecer la acción que conduce a mayores recompensas.

#### Limitaciones de REINFORCE y motivación para la línea de base

Aunque REINFORCE es conceptualmente sencillo y tiene la ventaja de ser insesgado (la esperanza de la actualización es el gradiente verdadero), adolece de una alta varianza. ¿Por qué? Porque el retorno $G_t$ puede variar enormemente de un episodio a otro, especialmente si las recompensas son muy dispersas o si los episodios son largos. Esta alta varianza hace que la convergencia sea lenta y que los parámetros $\theta$ puedan oscilar mucho.

Además, REINFORCE no aprovecha la estructura del problema: la misma muestra $G_t$ se usa para actualizar todos los pasos del episodio, aunque algunos estados puedan tener valores muy diferentes. Por ejemplo, en un episodio largo, un $G_t$ alto puede deberse a una acción excelente en un paso tardío, pero también contribuirá a actualizar los pasos anteriores, que quizás no merecían tanto crédito.

Para mitigar este problema, se introduce una **línea de base**. La idea es restar una cantidad $b(s_t)$ del retorno $G_t$ antes de multiplicar por el gradiente. La nueva actualización es:

$$
\theta \leftarrow \theta + \alpha \, (G_t - b(s_t)) \, \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

Si $b(s_t)$ no depende de la acción, el valor esperado de la actualización no cambia (porque $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)] = 0$). Sin embargo, la varianza puede reducirse si $b(s_t)$ se elige adecuadamente. La elección más común es utilizar el valor del estado $v_\pi(s_t)$ como línea de base. Así, el factor $(G_t - v_\pi(s_t))$ se interpreta como la **ventaja** de tomar la acción $a_t$: cuánto mejor (o peor) ha sido el resultado en comparación con lo que normalmente se espera de ese estado. Aprender esta línea de base requiere mantener una función de valor adicional, lo que nos lleva a los métodos actor-crítico, que veremos más adelante.



> [!note]
>
> A modo de recapitulación, el algoritmo REINFORCE se puede describir con los siguientes pasos:
>
> 1. **Inicializar** los parámetros $\theta$ (por ejemplo, todos ceros) y fijar la tasa de aprendizaje $\alpha$.
> 2. **Repetir** para cada episodio:
>    - Generar una trayectoria completa $s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T$ siguiendo la política actual $\pi_\theta$.
>    - Inicializar $G = 0$.
>    - **Recorrer** los pasos desde $t = T-1$ hasta $0$:
>      - $G = r_{t+1} + \gamma G$ (así se calcula $G_t$).
>      - Calcular $\nabla = \nabla_\theta \log \pi_\theta(a_t|s_t)$ usando la parametrización elegida.
>      - Actualizar $\theta \leftarrow \theta + \alpha G \nabla$.
> 3. **Terminar** cuando los parámetros converjan o después de un número fijo de episodios.
>
> Este algoritmo es fácil de implementar y funciona bien en problemas pequeños o de baja dimensionalidad. Sin embargo, como ya se ha mencionado, su principal debilidad es la varianza. En la siguiente sección veremos cómo reducir esa varianza introduciendo una línea de base, y posteriormente cómo combinar la política con una función de valor aprendida en el marco actor-crítico.

### El algoritmo REINFORCE con línea de base

Hemos visto que REINFORCE actualiza los parámetros de la política mediante la regla $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$. El retorno $G_t$ puede ser muy variable: a veces muy alto, a veces muy bajo, incluso para la misma acción en circunstancias similares. Esta variabilidad se traduce en una alta **varianza** de la estimación del gradiente, lo que hace que el aprendizaje sea lento y errático. Necesitamos una forma de reducir esa varianza sin alterar la dirección hacia la que apunta el gradiente de forma esperada.

Supongamos que restamos una cantidad $b(s_t)$ del retorno $G_t$ antes de multiplicar por el gradiente. La nueva actualización sería:

$$
\theta \leftarrow \theta + \alpha \, (G_t - b(s_t)) \, \nabla_\theta \log \pi_\theta(a_t|s_t).
$$

La pregunta inmediata es: ¿seguimos apuntando en la dirección correcta en promedio? Es decir, ¿el valor esperado de $(G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t)$ sigue siendo igual a $\nabla_\theta J(\theta)$? Para responder, descomponemos la esperanza:

$$
\mathbb{E}\left[ (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t) \right] = 
\mathbb{E}\left[ G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] - 
\mathbb{E}\left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right].
$$

El primer término es precisamente $\nabla_\theta J(\theta)$ (por el teorema del gradiente de política). El segundo término, si $b(s_t)$ no depende de la acción $a_t$, podemos calcularlo condicionando al estado $s_t$:

$$
\mathbb{E}\left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right] = 
\mathbb{E}_{s_t \sim \mu} \left[ b(s_t) \, \mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \right].
$$

Pero es una propiedad conocida que $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$ para cualquier estado $s$. La demostración es sencilla:
$$
\sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0
$$
Por tanto, el segundo término se anula. Conclusión: **introducir una línea de base $b(s_t)$ que no dependa de la acción no introduce sesgo en la estimación del gradiente**. Podemos restar cualquier función del estado (o incluso una constante) y la esperanza de la actualización seguirá siendo el gradiente verdadero.

¿Por qué se reduce la varianza con esta transformacion? La varianza de una variable aleatoria $X$ es $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$. Al restar una constante (o una función del estado), estamos cambiando $X$ por $X - c$. La esperanza se modifica en $-c$, pero la varianza también cambia: $\text{Var}(X-c) = \text{Var}(X)$ si $c$ es constante. Sin embargo, si $c$ no es constante sino que depende del estado, la varianza puede disminuir si $c$ está correlacionada positivamente con $X$. En nuestro caso, $X = G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$. La idea es elegir $b(s_t)$ de modo que se aproxime a lo que cabría esperar de $G_t$ en ese estado, para que la diferencia $G_t - b(s_t)$ sea pequeña en magnitud. Cuanto más se aproxime $b(s_t)$ al valor típico de $G_t$ dado $s_t$, menor será la varianza del producto.

##### La línea de base óptima (control por valores)

Se puede demostrar (aunque la demostración es algo técnica) que la línea de base que minimiza la varianza es la esperanza de $G_t$ ponderada por el cuadrado del gradiente. En la práctica, esa cantidad no es conocida, pero una excelente aproximación es utilizar el **valor del estado** $v_\pi(s_t)$, es decir, el retorno esperado desde $s_t$ siguiendo la política actual. La diferencia $G_t - v_\pi(s_t)$ se denomina **ventaja** (advantage): mide cuánto mejor o peor ha sido el resultado real en comparación con lo que se esperaba normalmente en ese estado.

Si $G_t$ es mucho mayor que $v_\pi(s_t)$, la ventaja es positiva y grande, lo que indica que la acción tomada fue excepcionalmente buena; entonces la actualización empujará con fuerza los parámetros para hacer más probable esa acción. Si $G_t$ es menor que $v_\pi(s_t)$, la ventaja es negativa, y la acción se hará menos probable. Si $G_t$ es igual a lo esperado, la ventaja es cero y la acción no se modifica (aunque podría modificarse por otras muestras). Esto tiene mucho sentido intuitivo: no queremos reforzar una acción que ha dado un resultado normal; solo queremos reforzar las que han dado resultados sorprendentemente buenos, y castigar las que han dado resultados sorprendentemente malos.

##### Implementación práctica: necesitamos aprender $v_\pi(s)$

El problema es que no conocemos $v_\pi(s)$; es justamente la función de valor que queremos estimar. Por tanto, para usar $v_\pi(s)$ como línea de base, debemos **aprenderla** a partir de la experiencia. Esto añade un segundo componente al algoritmo: además de la política (el actor), necesitamos una **función de valor** (el crítico) que nos proporcione una estimación $\hat{v}(s,w)$ con parámetros $w$. Esta función se puede aprender mediante los métodos de predicción que ya conocemos: por ejemplo, usando Monte Carlo (con los mismos retornos $G_t$ que ya hemos calculado) o mediante diferencias temporales.

El algoritmo REINFORCE con línea de base queda entonces:

- Inicializamos los parámetros de la política $\theta$ y los parámetros de la función de valor $w$.
- Para cada episodio:
  - Generamos una trayectoria completa $s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T$.
  - Calculamos los retornos $G_t$ para cada paso (recorriendo hacia atrás).
  - Para cada paso $t$ desde $0$ hasta $T-1$:
    - Actualizamos el crítico: $w \leftarrow w + \alpha_w (G_t - \hat{v}(s_t,w)) \nabla_w \hat{v}(s_t,w)$. Esta es la regla de actualización de Monte Carlo para la función de valor, que ya estudiamos en capítulos anteriores.
    - Calculamos la ventaja: $A_t = G_t - \hat{v}(s_t,w)$ (usando la estimación actualizada o la anterior, según la implementación; en la práctica suele usarse la anterior para mantener la estabilidad).
    - Actualizamos el actor: $\theta \leftarrow \theta + \alpha_\theta A_t \nabla_\theta \log \pi_\theta(a_t|s_t)$.

Observemos que ahora tenemos dos tasas de aprendizaje diferentes, $\alpha_w$ y $\alpha_\theta$, que pueden ajustarse por separado. El crítico aprende a predecir el valor de los estados, y el actor utiliza esa predicción para reducir la varianza de su actualización.

La clave es que el crítico actúa como un **control de referencia** (baseline). Al restar la estimación del valor del estado, estamos eliminando la parte de $G_t$ que es predecible a partir del propio estado. Lo que queda, $G_t - \hat{v}(s_t)$, es esencialmente el componente no predecible, que contiene la información específica sobre la calidad de la acción tomada. Este componente suele tener una magnitud mucho menor que $G_t$ y, por tanto, una varianza mucho más baja. La convergencia del actor se acelera notablemente.

Además, el crítico se puede aprender de forma incremental, incluso usando diferencias temporales (TD) en lugar de retornos completos, lo que permite actualizaciones paso a paso y no solo al final del episodio. Esto nos lleva directamente a los métodos actor-crítico, que veremos a continuación.

En resumen, la línea de base no es un simple truco: tiene una sólida justificación estadística (reduce la varianza sin introducir sesgo) y una interpretación natural (la ventaja). Aprender la línea de base como una función de valor añade un componente adicional al algoritmo, pero la recompensa en términos de eficiencia de aprendizaje suele ser muy grande. Por eso, REINFORCE con línea de base es casi siempre preferible al REINFORCE básico, y sienta las bases para los métodos actor-crítico más avanzados.

### El algoritmo Actor‑Crítico

Hasta ahora hemos visto dos formas de aprovechar una función de valor para reducir la varianza del gradiente de política. En REINFORCE con línea de base, utilizábamos el retorno completo $G_t$ y restábamos una estimación del valor del estado $v(s_t)$. Aunque esto funcionaba bien, seguía siendo un método de Monte Carlo: había que esperar a que terminara el episodio para calcular todos los $G_t$ y luego actualizar. En muchas tareas, los episodios pueden ser muy largos o ni siquiera existir (tareas continuas). Además, el retorno $G_t$ todavía tiene cierta varianza, porque acumula muchas recompensas futuras.

La idea del **actor‑crítico** es dar un paso más: en lugar de esperar al final del episodio, vamos a **actualizar después de cada transición**, utilizando solo la información disponible en ese momento. Para ello, sustituimos el retorno completo $G_t$ por una estimación basada en la recompensa inmediata más el valor estimado del siguiente estado. Es decir, utilizamos el **error de diferencia temporal** (TD error) que ya conocemos de los métodos de predicción:

$$
\delta_t = r_{t+1} + \gamma \, v(s_{t+1}) - v(s_t).
$$

Este $\delta_t$ tiene una interpretación muy clara: mide la sorpresa o la discrepancia entre lo que el crítico esperaba ($v(s_t)$) y la nueva evidencia ($r_{t+1} + \gamma v(s_{t+1})$). Si $\delta_t$ es positivo, significa que la transición ha resultado mejor de lo esperado; si es negativo, peor.

Recordemos que en REINFORCE con línea de base, la ventaja se definía como $A_t = G_t - v(s_t)$. En el actor‑crítico, definimos una **ventaja instantánea** como $\delta_t$. ¿Por qué es válido este cambio? Observemos que, en expectativa, el error de TD es una estimación de la ventaja. De hecho, se puede demostrar que $\mathbb{E}[\delta_t | s_t, a_t] = Q(s_t, a_t) - v(s_t)$, que es precisamente la ventaja. Al usar una sola muestra de $\delta_t$ en lugar de $G_t - v(s_t)$, estamos haciendo un bootstrap: reemplazamos el futuro incierto por la estimación actual del crítico. Esto reduce drásticamente la varianza, porque $\delta_t$ solo involucra una recompensa inmediata y una estimación, en lugar de una suma de muchas recompensas futuras. El precio que se paga es la introducción de un posible **sesgo**: si la estimación $v(s)$ no es perfecta, la dirección de la actualización puede no ser exactamente la del gradiente verdadero. Sin embargo, en la práctica, con buenas aproximaciones y tasas de aprendizaje adecuadas, el sesgo es manejable y la reducción de varianza compensa con creces.

#### El esquema general del actor‑crítico

Un algoritmo actor‑crítico mantiene dos componentes que aprenden simultáneamente:

- **El actor**: la política parametrizada $\pi_\theta(a|s)$. Su objetivo es maximizar la recompensa esperada. Se actualiza en la dirección sugerida por el crítico.
- **El crítico**: la función de valor $v_w(s)$ (parámetros $w$). Su objetivo es estimar correctamente el valor de los estados para que el error de TD sea pequeño.

La interacción es la siguiente:

1. En cada paso $t$, el agente observa el estado $s_t$, selecciona una acción $a_t$ según la política actual $\pi_\theta$, la ejecuta y recibe $r_{t+1}$ y $s_{t+1}$.
2. El crítico calcula su error: $\delta_t = r_{t+1} + \gamma v_w(s_{t+1}) - v_w(s_t)$.
3. El crítico se actualiza para reducir ese error, por ejemplo mediante la regla de TD(0): $w \leftarrow w + \alpha_w \delta_t \nabla_w v_w(s_t)$.
4. El actor se actualiza utilizando $\delta_t$ como una estimación de la ventaja: $\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)$.
5. Se repite el proceso con el nuevo estado $s_{t+1}$.

Observemos que todo ocurre en el mismo paso; no es necesario esperar al final del episodio. Si el episodio termina, entonces $v_w(s_{t+1})$ se toma como 0 (valor del estado terminal) y el error se calcula igualmente. Esto hace que el actor‑crítico sea aplicable tanto a tareas episódicas como a tareas continuas.

El actor‑crítico presenta varias ventajas sobre REINFORCE con línea de base. La primera y más evidente es que **aprende en tiempo real**. No necesita almacenar trayectorias completas ni esperar a que termine el episodio para actualizar. Esto es crucial en entornos donde los episodios son muy largos o donde el agente debe adaptarse sobre la marcha. La segunda ventaja es que **reduce la varianza** aún más, porque $\delta_t$ solo depende de una recompensa y una estimación, en lugar de sumar muchas recompensas futuras que pueden ser muy variables. La tercera ventaja es que el crítico se actualiza con el mismo error de TD, lo que permite aprovechar toda la potencia de los métodos de diferencia temporal, que ya sabemos que son más eficientes que Monte Carlo en muchos problemas.

Por supuesto, no todo son ventajas. La principal desventaja es la introducción de **sesgo**. Si el crítico aprende lentamente o se queda atrapado en una mala aproximación, el actor puede recibir señales engañosas y converger a una política subóptima. Por eso es importante elegir bien las parametrizaciones, las tasas de aprendizaje y, en ocasiones, usar trucos como mantener una copia del crítico congelada (target network) para estabilizar el entrenamiento. En la práctica, sin embargo, el actor‑crítico funciona muy bien y es la base de muchos algoritmos modernos como A2C, A3C, PPO y SAC.

#### Ejemplo de uso con el agente saltarín

Para fijar ideas, apliquemos la idea al agente saltarín, pero de forma simplificada. Supongamos que solo nos interesa el estado 0 y que hemos aprendido una buena función de valor para ese estado. Inicialmente, $v(0)$ podría ser, por ejemplo, 0.5 (un valor arbitrario). El actor es una sigmoide con parámetro $\theta$. En un paso concreto:

- Estado $s_t = 0$. El actor elige saltar (por ejemplo, con $\theta=0.5$, probabilidad $\approx 0.62$).
- El entorno transita: supongamos que va a 2 con recompensa $-1$ (no terminal).
- El crítico estima $v(0) = 0.5$ y $v(2)$ (necesitamos también una estimación para el estado 2, digamos $v(2)=1.0$).
- Calculamos $\delta = r + \gamma v(2) - v(0) = -1 + 0.9 \times 1.0 - 0.5 = -1 + 0.9 - 0.5 = -0.6$.
- Actualizamos el crítico: por ejemplo, si $v(0)$ se parametriza linealmente con un solo peso $w$, la actualización sería $w \leftarrow w + \alpha_w (-0.6) \times 1$.
- Actualizamos el actor: $\nabla_\theta \log \pi(\text{saltar}) = 1 - \pi(\text{saltar}) = 0.38$ (aproximadamente). Entonces $\theta \leftarrow \theta + \alpha_\theta (-0.6) \times 0.38$. Como $\delta$ es negativo, $\theta$ disminuye, haciendo ligeramente menos probable saltar en el futuro.

Si, en cambio, la transición hubiera llevado directamente a la meta con recompensa $+19$, entonces $\delta$ sería positivo y el actor aumentaría la probabilidad de saltar. Así, paso a paso, el actor va ajustando su comportamiento en función de la retroalimentación inmediata del crítico, sin necesidad de esperar a que termine el episodio.



> Es importante notar que el actor‑crítico no invalida ni reemplaza a REINFORCE con línea de base. Son dos enfoques complementarios. REINFORCE con línea de base es más adecuado cuando los episodios son relativamente cortos y se puede permitir esperar a que terminen, porque su estimación del gradiente es insesgada. El actor‑crítico es más adecuado para tareas continuas o de episodios muy largos, o cuando se necesita un aprendizaje muy rápido, a costa de introducir un pequeño sesgo. En la práctica, los algoritmos más exitosos suelen ser variantes del actor‑crítico con múltiples mejoras (entropía, ventaja generalizada, etc.).
>
