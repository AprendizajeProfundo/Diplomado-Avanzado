import numpy as np
from typing import Union


class ActionSelector:
    """
    Clase abstracta que convierte puntuaciones (scores) en acciones,
    para un lote scores.
    En las clases derivadas
    Se recibe una matriz de scores en donde:
    - cada fila es un ejemplo (batch de scores)
    - cada columna un score asociado al ejemplo de la columna
    es decir , los scores viene por las las filas
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selecciona acciones usando argmax.
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    """
    Selecciona acciones usando epsilon-gredy.
    - epsilon: se selecciona la acción con probabilidad epsilon
    - selector: Action selector que se ejecuta con probabilidad (1-epsilon)
    """
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        # selecciona una accion por cada fila, usando el selector de acciones
        actions = self.selector(scores)
        # con probabilidad epsilon escoge m ejemplos del batch
        # para los cuales se escogera una acción aleatoriamente
        # y se hace una máscara con los escogidos (valor=True, para los escogidos)
        mask = np.random.random(size=batch_size) < self.epsilon
        # se escogen las m acciones aleatoriamente
        rand_actions = np.random.choice(n_actions, sum(mask))
        # se sube la acción aleatoria a cada ejemplo selecionado 
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Convierte probabilidades de acciones en acciones al muestrearlas
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)
