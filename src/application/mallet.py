import pygame

from pyglm import glm


class Mallet:
    def __init__(self) -> None:
        self.radius: float = 50
        self.pos = glm.vec2(self.radius, self.radius)
        self.color = (255, 0, 0)
        self._last_position = glm.vec2(0, 0)
        self.velocity = glm.vec2(0, 0)
        self.mass = 10.0

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, self.color,
                           (self.pos.x, self.pos.y), self.radius)

    def update_velocity(self, xpos: float, ypos: float) -> glm.vec2:
        self.velocity.x = xpos - self._last_position.x
        self.velocity.y = ypos - self._last_position.y
        self._last_position.x = xpos
        self._last_position.y = ypos
        return self.velocity
