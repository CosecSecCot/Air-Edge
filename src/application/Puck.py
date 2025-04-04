import pygame
from pyglm import glm


class Puck:
    def __init__(self) -> None:
        self.radius: float = 30
        self.pos = glm.vec2(self.radius, self.radius)
        self.color = (0, 255, 0)
        self.velocity = glm.vec2(1, 1)
        self.mass = 5.0

    def draw(self, screen: pygame.Surface) -> None:
        """Draws the puck on the screen"""
        pygame.draw.circle(screen, self.color,
                           (self.pos.x, self.pos.y), self.radius)
