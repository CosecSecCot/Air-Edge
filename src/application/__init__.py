from pygame.math import clamp
from utils import logger
import pygame


class Application:
    __screen_width = 540
    __screen_height = 960

    running = True
    screen: pygame.Surface

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.__screen_width, self.__screen_height)
        )

    def __del__(self) -> None:
        logger.client_logger.info("Application Closed!")
        pygame.quit()

    def run(self) -> None:
        logger.client_logger.info("Application Started!")
        radius = 50
        while (self.running):
            self.screen.fill((14, 14, 14))

            pygame.draw.line(self.screen, (255, 255, 255), (0, self.__screen_height//2),
                             (self.__screen_width, self.__screen_height//2))

            x, y = pygame.mouse.get_pos()
            x = clamp(x, radius, self.__screen_width - radius)
            y = clamp(y, self.__screen_height//2 + radius,
                      self.__screen_height - radius)

            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), radius)

            # send position to the server
            # get the position of the puck from the server (physics will be calculated on server)
            # draw the puck
            # get the position of the mallet of opponent
            # draw the mallet of opponent

            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    self.running = False

            pygame.display.flip()
