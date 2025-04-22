from pygame.math import clamp
from pyglm import glm
from application.Puck import Puck
from application.mallet import Mallet
from networking import NetworkClient
from utils import logger
import pygame


class Application:
    __screen_width = 540
    __screen_height = 960

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.__screen_width, self.__screen_height)
        )
        self.clock = pygame.time.Clock()
        self.delta_time = 0.1
        # self.network_client = NetworkClient(uri="ws://localhost:8765")
        self.network_client = NetworkClient(
            uri="wss://splendid-wistful-umbrella.glitch.me")
        self.running = True

        self.puck = Puck()
        self.puck.pos.x = self.__screen_width // 2
        self.puck.pos.y = self.__screen_height * 2 // 3

        self.mallet = Mallet()

    def __del__(self) -> None:
        logger.client_logger.info("Application Closed!")
        self.network_client.close()
        pygame.quit()

    def run(self) -> None:
        logger.client_logger.info("Application Started!")
        while (self.running):
            self.screen.fill((14, 14, 14))

            pygame.draw.line(self.screen, (255, 255, 255), (0, self.__screen_height//2),
                             (self.__screen_width, self.__screen_height//2))

            x, y = pygame.mouse.get_pos()
            self.mallet.pos.x = pygame.math.clamp(x, self.mallet.radius,
                                                  self.__screen_width - self.mallet.radius)
            self.mallet.pos.y = clamp(y, self.__screen_height//2 + self.mallet.radius,
                                      self.__screen_height - self.mallet.radius)

            self.mallet.draw(self.screen)

            # send position to the server
            self.network_client.send_mallet_position(
                self.mallet.pos.x, self.mallet.pos.y)

            # get mallet velocity
            self.mallet.update_velocity(self.mallet.pos.x, self.mallet.pos.y)
            logger.client_logger.debug(self.mallet.velocity)

            # get the position of the puck from the server (physics will be calculated on server)
            ITERATIONS = 100
            for _ in range(ITERATIONS):
                self.physics()
                self.puck.pos.x += self.puck.velocity.x/ITERATIONS
                self.puck.pos.y += self.puck.velocity.y/ITERATIONS

                if (self.puck.pos.x < self.puck.radius or self.puck.pos.x > self.__screen_width - self.puck.radius):
                    self.puck.velocity.x *= -1
                if (self.puck.pos.y < self.puck.radius or self.puck.pos.y > self.__screen_height - self.puck.radius):
                    self.puck.velocity.y *= -1

            # self.puck.velocity *= 0.995

            # draw the puck
            self.puck.draw(self.screen)

            # get the position of the mallet of opponent
            opp = self.network_client.get_opponent_mallet_pos()
            # draw the mallet of opponent
            if opp != None:
                [oppx, oppy] = opp
                oppx = self.__screen_width - oppx
                oppy = 2*self.__screen_height//2 - oppy
                # oppx = pygame.math.clamp(oppx, self.mallet.radius,
                #                          self.__screen_width - self.mallet.radius)
                # oppy = clamp(oppx, self.mallet.radius,
                #              self.__screen_height//2 - self.mallet.radius)
                pygame.draw.circle(self.screen, self.mallet.color,
                                   (oppx, oppy), self.mallet.radius)

            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    self.running = False

            self.delta_time = self.clock.tick(60) / 1000
            self.delta_time = clamp(self.delta_time, 0.001, 0.1)
            pygame.display.flip()

    def physics(self) -> None:
        d = glm.distance(self.mallet.pos, self.puck.pos)
        if (d < self.mallet.radius + self.puck.radius):
            logger.client_logger.info("Collision!")

            # Calculate velocity for puck
            m1 = self.puck.mass
            m2 = self.mallet.mass
            v1 = self.puck.velocity
            v2 = self.mallet.velocity
            x1 = self.puck.pos
            x2 = self.mallet.pos

            overlap = d - (self.mallet.radius + self.puck.radius)
            dir = (x2 - x1) * (overlap * 0.5) / glm.length(x2 - x1)
            self.puck.pos += dir
            self.puck.pos.x = clamp(
                self.puck.pos.x, self.puck.radius, self.__screen_width - self.puck.radius)
            self.puck.pos.y = clamp(
                self.puck.pos.y, self.puck.radius, self.__screen_height - self.puck.radius)

            self.puck.velocity += ((2*m2)/(m1 + m2)) * (glm.dot(v2 - v1,
                                                                x2 - x1) / glm.length2((x2 - x1))) * (x2 - x1)

            self.puck.velocity.x = clamp(self.puck.velocity.x, -100, 100)
            self.puck.velocity.y = clamp(self.puck.velocity.y, -100, 100)
