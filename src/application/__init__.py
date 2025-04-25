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
            (self.__screen_width, self.__screen_height),
            # pygame.FULLSCREEN,
        )

        # ── game state & UI──
        self.state = "waiting"            # waiting, countdown, playing, end
        self.timer = 0                    # seconds left
        self.countdown = 0                # countdown seconds
        self.score_us = 0
        self.score_opponent = 0
        self.message = ""                 # server-sent waiting message

        # fonts for UI
        self.font = pygame.font.SysFont(None, 72)    # large
        self.small_font = pygame.font.SysFont(None, 36)  # small
        # ready-button
        self.button_text = "READY"
        w, h = 200, 60
        self.ready_button = pygame.Rect(
            (self.__screen_width - w)//2,
            (self.__screen_height)//2 + 50,
            w, h
        )

        self.clock = pygame.time.Clock()
        self.delta_time = 0.1
        # self.network_client = NetworkClient(uri="ws://localhost:8765")
        self.network_client = NetworkClient(
            uri="wss://splendid-wistful-umbrella.glitch.me")
        self.running = True

        self.puck = Puck()
        self.mallet = Mallet()

    def __del__(self) -> None:
        logger.client_logger.info("Application Closed!")
        self.network_client.close()
        pygame.quit()

    def run(self) -> None:
        logger.client_logger.info("Application Started!")
        while (self.running):
            self.screen.fill((14, 14, 14))

            # ── poll server state ──
            state = self.network_client.get_server_state()
            ox, oy = None, None
            if state:
                logger.client_logger.warning(
                    f"Client ID: {self.network_client.client_id}")
                # update local UI state
                self.state = state["state"]
                self.timer = state.get("timer", self.timer)
                self.countdown = state.get("countdown", 0)
                sq = state.get("score", [0, 0])
                self.score_us = 0
                self.score_opponent = 0
                # if self.client_id is None and state.get("client_id") is not None:
                #     self.client_id = state["client_id"]
                try:
                    # self.score_us, self.score_opponent = sq[0], sq[1]
                    if self.network_client.client_id is not None:
                        self.score_us = sq[self.network_client.client_id]
                        self.score_opponent = sq[self.network_client.client_id ^ 1]
                except Exception as e:
                    logger.client_logger.error(f"Error Getting Score! {e}")
                # waiting-screen message (disconnect / waiting)
                self.message = state.get("message", "")
                # dynamic button text
                btns = state.get("buttons", [])
                # assume client index = 0 if assigned first, else 1
                if btns and self.network_client.client_id is not None:
                    # clients sent in connect order, we approximate index by id parity
                    # idx = 0 if self.network_client.client_id == min(
                    #     btns, default=self.network_client.client_id) else 1
                    self.button_text = btns[self.network_client.client_id]
                # puck position (normalized → pixels)
                px, py = state["puck"]["x"], state["puck"]["y"]
                self.puck.pos.x = px * self.__screen_width
                self.puck.pos.y = py * self.__screen_height
                # opponent normalized
                try:
                    if self.network_client.client_id is not None:
                        opp = state["mallets"][self.network_client.client_id ^ 1]
                        ox = opp["x"] * self.__screen_width
                        oy = opp["y"] * self.__screen_height
                        # ox = (1 - opp["x"]) * self.__screen_width
                        # oy = (1 - opp["y"]) * self.__screen_height
                except Exception as e:
                    logger.client_logger.error(
                        f"Error Getting Opponent Mallet! {e}")
                logger.client_logger.warning(
                    f"{self.score_us}, {self.score_opponent}")
            else:
                logger.client_logger.warning(
                    "No Game State Yet!")
                continue

            pygame.draw.line(self.screen, (255, 255, 255), (0, self.__screen_height//2),
                             (self.__screen_width, self.__screen_height//2))

            # x, y = pygame.mouse.get_pos()
            # self.mallet.pos.x = pygame.math.clamp(x, self.mallet.radius,
            #                                       self.__screen_width - self.mallet.radius)
            # self.mallet.pos.y = clamp(y, self.__screen_height//2 + self.mallet.radius,
            #                           self.__screen_height - self.mallet.radius)
            #
            # self.mallet.draw(self.screen)
            #
            # # send position to the server
            # self.network_client.send_mallet_position(
            #     self.mallet.pos.x, self.mallet.pos.y)

            # # get mallet velocity
            # self.mallet.update_velocity(self.mallet.pos.x, self.mallet.pos.y)
            # logger.client_logger.debug(self.mallet.velocity)
            #
            # # get the position of the puck from the server (physics will be calculated on server)
            # ITERATIONS = 100
            # for _ in range(ITERATIONS):
            #     self.physics()
            #     self.puck.pos.x += self.puck.velocity.x/ITERATIONS
            #     self.puck.pos.y += self.puck.velocity.y/ITERATIONS
            #
            #     if (self.puck.pos.x < self.puck.radius or self.puck.pos.x > self.__screen_width - self.puck.radius):
            #         self.puck.velocity.x *= -1
            #     if (self.puck.pos.y < self.puck.radius or self.puck.pos.y > self.__screen_height - self.puck.radius):
            #         self.puck.velocity.y *= -1
            #
            # # self.puck.velocity *= 0.995
            #
            # # draw the puck
            # self.puck.draw(self.screen)
            #
            # # get the position of the mallet of opponent
            # opp = self.network_client.get_opponent_mallet_pos()
            # # draw the mallet of opponent
            # if opp != None:
            #     [oppx, oppy] = opp
            #     oppx = self.__screen_width - oppx
            #     oppy = 2*self.__screen_height//2 - oppy
            #     # oppx = pygame.math.clamp(oppx, self.mallet.radius,
            #     #                          self.__screen_width - self.mallet.radius)
            #     # oppy = clamp(oppx, self.mallet.radius,
            #     #              self.__screen_height//2 - self.mallet.radius)
            #     pygame.draw.circle(self.screen, self.mallet.color,
            #                        (oppx, oppy), self.mallet.radius)

            # ── render by state ──
            if self.state == "waiting":
                # # title
                # txt = self.font.render("Waiting...", True, (255, 255, 255))  #

                # title
                # :contentReference[oaicite:0]{index=0}
                txt = self.font.render(
                    self.message or "Waiting...", True, (255, 255, 255))
                rect = txt.get_rect(
                    center=(self.__screen_width//2, self.__screen_height//2 - 50))
                self.screen.blit(txt, rect)
                # ready button
                pygame.draw.rect(self.screen, (100, 100, 100),
                                 self.ready_button)  #
                # lab = self.small_font.render("READY", True, (255, 255, 255))  #
                # :contentReference[oaicite:1]{index=1}
                lab = self.small_font.render(
                    self.button_text, True, (255, 255, 255))
                lrect = lab.get_rect(center=self.ready_button.center)
                self.screen.blit(lab, lrect)

            elif self.state == "countdown":
                num = str(self.timer)
                txt = self.font.render(num, True, (255, 255, 255))  #
                self.screen.blit(txt, txt.get_rect(
                    center=(self.__screen_width//2, self.__screen_height//2)))

            elif self.state == "playing":
                x, y = pygame.mouse.get_pos()
                if self.network_client.client_id == 0:
                    self.mallet.pos.x = pygame.math.clamp(x, self.mallet.radius,
                                                          self.__screen_width - self.mallet.radius)
                    self.mallet.pos.y = clamp(y, self.__screen_height//2 + self.mallet.radius,
                                              self.__screen_height - self.mallet.radius)
                elif self.network_client.client_id == 1:
                    self.mallet.pos.x = pygame.math.clamp(x, self.mallet.radius,
                                                          self.__screen_width - self.mallet.radius)
                    self.mallet.pos.y = clamp(y, self.mallet.radius,
                                              self.__screen_height//2 - self.mallet.radius)

                self.mallet.draw(self.screen)

                # send position to the server
                self.network_client.send_mallet_position(
                    self.mallet.pos.x / self.__screen_width, self.mallet.pos.y / self.__screen_height)

                # # get mallet velocity
                # self.mallet.update_velocity(self.mallet.pos.x, self.mallet.pos.y)
                # logger.client_logger.debug(self.mallet.velocity)

                # draw puck only (physics on server)
                self.puck.draw(self.screen)
                # draw opponent mallet
                # if ox is not None and oy is not None:

                if ox is not None and oy is not None:
                    pygame.draw.circle(
                        self.screen, (0, 0, 255), (ox, oy), self.mallet.radius)  #

                # draw scores & timer
                score_txt = self.small_font.render(
                    f"{self.score_us} : {self.score_opponent}", True, (255, 255, 255))  #
                self.screen.blit(score_txt, (20, 20))
                time_txt = self.small_font.render(
                    f"Time: {self.timer}", True, (255, 255, 255))  #
                self.screen.blit(time_txt, (self.__screen_width-160, 20))

            elif self.state == "result":
                # show final result for result countdown
                if self.score_us > self.score_opponent:
                    res = "You Win!"
                elif self.score_us < self.score_opponent:
                    res = "You Lose"
                else:
                    res = "Tie"
                # :contentReference[oaicite:2]{index=2}
                txt = self.font.render(res, True, (255, 255, 255))
                self.screen.blit(txt, txt.get_rect(
                    center=(self.__screen_width//2, self.__screen_height//2 - 50)))
                # show result timer
                t2 = self.small_font.render(
                    f"{self.timer}", True, (255, 255, 255))
                self.screen.blit(t2, t2.get_rect(
                    center=(self.__screen_width//2, self.__screen_height//2 + 20)))
                # play again button
                pygame.draw.rect(self.screen, (100, 100, 100),
                                 self.ready_button)  #
                # :contentReference[oaicite:3]{index=3}
                lab = self.small_font.render(
                    self.button_text, True, (255, 255, 255))
                self.screen.blit(lab, lab.get_rect(
                    center=self.ready_button.center))

            else:  # end
                # result
                if self.score_us > self.score_opponent:
                    res = "You Win!"
                elif self.score_us < self.score_opponent:
                    res = "You Lose"
                else:
                    res = "Tie"
                txt = self.font.render(res, True, (255, 255, 255))  #
                self.screen.blit(txt, txt.get_rect(
                    center=(self.__screen_width//2, self.__screen_height//2 - 50)))
                # replay button
                pygame.draw.rect(self.screen, (100, 100, 100),
                                 self.ready_button)  #
                lab = self.small_font.render(
                    "PLAY AGAIN", True, (255, 255, 255))  #
                self.screen.blit(lab, lab.get_rect(
                    center=self.ready_button.center))

            # ── event handling ──
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    # :contentReference[oaicite:0]{index=0}
                    pos = pygame.mouse.get_pos()
                    # in waiting or end, clicking ready
                    # :contentReference[oaicite:1]{index=1}
                    # if self.ready_button.collidepoint(pos) and self.state in ("waiting", "end"):
                    if self.ready_button.collidepoint(pos) and self.state in ("waiting", "result"):
                        self.network_client.send_ready()

            self.delta_time = self.clock.tick(30) / 1000
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
