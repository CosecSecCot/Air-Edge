from application import Application
from utils.logger import client_logger


def main():
    client_logger.debug("WELCOME TO AIR EDGE!")

    app = Application()
    app.run()


if __name__ == '__main__':
    main()
