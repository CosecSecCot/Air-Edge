import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = '\x1b[32m'
    bold_red = "\x1b[31;1m"
    bg_red = "\x1b[41m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s [%(name)s][%(levelname)-8s] (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bg_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


server_logger = logging.getLogger("Server")
server_logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()  # using sys.stdcerr
console_handler.setLevel(logging.DEBUG)

console_handler.setFormatter(CustomFormatter())

server_logger.addHandler(console_handler)
