import logging


def get_logger(name):

    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/connect_x.log",
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    return logger
