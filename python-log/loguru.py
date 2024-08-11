from loguru import logger
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger():
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

setup_logger()

# 现在你可以使用logging记录日志，它会被重定向到loguru
logging.info("This is an info message")
logging.warning("This is a warning message")


from loguru import logger

# 添加一个文件处理器，设置日志切片为每天，并保留最近3天的日志
logger.add("file_{time}.log", rotation="1 day", retention="3 days")

# 以下是基于配置的方法
# loguru_config.toml
[loguru]
handlers=file

[loguru.handlers.file]
level=DEBUG
path=/path/to/log/file.log
rotation=1 day
retention=3 days


import loguru
from loguru import logger

# 加载配置文件
logger.configure(handlers=[{"sink": "/path/to/log/file.log"}], extra={"file_path": "/path/to/log/loguru_config.toml"})