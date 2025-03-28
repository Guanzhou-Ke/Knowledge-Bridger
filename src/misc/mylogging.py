import logging

# ANSI 转义序列
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # 蓝色
        'INFO': '\033[92m',     # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',    # 红色
        'CRITICAL': '\033[41m', # 红色背景
    }
    RESET = '\033[0m'  # 重置颜色

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{level_color}{message}{self.RESET}"

def get_logger():
    logger = logging.getLogger("ColoredLogger")
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

