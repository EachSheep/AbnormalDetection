import logging
import os

def set_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='a') # 输出到文件
    fh.setLevel(logging.INFO) 
    ch = logging.StreamHandler() # 输出到控制台
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger