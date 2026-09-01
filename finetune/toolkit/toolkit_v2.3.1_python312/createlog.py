import logging
import datetime
from datetime import timedelta
 
class Log:
    def __init__(self, TC_log_path) -> None:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level = logging.INFO,
            format = '[%(asctime)s] - %(levelname)-5s - %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S',
            handlers = [
            logging.FileHandler(TC_log_path),
            logging.StreamHandler()]
        )

    @staticmethod
    def normal(message):
        print(f"[INFO] - {message}")
        logging.info(message)

    @staticmethod
    def warning(message):
        print(f"[WARN] - {message}")
        logging.warning(message)

    @staticmethod
    def error(message):
        print(f"[ERROR] - {message}")
        logging.error(message)

    @staticmethod
    def test_result(message, *args, **kws):
        """
        message must include "### Pass ###" or "### Fail ###" at the first place
        """
        pass_debug_level = 45
        logging.addLevelName(pass_debug_level, "Test Result")
        if logging.getLogger(__name__).isEnabledFor(pass_debug_level):
            logging.getLogger(__name__)._log(pass_debug_level, message, args, **kws)
            print(f"[Test Result] - {message}")

    @staticmethod
    def expect_finish_time(hour, *args, **kws):
        """
        each parameters must be int
        """
        pass_debug_level = 45
        logging.addLevelName(pass_debug_level, "##### Test case expect finish time")
        if logging.getLogger(__name__).isEnabledFor(pass_debug_level):
            finish_time = datetime.datetime.now() + timedelta(hours=hour)
            logging.getLogger(__name__)._log(pass_debug_level, finish_time.strftime("%Y-%m-%d %H:%M") + " #####", args, **kws)


if __name__ == "__main__":
    pass