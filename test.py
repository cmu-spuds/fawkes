from fawkes.protection import Fawkes
from keras import utils

test_img = utils.get_file(
    fname="fawkes.png",
    origin="https://i.pinimg.com/originals/07/33/ba/0733ba760b29378474dea0fdbcb97107.png",
    cache_subdir="image",
)

fwks = Fawkes("extractor_2", 1, mode="low")
fwks.run_protection([test_img], format="png")
