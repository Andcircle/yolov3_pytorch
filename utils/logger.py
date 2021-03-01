import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from tensorboardcolab import TensorBoardColab


class Logger(object):
    def __init__(self, log_dir, log_hist=True, colab=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)
        self.colab = colab
        if self.colab:
            self.colab_writer = TensorBoardColab()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        if self.colab:
            self.colab_writer.save_value("Graph",tag,step,value)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
            if self.colab:
                self.colab_writer.save_value("Graph",tag,step,value)

    def image_summary(self, tag, image, step):
        """Log image variables."""
        self.writer.add_image(tag, image, step, dataformats='HWC')
        if self.colab:
            self.colab_writer.save_value(tag,image)
