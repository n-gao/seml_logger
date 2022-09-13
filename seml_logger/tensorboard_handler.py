from io import StringIO
from logging import StreamHandler

from tensorboardX import SummaryWriter


class TensorBoardHandler(StreamHandler):    
    def __init__(self, writer: SummaryWriter, tag='Log'):
        self.writer = writer
        self.tag = tag
        self.str_stream = StringIO()
        StreamHandler.__init__(self, self.str_stream)
    
    def emit(self, record):
        StreamHandler.emit(self, record)
        self.writer.add_text(self.tag, f'```\n{self.str_stream.getvalue()}\n```', global_step=0)
