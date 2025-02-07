
import torch
from torch.utils.tensorboard import SummaryWriter
import scipy.io
from functions import stats
import os



def add_loop(writer, title,list):
    # for epoch in range(len(list)):
    for epoch in range(11,20):
        writer.add_scalar(title, list[epoch], epoch)
    return writer

def resume_writer(writer,data_dir):
    checkpoint_path=os.path.join(data_dir,'checkpoint.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    start_epoch=checkpoint['epoch']
    stats_ = stats(data_dir,start_epoch)
    writer = add_loop(writer,'Loss/val',stats_.valObj)
    writer = add_loop(writer, 'top1/val', stats_.valTop1)
    writer = add_loop(writer, 'top5/val', stats_.valTop5)
    writer = add_loop(writer, 'Loss/train', stats_.trainObj)
    writer = add_loop(writer, 'top1/train', stats_.trainTop1)
    writer = add_loop(writer, 'top5/train', stats_.trainTop5)
    return writer


data_dir= '/home/zchen/Proposed_Methods/ConvPooling/Results/resnet50-LCMCOV-Cholesky-eps0.001-lr0.003-bs8/'
writer_path='Results/test/test_tensorboard'
writer = SummaryWriter(writer_path)
writer = resume_writer(writer,data_dir)
writer.close()


