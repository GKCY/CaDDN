from json.tool import main
from turtle import forward
from unicodedata import name
import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils

class SingleBoxesOverlapNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        #input (m+n) * 7
        m = input.shape[0] / 3
        n = input.shape[0] - n
        boxes_a = input[0:m]
        boxes_b = input[-n:]
        ans = iou3d_nms_utils.boxes_overlap_gpu(boxes_a, boxes_b)
        return ans

def main():
    net = SingleBoxesOverlapNet();
    t = torch.randn(10, 7)
    print(net(t))

if __name__ == '__main__':
    main()