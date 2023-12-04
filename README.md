```
import numpy as np 

# A : x1, y1, x2, y2
# B : x1, y1, x2, y2

low = np.s_[...,:2]
high = np.s_[...,2:]

# Effectively calculate IOU of 2 lists of boxes. 
def iou_for_list(A,B):
    A,B = A.copy(),B.copy()
    A[high] += 1; B[high] += 1
    intrs = (np.maximum(0,np.minimum(A[high],B[high])
                        -np.maximum(A[low],B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3]) 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

# Notes

https://github.com/hqucv/siamrn/blob/448080eeee90200d1aef2af20cf6f02a12917d5f/siamban/tracker/siamban_tracker.py#L17

https://arxiv.org/pdf/2105.03817.pdf

https://arxiv.org/abs/2003.06761



https://github.com/HonglinChu/SiamTrackers/blob/master/TrTr


Notes
+ EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention https://arxiv.org/pdf/2305.07027.pdf
+ EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf
+ Rethinking Vision Transformers for MobileNet Size and Speed: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Rethinking_Vision_Transformers_for_MobileNet_Size_and_Speed_ICCV_2023_paper.pdf
+ FLatten Transformer: Vision Transformer using Focused Linear Attention: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_FLatten_Transformer_Vision_Transformer_using_Focused_Linear_Attention_ICCV_2023_paper.pdf
