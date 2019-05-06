from utils import *
import dataloader
import os
from flow_train_partial import FLOW_Inception
from rgb_train_partial import RGB_Inception

os.environ["CUDA_VISIBLE_DEVICES"] = "6"



motion_loader = dataloader.EgoCentricDataLoader(
    BATCH_SIZE=1,
    num_workers=8,
    img_stack=10,
    train_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_train_list.txt',
    test_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_test_list.txt',
    val_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_val_list.txt',
    mode='both',
    onlytest=True,
    # onlyval = True,
    pin_memory=False
)

motion_test = motion_loader()


MCNN = FLOW_Inception(
    nb_epochs=None,
    lr=1e-2,
    batch_size=None,
    resume='record_partial/flow/model_best.pth.tar',
    start_epoch=None,
    evaluate=None,
    train_loader=None,
    val_loader=None,
    # channel=20,
    test_loader=motion_test
    )

# MCNN = Motion_CNN(
#     nb_epochs=None,
#     lr=1e-2,
#     batch_size=None,
#     resume='record/motion_untrained/model_best.pth.tar',
#     start_epoch=None,
#     evaluate=None,
#     train_loader=None,
#     val_loader=None,
#     channel=20,
#     test_loader=motion_test
# )

# SCNN = Spatial_CNN(
#     nb_epochs=None,
#     lr=1e-2,
#     batch_size=None,
#     resume='record_spatial/spatial_untrained/model_best.pth.tar',
#     start_epoch=None,
#     evaluate=None,
#     train_loader=None,
#     test_loader=None,
#     val_loader=None,
#     channel=30
# )

SCNN = RGB_Inception(
    nb_epochs=None,
    lr=1e-2,
    batch_size=None,
    resume='record_partial/rgb/model_best.pth.tar',
    start_epoch=None,
    evaluate=None,
    train_loader=None,
    test_loader=None,
    val_loader=None,
)

T1 = []
T3 = []
T5 = []
for i, ((d1, d2), l) in enumerate(motion_test):
    break


    o1,_ = MCNN.extractProbability(d1)
    o2,_ = SCNN.extractProbability(d2)

    li = l['label'].item()

    o = (o1 + o2)/2

    top1 = o.topk(1)[1].tolist()[0]
    top3 = o.topk(3)[1].tolist()[0]
    top5 = o.topk(5)[1].tolist()[0]

    T1.append((li in top1)*1)
    T3.append((li in top3)*1)
    T5.append((li in top5)*1)

# P = lambda x: 100 * (sum(x) / len(x))
#
# print("TOP1 ACCURACY {}".format(P(T1)))
# print("TOP3 ACCURACY {}".format(P(T3)))
# print("TOP5 ACCURACY {}".format(P(T5)))
