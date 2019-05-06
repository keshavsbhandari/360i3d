import pickle,os
import pandas as pd
import shutil
import torch
from itertools import chain
import tabulate





def topk(output, target , k = 1):
    _, l = torch.topk(output,k)
    l = [*chain.from_iterable(l.tolist())]
    belongs_to = (target.item() in l) * 1
    return belongs_to



def topK(output, target , k = 1):
    _, l = torch.topk(output,k)
    l = l.tolist()
    belongs_to = (target.item() in l) * 1
    return belongs_to


def kaccuracy_ontraining(output, target):
    percent = lambda x:(sum(x)/len(x)) * 100
    M1 = []
    M3 = []
    M5 = []
    for o,t in zip(output, target):
        # print(o,t)
        M1.append(topK(o, t, 1))
        M3.append(topK(o, t, 3))
        M5.append(topK(o, t, 5))
    # return 1,2,3
    return percent(M1), percent(M3), percent(M5)



def print_dict_to_table(dic, drop = None, transpose = True):
    for d in drop:
        dic.pop(d)
    if transpose:
        dic = get_dic_transpose(dic,on="Top K")
    printinfo = tabulate.tabulate(dic, headers=dic.keys(), tablefmt='github',floatfmt=".4f")
    to = len(printinfo.split('\n')[0])

    print("="*to)
    print(printinfo)
    print("=" * to)


def get_dic_transpose(dic, on):
    new_keys = ['header']
    new_keys += [*map(lambda x:on+'_'+str(x),dic.pop(on))]
    old_keys = dic.keys()
    old_values = dic.values()

    new_dic = {}
    new_dic[new_keys[0]] = list(old_keys)
    new_dic[new_keys[1]] = [*map(lambda x:x[0],old_values)]
    new_dic[new_keys[2]] = [*map(lambda x: x[1], old_values)]
    new_dic[new_keys[3]] = [*map(lambda x: x[2], old_values)]

    return new_dic

def accuracy(output, target, topk=None):
    _, o = torch.max(output, 1)
    t = target
    if topk:
        k = max(topk)
        t, o = t[:k].float(), o[:k].float()
    N = t.numel()
    correct_preds = t.eq(o).sum().item()
    return correct_preds * (100.0 / N),  correct_preds, N

def pickleme(fname,dic):
    with open(fname, 'wb') as f:
        pickle.dump(dic, f)

def append_update_dic(key,dic,val):

    if dic.get(key) is None:
        dic[key] = [val]
    else:
        dic[key].append(val)

    return dic

def get_video_accuracy(value):
    k1, k3, k5 = torch.tensor(value,dtype = torch.float).mean(0).tolist()
    return float(k1 * 100) ,float(k3 * 100) ,float(k5 * 100)

def get_mean_average_accuracy(dic):
    T = lambda x: torch.tensor(x, dtype=torch.float).mean(0)
    value = [*chain(dic.values())]
    value = [*map(T, value)]
    k1, k3, k5 = torch.stack(value).mean(0).tolist()
    return k1 * 100 ,k3 * 100 ,k5 * 100

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):
    if mode =='train':
        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))
        print(result)
        df = pd.DataFrame(data = info)
    if mode =='test':
        df = pd.DataFrame(data = info)
    if not os.path.isfile(filename):
        df.to_csv(filename,index=None)
    else:
        df.to_csv(filename,mode = 'a', header= False, index=None)


def get_mAP_and_mAR(df):
    values = df.values
    pred_sum = df.sum(axis = 1).values
    real_sum = df.sum(axis = 0).values
    precision = values / (pred_sum.reshape(-1,1) + 1e-9)
    recall = values / (real_sum + 1e-9)

    avg_recall = recall.diagonal().mean() * 100
    avg_precision = precision.diagonal().mean() * 100
    return avg_precision, avg_recall