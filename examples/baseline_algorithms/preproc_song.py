import os
import shutil
import sys
import numpy as np
from scipy import sparse
import pandas as pd

DATA_DIR = '/home/xiaojie/Downloads/Webscope_R3'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')
if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

train_txt = os.path.join(DATA_DIR, 'ydata-ymusic-rating-study-v1_0-train.txt')
test_txt = os.path.join(DATA_DIR, 'ydata-ymusic-rating-study-v1_0-test.txt')
train_tmp = os.path.join(pro_dir, 'train.tmp')
test_tmp = os.path.join(pro_dir, 'test.tmp')

def txt_to_tmp(txt_file, tmp_file):
    if os.path.isfile(tmp_file):
        return
    with open(txt_file) as fin, open(tmp_file, 'w') as fout:
        fout.write('userId,movieId,rating\n')
        for line in fin.readlines():
            fields = line.split()
            fout.write('%s\n' % (','.join(fields)))

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def numerize(tp):
    uid = [profile2id[x] for x in tp['userId']]
    sid = [show2id[x] for x in tp['movieId']]
    return pd.DataFrame(
            data={
                'uid': uid,
                'sid': sid,
                'rating': tp['rating'],
            },
            columns=[
                'uid',
                'sid',
                'rating',
            ]
    )

txt_to_tmp(train_txt, train_tmp)
txt_to_tmp(test_txt, test_tmp)

tr_data = pd.read_csv(train_tmp, header=0)
te_data = pd.read_csv(test_tmp, header=0)

user_activity = get_count(tr_data, 'userId')
item_popularity = get_count(tr_data, 'movieId') 

unique_uid = user_activity.index
unique_sid = pd.unique(tr_data['movieId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

te_users = pd.unique(te_data['userId'])
train_plays = tr_data.loc[~tr_data['userId'].isin(te_users)]

test_plays_tr = tr_data.loc[tr_data['userId'].isin(te_users)]
test_plays_te = te_data

vad_plays_tr, vad_plays_te = test_plays_tr, test_plays_te

train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

