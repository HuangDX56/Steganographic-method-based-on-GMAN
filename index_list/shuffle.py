# 将数据随机分为训练集和测试集
import numpy as np
import random
'''
# BossBase
list = list(range(1,10001))
random.shuffle(list)
list = np.array(list)
#print(len(list))

train_list = list[:4000]
train_list = np.array(train_list)
np.save('bossbase_train_index.npy', train_list)
#print(len(train_list))

valid_list = list[4000:5000]
valid_list = np.array(valid_list)
np.save('bossbase_valid_index.npy', valid_list)
#print(len(valid_list))

test_list = list[5000:]
test_list = np.array(test_list)
np.save('bossbase_test_index.npy', test_list)
#print(len(test_list))

'''

test_list = list(range(1,10001))
random.shuffle(test_list)
test_list = np.array(test_list)
print(test_list)
print(len(test_list))
np.save('boss_gan_train_1w.npy', test_list)
#print(len(test_list))



print('split down')

