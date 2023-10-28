import numpy as np

test = np.load('bossbase_test_index.npy')
print(test.shape, test)

train = np.load('bossbase_train_index.npy')
print(train.shape, train)

valid = np.load('bossbase_valid_index.npy')
print(valid.shape, valid)

evaluate = np.load('boss_gan_train_1w.npy')
print(evaluate.shape, evaluate)

