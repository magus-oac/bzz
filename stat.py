import pandas as pd
import random
import json

test_num = 100

df = pd.read_csv('/usr/local/cv/zzb/data/train_labels.csv')
# l = df.values.tolist()
dg = df.groupby('accent')
p = [[] for i in range(3)]
for i, x in enumerate(dg):
    p[i].extend(x[1].values.tolist())

my_train = []
my_train.extend(p[0][test_num::])
my_train.extend(p[1][test_num::])
my_train.extend(p[2][test_num::])
random.shuffle(my_train)

my_test = []
my_test.extend(p[0][0:test_num])
my_test.extend(p[1][0:test_num])
my_test.extend(p[2][0:test_num])
random.shuffle(my_test)

with open('./my_train.json', 'w') as f:
    json.dump(my_train, f)

with open('./my_test.json', 'w') as f:
    json.dump(my_test, f)
