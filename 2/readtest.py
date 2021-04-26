import re
import matplotlib.pyplot as plt
precision10 = []
precision20 = []
map10 = []
map20 = []
ndcg10 = []
ndcg20 = []
len = 0
with open('test.txt') as f:
    lines = f.readlines()
for line in lines:
    len = len + 1
    a_list = [i.start() for i in re.finditer(':', line)]
    b_list = [i.start() for i in re.finditer(',', line)]
    precision10.append(float(line[a_list[0] + 1:b_list[0]]))
    precision20.append(float(line[a_list[1] + 1:b_list[1]]))
    map10.append(float(line[a_list[2] + 1:b_list[2]]))
    map20.append(float(line[a_list[3] + 1:b_list[3]]))
    ndcg10.append(float(line[a_list[4] + 1:b_list[4]]))
    ndcg20.append(float(line[a_list[5] + 1:-1]))

ranges = []
for i in range(0, len):
    ranges.append(i*20)

plt.plot(ranges, precision10, label='precision10')
plt.plot(ranges, precision20, label='precision20')
plt.plot(ranges, map10, label='map10')
plt.plot(ranges, map20, label='map20')
plt.plot(ranges, ndcg10, label='ndcg10')
plt.plot(ranges, ndcg20, label='ndcg20')

plt.xlabel('epoch')
plt.ylabel('parameters')
plt.grid()
plt.legend()
plt.show()
