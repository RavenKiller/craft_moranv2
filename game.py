import random
a = 0
b = 0
total = 10000
for _ in range(total):
    seq = []
    while True:
        n = len(seq)
        if n>=3 and ''.join(seq[n-3:n])=='YNN':
            a += 1
            break
        if n>=3 and ''.join(seq[n-3:n])=='NNY':
            b += 1
            break
        seq.append(('N' if random.random()<0.5 else 'Y'))

print("共试验%d次" % (total))
print("甲获胜%d场" % (a))
print("乙获胜%d场" % (b))