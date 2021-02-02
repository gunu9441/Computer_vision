import random

lotto = []
game = []
for i in range(5):
    while True:
        if len(lotto) == 6:
            break
        a = random.randint(1, 45)
        if a in lotto:
            continue
        else:
            lotto.append(a)
    game.append(lotto)
print(lotto)
