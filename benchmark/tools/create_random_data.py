#!/usr/bin/env python3

import random

SIZE = 16

for i in range(SIZE):
    for j in range(SIZE):
        print(random.randint(0, 99), ',', sep='', end='')
    print()
