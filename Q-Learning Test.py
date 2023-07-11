
import numpy as np

import random

import time

start = time.time()

# 中间写上代码块


r = np.array([[0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],  # 0

              [-1, 0, 1, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],

              [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],

              [-1, -1, -1, 0, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],  # 3

              [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],

              [-1, -1, -1, -1, -1, 0, 1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],  # 5

              [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],  # 6

              [-1, -1, -1, -1, -1, -1, -1, 0, 1, 5, 1, 6, 1, 11, -1, -1, -1, -1, -1],

              [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 1, -1, 11, 6, -1, -1, 1],  # 9

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 0, 1, -1, -1, 5, 11, -1, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 0, 5, -1, 6, -1, 1],  # 13

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, 0, -1, -1, 17, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, -1, -1, 0, -1, 17, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, 0, 17, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1],

              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],  # 18

              ])

q = np.zeros([19, 19], dtype=np.float32)

gamma = 0.9

step = 1

while step < 800:

    state = random.randint(0, 18)

    if step % 799 == 0:
        print('step %d:' % step)

        print(q)

    if state != 18:

        next_state_list = []

        for i in range(19):

            if r[state, i] != -1:
                next_state_list.append(i)

        next_state = next_state_list[random.randint(0, len(next_state_list) - 1)]

        qval = r[state, next_state] + gamma * max(q[next_state])

        q[state, next_state] = qval

        step += 1

for i in range(3):

    print("\nvalidation epoch: {}".format(i + 1))

    state = 1

    print(' first state: {}'.format(state))

    count = 0

    while state != 18:

        if count > 18:
            print('fail')

            break

        q_max = q[state].max()

        q_max_action = []

        for action in range(19):

            if q[state, action] == q_max:
                q_max_action.append(action)

        next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]

        print("the state goes to " + str(next_state) + '.')

        state = next_state

        count += 1

end = time.time()

print('Running time: %s Seconds' % (end - start))