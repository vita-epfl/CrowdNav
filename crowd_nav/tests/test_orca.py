import rvo2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from matplotlib import animation


def avg(li):
    if len(li) == 0:
        return 0
    else:
        return sum(li) / len(li)


neighborDist = 15
maxNeighbors = 10
timeHorizon = 10
timeHorizonObst = 10
radius = 1.5
v_pref = 2

M_PI = 3.14159265358979323846
circle_radius = 200
human_num = 100
time_step = 0.25
max_time = 1000
positions = []

all_collisions = []
all_times = []
for test_case in range(1):
    np.random.seed(test_case)
    sim = rvo2.PyRVOSimulator(time_step, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, v_pref)
    for i in range(human_num):
        # while True:
        #     angle = np.random.random() * np.pi * 2
        #     # add some noise to simulate all the possible cases robot could meet with human
        #     px_noise = (np.random.random() - 0.5) * v_pref
        #     py_noise = (np.random.random() - 0.5) * v_pref
        #     px = circle_radius * np.cos(angle) + px_noise
        #     py = circle_radius * np.sin(angle) + py_noise
        #     collide = False
        #     for pos in positions:
        #         if norm((px - pos[0], py - pos[1])) < radius * 2:
        #             collide = True
        #             break
        #     if not collide:
        #         break
        # positions.append(np.array((px, py)))
        positions.append(200 * np.array((np.cos(i * 2 * M_PI / human_num), np.sin(i * 2 * M_PI / human_num))))

    for pos in positions:
        sim.addAgent(tuple(pos.tolist()))

    done = False
    states = [positions]
    collisions = 0
    global_time = 0
    min_dist = float('inf')
    while not done:
        # set preferred velocity
        for i in range(human_num):
            vel_pref = -positions[i] - sim.getAgentPosition(i)
            if norm(vel_pref) > 1:
                vel_pref /= norm(vel_pref)
            sim.setAgentPrefVelocity(i, tuple(vel_pref))
        pos = sim.doStep()
        # for i in range(human_num):
        #     print('python: ', sim.getAgentPosition(i))
        # if global_time > 114.75:
        #     exit()
        global_time += time_step
        print('Global time: {}\n'.format(global_time))
        if global_time > max_time:
            print('Overtime in case {}'.format(test_case))
            break
        states.append([np.array(sim.getAgentPosition(i)) for i in range(human_num)])

        # collision detection
        for i in range(human_num):
            pos_i = np.array(sim.getAgentPosition(i))
            for j in range(human_num):
                if i != j:
                    pos_j = np.array(sim.getAgentPosition(j))
                    dist = norm(pos_i - pos_j) - 2 * radius
                    if dist < min_dist:
                        min_dist = dist
                    if i != j and dist < -1e-4:
                        print('{}th pos: {}'.format(i, pos_i))
                        print('{}th pos: {}'.format(j, pos_j))
                        print('Collision with distance {:.2E}'.format(dist))
                        collisions += 1
                        exit()

        reached_goal = [norm(sim.getAgentPosition(i) - (-positions[i])) < radius
                        for i in range(human_num)]
        done = all(reached_goal)

    print('Case {} has min distance {:.2E} and collisions {}'.format(test_case, min_dist, collisions))
    all_collisions.append(collisions)
    all_times.append(global_time)

print('Average collision numbers: {} in {} seconds'.format(avg(all_collisions), avg(all_times)))

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-circle_radius-2, circle_radius+2)
ax.set_ylim(-circle_radius-2, circle_radius+2)
humans = [plt.Circle(states[0][i], radius, fill=True, color='black')
        for i in range(human_num)]
for i, human in enumerate(humans):
    ax.add_artist(human)


def update(frame_num):
    for i, human in enumerate(humans):
        human.center = states[frame_num][i]


def on_click(event):
    if anim.running:
        anim.event_source.stop()
    else:
        anim.event_source.start()
    anim.running ^= True


fig.canvas.mpl_connect('key_press_event', on_click)
anim = animation.FuncAnimation(fig, update, frames=len(states), interval=time_step*25)
anim.running = True

plt.show()
