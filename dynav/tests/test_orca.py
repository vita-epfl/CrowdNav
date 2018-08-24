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


neighborDist = 10
maxNeighbors = 10
timeHorizon = 5
timeHorizonObst = 5
radius = 0.3
v_pref = 1

circle_radius = 4
ped_num = 10
time_step = 0.25
max_time = 1000
positions = []

all_collisions = []
all_times = []
for test_case in range(5):
    np.random.seed(test_case)
    sim = rvo2.PyRVOSimulator(time_step, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, v_pref)
    for i in range(ped_num):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases navigator could meet with pedestrian
            px_noise = (np.random.random() - 0.5) * v_pref
            py_noise = (np.random.random() - 0.5) * v_pref
            px = circle_radius * np.cos(angle) + px_noise
            py = circle_radius * np.sin(angle) + py_noise
            collide = False
            for pos in positions:
                if norm((px - pos[0], py - pos[1])) < radius * 2:
                    collide = True
                    break
            if not collide:
                break
        positions.append(np.array((px, py)))

    for pos in positions:
        sim.addAgent(tuple(pos.tolist()))

    done = False
    states = [positions]
    collisions = 0
    global_time = 0
    min_dist = float('inf')
    while not done:
        # set preferred velocity
        for i in range(ped_num):
            vel_pref = -positions[i] - sim.getAgentPosition(i)
            if norm(vel_pref) > 1:
                vel_pref /= norm(vel_pref)
            sim.setAgentPrefVelocity(i, tuple(vel_pref))
        sim.doStep()
        global_time += time_step
        if global_time > max_time:
            print('Overtime in case {}'.format(test_case))
            break
        states.append([np.array(sim.getAgentPosition(i)) for i in range(ped_num)])

        # collision detection
        for i in range(ped_num):
            pos_i = np.array(sim.getAgentPosition(i))
            for j in range(ped_num):
                if i != j:
                    pos_j = np.array(sim.getAgentPosition(j))
                    dist = norm(pos_i - pos_j) - 2 * radius
                    if dist < min_dist:
                        min_dist = dist
                    if i != j and dist < -1e-2:
                        print('Collision with distance {:.2E}'.format(dist))
                        collisions += 1

        reached_goal = [norm(sim.getAgentPosition(i) - (-positions[i])) < radius
                        for i in range(ped_num)]
        done = all(reached_goal)

    print('Case {} has min distance {:.2E} and collisions {}'.format(test_case, min_dist, collisions))
    all_collisions.append(collisions)
    all_times.append(global_time)

print('Total collision numbers: {} in {} seconds'.format(avg(all_collisions), avg(all_times)))

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-circle_radius-2, circle_radius+2)
ax.set_ylim(-circle_radius-2, circle_radius+2)
peds = [plt.Circle(states[0][i], radius, fill=True, color='black')
        for i in range(ped_num)]
for i, ped in enumerate(peds):
    ax.add_artist(ped)


def update(frame_num):
    for i, ped in enumerate(peds):
        ped.center = states[frame_num][i]


def on_click(event):
    if anim.running:
        anim.event_source.stop()
    else:
        anim.event_source.start()
    anim.running ^= True


fig.canvas.mpl_connect('key_press_event', on_click)
anim = animation.FuncAnimation(fig, update, frames=len(states), interval=time_step*250)
anim.running = True

plt.show()
