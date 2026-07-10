import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

cheat_values = []

def update_graph(i):
    global cheat_values

    # Generate fake cheat probability for now (AI model will come later)
    cheat_prob = random.uniform(0.0, 1.0)
    cheat_values.append(cheat_prob)

    if len(cheat_values) > 50:
        cheat_values.pop(0)

    plt.cla()
    plt.plot(cheat_values, linewidth=2)
    plt.ylim(0, 1)
    plt.title("Suspicious Behaviour Detection")
    plt.ylabel("Cheat Probability")
    plt.xlabel("Time (frames)")
    plt.grid(True)


def run_detection():
    fig = plt.figure("Suspicious Behaviour Detection")
    ani = animation.FuncAnimation(fig, update_graph, interval=500)
    plt.show()


if __name__ == "__main__":
    run_detection()
