import matplotlib.pyplot as plt

def render_mac(env):
    frame = env.render(mode='rgb_array')
    plt.imshow(frame)
    plt.axis('off')
    plt.pause(0.001)  # Adjust the pause duration as needed
    plt.clf()
