import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inverted_pendulum import InvertedPendulum


# PD Controller for Inverted Pendulum
class PDController:
    def __init__(self, gain):
        self.gain_p, self.gain_d = gain
        self.time = 0
        self.ref = 0 # setpoint for the angle theta (state[2])
        self.ref_dot = 0

    def update_ref(self):
        self.ref = np.floor(self.time) * np.math.pi / 5
        self.ref_dot = 0

    def process(self, state, dt):
        self.time += dt

        # input
        self.update_ref()

        # PD Controller
        error = self.ref - state[2]
        error_dot = self.ref_dot - state[3]
        proportional = self.gain_p * error
        derivative = self.gain_d * error_dot
        out = proportional + derivative

        # output
        force = self.non_linear(out, state)

        return force

    def non_linear(self, y, state):
        # We apply a heuristic auto gain calculated below to the output
        # value of the PD controller to compensate varying sensitivity
        # between the input force and the pendulum angle.
        SAFE_THRESHOLD = 0.1
        cos_theta = np.cos(state[2])
        if SAFE_THRESHOLD < np.abs(cos_theta):
            auto_gain = 1 / cos_theta
        else:
            auto_gain = cos_theta / (SAFE_THRESHOLD * SAFE_THRESHOLD)
        y = np.clip(auto_gain * y, -500, 500)
        return y


SIM_FPS = 60


def run_simulation(initial_state, controller, duration):
    sim_step = 1 / SIM_FPS
    sim_iteration = int(duration * SIM_FPS)

    ref_log = []
    state_log = []
    input_log = []

    ip = InvertedPendulum()
    ip.state = initial_state

    for _ in range(sim_iteration):
        ip.force = controller.process(ip.state, dt=sim_step)

        ref_log.append(controller.ref)
        state_log.append(np.copy(ip.state))
        input_log.append(np.copy(ip.force))

        ip.step_rk4(sim_step)

    return (ref_log, state_log, input_log)


def make_animation(ref_log, state_log, input_log):
    fig = plt.figure(figsize=(18.0, 9.6))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_aspect('equal')
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    plt.axis(xmin = -5.5, xmax = 5.5, ymin = -1.2, ymax = 1.2)

    ip = InvertedPendulum()

    t = 0
    frames = []

    for (state, input) in zip(state_log, input_log):
        t += 1 / SIM_FPS
        ip.state = state.copy()
        ip.force = input

        offset = -np.round(ip.state[0] / 14) * 14
        ip.state[0] += offset

        frame = ip.draw(ax)
        frame.append(plt.text(0.5, -0.5,'x=%.5f' % state[0]))
        frame.append(plt.text(0.5, -0.6,'x\'=%.5f' % state[1]))
        frame.append(plt.text(0.5, -0.7,'a=%.5f' % ip.state[2]))
        frame.append(plt.text(0.5, -0.8,'a\'=%.5f' % ip.state[3]))
        frame.append(plt.text(0.5, -0.9,'f=%.5f' % ip.force))
        frame.append(plt.text(0.5, -1.0,'E=%.5f' % ip.total_energy()))
        frame.append(plt.text(0.5, -1.1,'t=%.5f' % t))
        frames.append(frame)

    ax2 = fig.add_subplot(2, 1, 2)
    t = np.arange(len(state_log)) / SIM_FPS
    ax2.plot(t, ref_log, color='magenta', label='ref')
    ax2.plot(t, np.array(state_log)[:,2], color='black', label='a')
    ax2.legend(loc='upper left')

    anim = animation.ArtistAnimation(fig, frames, interval = 1000 / SIM_FPS)
    #anim.save("output.mp4")
    #anim.save("output.gif", writer='imagemagick')
    plt.show()


def main():
    print('...making animation')

    gain  = [125, 25]
    controller = PDController(gain)

    # initial_state = np.random.randn(4) * [0, 1, 2, 3] + [0, 0, 2, 0]
    initial_state = np.array([ 0.0, 0.0, -0.1, 0.0 ])

    print(' gain          = ', gain)
    print(' initial state = ', initial_state)

    ref_log, state_log, input_log = run_simulation(initial_state, controller, 11)
    make_animation(ref_log, state_log, input_log)


if __name__ == '__main__':
    main()
