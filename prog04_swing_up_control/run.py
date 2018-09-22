import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inverted_pendulum import InvertedPendulum


# Swing-up Controller
class SwingUpController:
  def __init__(self, gain):
    self.gain = gain

  def process(self, state):
    
    if np.abs(state[2]) < 1.0:
      force = np.dot(self.gain[0:4], state)
    else:
      force = np.dot(self.gain[4:8], np.array([
          state[0],
          state[1],
          state[2] - 3.0 * np.sign(state[2]),
          state[3]
      ]))
    
    force = np.clip(force, -500, 500)
    return force


SIM_FPS = 60


def run_simulation(initial_state, controller, duration):
  sim_step = 1 / SIM_FPS
  sim_iteration = int(duration * SIM_FPS)
  
  state_log = []
  input_log = []
  
  ip = InvertedPendulum()
  ip.state = initial_state
  
  for i in range(sim_iteration):
    ip.force = controller.process(ip.state)
    
    state_log.append(np.copy(ip.state))
    input_log.append(np.copy(ip.force))
    
    ip.step_rk4(sim_step)
  
  return (state_log, input_log)


def make_animation(state_log, input_log):
  fig = plt.figure()
  ax = plt.axes()
  ax.set_aspect('equal')
  plt.axis(xmin = -1.8, xmax = 1.8, ymin = -1.2, ymax = 1.2)
  
  ip = InvertedPendulum();
  
  frames = []
  
  for (state, input) in zip(state_log, input_log):
    ip.state = state
    ip.force = input
    
    frame = ip.draw(ax)
    frame.append(plt.text(0.5, -0.5,'x=%.5f' % ip.state[0]))
    frame.append(plt.text(0.5, -0.6,'x\'=%.5f' % ip.state[1]))
    frame.append(plt.text(0.5, -0.7,'a=%.5f' % ip.state[2]))
    frame.append(plt.text(0.5, -0.8,'a\'=%.5f' % ip.state[3]))
    frame.append(plt.text(0.5, -0.9,'f=%.5f' % ip.force))
    frame.append(plt.text(0.5, -1.0,'E=%.5f' % ip.total_energy()))
    frames.append(frame)
  
  anim = animation.ArtistAnimation(fig, frames, interval = 1000 / SIM_FPS)
  
  #anim.save("output.mp4")
  #anim.save("output.gif", writer='imagemagick')
  plt.show()


if __name__ == '__main__':
  print('...making animation')
  
  gain  = np.array([ 8, 9, -36, -12, -18, -30, -36, 12])
  controller = SwingUpController(gain)
  
  initial_state = np.random.randn(4) * 2.0
  #initial_state = np.array([ 1.0, 0.0, 2.0, 3.0 ])
  
  print(' gain          = ', gain)
  print(' initial state = ', initial_state)
  
  state_log, input_log = run_simulation(initial_state, controller, 8)
  make_animation(state_log, input_log)
