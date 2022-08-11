import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inverted_pendulum import InvertedPendulum
from inverted_pendulum_tf import InvertedPendulumTF

# State Feedback Controller
class StateFeedbackController:

  def __init__(self, gain):
    self.gain = gain

  def process(self, state):
    force = np.dot(state, self.gain)
    force = np.clip(force, -500, 500)
    return force

# State Feedback Controller for TensorFlow
class StateFeedbackControllerTF:

  def __init__(self, gain):
    self.gain = gain

  def process(self, state):
    force = tf.reduce_sum(state * self.gain, axis=1)
    force = tf.clip_by_value(force, -500, 500)
    return force


SIM_FPS = 60
SIM_DURATION = 5

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

# Run simulation in normal way
def run_simulation(initial_state, controller, duration = SIM_DURATION):
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

# Run simulation using TensorFlow
def run_simulation_tf(initial_state, controller, duration = SIM_DURATION):
  sim_step = 1 / SIM_FPS
  sim_iteration = int(duration * SIM_FPS)
  
  ip = InvertedPendulumTF()
  ip.state = tf.constant(initial_state)
  
  for i in range(sim_iteration):
    ip.force = controller.process(ip.state)
    ip.step_rk4(sim_step)
  
  return ip.state


# USE_BFGS_OPTIMIZER = False    # use Adam
USE_BFGS_OPTIMIZER = True   # use BFGS


if __name__ == '__main__':
  print('...creating simulator model')
  
  #initial_state = np.random.randn(100, 4) * 0.2
  #initial_state = np.random.randn(100, 4) * 0.1
  initial_state = np.random.randn(1000, 4) * 0.01
  
  gain = tf.Variable(np.array([ 0.0, 0.0, 0.0, 0.0 ]))
  #gain = tf.Variable(np.array([ 500.0, 30.0, -1000.0, -30.0 ]))
  
  loss = None
  
  def calc_loss(gain):
    global loss
    controller_tf = StateFeedbackControllerTF(gain)
    last_state = run_simulation_tf(initial_state, controller_tf, 1.0)
    clamped_last_state = tf.clip_by_value(last_state, -1, 1)
    loss = tf.reduce_mean(tf.square(clamped_last_state))
    return loss

  if USE_BFGS_OPTIMIZER:
    @tf.function
    def optimize():
      return tfp.optimizer.lbfgs_minimize(
        lambda x: tfp.math.value_and_gradient(calc_loss, x),
        initial_position=tf.constant(np.array([ 0.0, 0.0, 0.0, 0.0 ])),
        tolerance=1e-5
      )
  else:
    optimizer = tf.keras.optimizers.Adam(0.5)
  
  print('...initial evaluation')
  calc_loss(gain)
  print(' gain =', gain.numpy())
  print(' loss =', loss.numpy())

  print('...training')
  
  if USE_BFGS_OPTIMIZER:
    results = optimize()
    print(f' converged: {results.converged}')
    print(f' # of iterations: {results.num_iterations}')
    print(f' gain = {results.position}')
    print(f' loss = {results.objective_value}')
    gain = results.position
  else:
    for i in range(1000):
      print('=== epoch #%d ===' % (i + 1))
      optimizer.minimize(lambda: calc_loss(gain), [gain])
      print(' gain =', gain.numpy())
      print(' loss =', loss.numpy())
  
  print('...making animation using trained controller')
  
  gain = gain.numpy()
  #gain = np.array([ 500, 30, -1000, -30 ])
  
  controller = StateFeedbackController(gain)
  #initial_state = np.random.randn(4) * 0.5
  initial_state = np.array([ 0.4, 0.0, 0.6, 0.8 ])
  state_log, input_log = run_simulation(initial_state, controller)
  make_animation(state_log, input_log)
