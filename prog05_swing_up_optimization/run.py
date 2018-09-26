import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inverted_pendulum import InvertedPendulum
from inverted_pendulum_tf import InvertedPendulumTF


# Swing-up Controller
class SwingUpController:

  def __init__(self, gain):
    self.gain = gain

  def process(self, state):
    force = np.where(
      np.abs(state[2]) < 1.0,
      np.dot(self.gain[0:4], state),
      np.dot(self.gain[4:8], np.array([
        state[0],
        state[1],
        state[2] - 3.0 * np.sign(state[2]),
        state[3]
      ])))
    force = np.clip(force, -500, 500)
    return force

# Swing-up Controller with TensorFlow
class SwingUpControllerTF:

  def __init__(self, gain):
    self.gain = gain

  def process(self, state):
    force = tf.where(
      tf.abs(state[:,2]) < 1.0,
      tf.reduce_sum(self.gain[0:4] * state, axis=1),
      tf.reduce_sum(self.gain[4:8] * tf.stack([
        state[:,0],
        state[:,1],
        state[:,2] - 3.0 * tf.sign(state[:,2]),
        state[:,3]
      ], axis=1), axis=1))
    force = tf.clip_by_value(force, -500, 500)
    return force


SIM_FPS = 60

# Run simulation in normal way
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

# Run simulation using TensorFlow
def run_simulation_tf(initial_state, controller, duration):
  sim_step = 1 / SIM_FPS
  sim_iteration = int(duration * SIM_FPS)
  
  ip = InvertedPendulumTF()
  ip.state = tf.constant(initial_state)
  
  for i in range(sim_iteration):
    ip.force = controller.process(ip.state)
    ip.step_rk4(sim_step)
  
  return ip.state


USE_SCIPY_OPTIMIZER = True


def calc_state_loss(state):
  dist_a = tf.reduce_sum(tf.square(state), axis=1)
  dist_b = tf.reduce_sum(tf.square(tf.abs(state) - np.array([0, 0, 3, 0])), axis=1)

  # The objective is to be near A while be apart from B.
  #   A ... unstable equibrium point (origin)
  #   B ... stable equibrium point
  # --> minimize dist_a while maximize dist_b  ??
  # --> minimize (dist_a / (dist_a + dist_b))

  return dist_a / (dist_a + 5 * tf.clip_by_value(dist_b, 0, 10))


def make_objective_function(gain):
  initial_state = np.vstack((
    np.random.randn(200, 4) * 0.0001,
    np.random.randn(200, 4) * 0.001,
    np.random.randn(200, 4) * 0.01,
    np.random.randn(200, 4) * 0.1,
    np.random.randn(200, 4) * 1.0
  ))
  initial_state = np.clip(initial_state, -1, 1)
  initial_state_loss = calc_state_loss(initial_state)

  controller_tf = SwingUpControllerTF(gain)
  last_state = run_simulation_tf(initial_state, controller_tf, 2.0)

  last_state_loss = calc_state_loss(last_state)
  loss = tf.reduce_mean(last_state_loss / initial_state_loss)

  return (loss, last_state)


def run_optimization(initial_gain):
  print('...creating simulator model')

  gain = tf.Variable(initial_gain)
  loss, last_state = make_objective_function(gain)

  if USE_SCIPY_OPTIMIZER:
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss)
  else:
    optimizer = tf.train.AdamOptimizer(0.5)
    #optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
  
  print('...starting ML session')
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  print('...initial evaluation')
  
  print(' gain =', sess.run(gain))
  print(' last state =', sess.run(last_state))
  print(' loss =', sess.run(loss))
  
  print('...training')
  
  if USE_SCIPY_OPTIMIZER:
    optimizer.minimize(sess)
    print(' gain =', sess.run(gain))
    print(' loss =', sess.run(loss))
  else:
    for i in range(1000):
      print('=== epoch #%d ===' % (i + 1))
      sess.run(train)
      print(' gain =', sess.run(gain))
      print(' loss =', sess.run(loss))
  
  return sess.run(gain)


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

  if True:
    initial_gain = np.zeros(8)
    gain = run_optimization(initial_gain)
  else:
    gain = np.array([ 8, 9, -36, -12, -18, -30, -36, 12])
    print(' gain          = ', gain)

  #initial_state = np.random.randn(4) * 2.0
  initial_state = np.array([ 1.0, 0.0, 2.0, 3.0 ])

  print('...making animation using trained controller')
  print(' initial state = ', initial_state)

  controller = SwingUpController(gain)
  state_log, input_log = run_simulation(initial_state, controller, 8)
  make_animation(state_log, input_log)
