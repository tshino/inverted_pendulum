from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Simulator base
class SimulatorBase:
  # Euler method (not good)
  def step_euler(self, dt):
    deriv = self.deriv(self.state)
    self.state += dt * deriv

  # Runge-Kutta method
  def step_rk4(self, dt):
    s0 = self.state
    k1 = dt * self.deriv(s0)
    k2 = dt * self.deriv(s0 + k1 / 2)
    k3 = dt * self.deriv(s0 + k2 / 2)
    k4 = dt * self.deriv(s0 + k3)
    self.state += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Inverted Pendulum Simulator
class InvertedPendulum(SimulatorBase):

  class Model:
    def __init__(self):
      ObjectProperty = namedtuple('ObjectProperty', 'width height mass inertia')
      self.CART = ObjectProperty(
        width = 0.30,
        height = 0.10,
        mass = 0.2,
        inertia = 0.0 # not used
      )
      self.POLE = ObjectProperty(
        width = 0.04,
        height = 1.0,
        mass = 0.5,
        inertia = 0.04
      )
      self.GRAVITY = 9.8

  def __init__(self, model = Model()):
    self.model = model
    self.state = np.array([ 0.0, 0.0, 0.0, 0.0 ])
    self.force = 0.0

  # Calculate derivative
  # in: state numpy array [x, xdot, a, adot]
  # out: derivative numpy array [xdot, xddot, adot, addot]
  def deriv(self, state):
    m1 = self.model.CART.mass
    m2 = self.model.POLE.mass
    I2 = self.model.POLE.inertia
    h = self.model.POLE.height / 2
    
    x, xdot, a, adot = state
    c = np.cos(a)
    s = np.sin(a)
    
    coef = 1 / (h * h * m2 * (m1 + m2 * s * s) + (m1 + m2) * I2)
    b1 = -h * m2 * s * adot * adot + self.force
    b2 = h * m2 * self.model.GRAVITY * s
    
    xddot = coef * ((h * h * m2 + I2) * b1 + (h * m2 * c) * b2)
    addot = coef * ((h * m2 * c) * b1 + (m1 + m2) * b2)
    
    return np.array([xdot, xddot, adot, addot])

  def total_energy(self):
    m1 = self.model.CART.mass
    m2 = self.model.POLE.mass
    I2 = self.model.POLE.inertia
    h = self.model.POLE.height / 2
    x, xdot, a, adot = self.state
    
    T = (m1 + m2) * xdot * xdot / 2
    T += (h * h * m2 + I2) * adot * adot / 2
    T += -h * m2 * np.cos(a) * xdot * adot
    U = h * m2 * self.model.GRAVITY * np.cos(a)
    return T + U

  def draw(self, ax):
    x, xdot, a, adot = self.state
    s = np.sin(a)
    c = np.cos(a)
    cart = self.model.CART
    pole = self.model.POLE
    p1 = patches.Rectangle(
      xy = (x - cart.width / 2, -cart.height / 2),
      width = cart.width,
      height = cart.height,
      fc='c', ec='k'
    )
    p2 = patches.Rectangle(
      xy = (x - pole.width / 2 * c, - pole.width / 2 * s),
      width = pole.width,
      height = pole.height,
      angle = np.degrees(a),
      fc='b', ec='k'
    )
    return [ax.add_patch(p1), ax.add_patch(p2)]

SIM_FPS = 60
SIM_STEP = 1.0 / SIM_FPS
SIM_DURATION = 15

if __name__ == '__main__':
  
  fig = plt.figure()
  ax = plt.axes()
  ax.set_aspect('equal')
  plt.axis(xmin = -1.8, xmax = 1.8, ymin = -1.2, ymax = 1.2)
  
  ip = InvertedPendulum();
  ip.state = np.random.randn(4) * 0.5
  #ip.state = np.array([ 0.4, 0.0, 0.6, 0.8 ])
  
  GAIN = np.array([ 500, 30, -1000, -30 ])
  
  frames = []
  
  for i in range(int(SIM_DURATION * SIM_FPS)):
    ip.force = np.dot(ip.state, GAIN)
    ip.force = np.clip(ip.force, -500, 500)
    
    frame = ip.draw(ax)
    frame.append(plt.text(0.5, -0.5,'x=%.5f' % ip.state[0]))
    frame.append(plt.text(0.5, -0.6,'x\'=%.5f' % ip.state[1]))
    frame.append(plt.text(0.5, -0.7,'a=%.5f' % ip.state[2]))
    frame.append(plt.text(0.5, -0.8,'a\'=%.5f' % ip.state[3]))
    frame.append(plt.text(0.5, -0.9,'f=%.5f' % ip.force))
    frame.append(plt.text(0.5, -1.0,'E=%.5f' % ip.total_energy()))
    frames.append(frame)
    
    #ip.step_euler(SIM_STEP)
    ip.step_rk4(SIM_STEP)
  
  anim = animation.ArtistAnimation(fig, frames, interval = 1000 * SIM_STEP)
  
  #anim.save("output.mp4")
  #anim.save("output.gif", writer='imagemagick')
  plt.show()
