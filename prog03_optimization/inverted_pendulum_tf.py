from collections import namedtuple
import numpy as np
import tensorflow as tf


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
class InvertedPendulumTF(SimulatorBase):

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
    self.state = None
    self.force = None

  # Calculate derivative
  # in:  state array      [x, xdot, a, adot]
  # out: derivative array [xdot, xddot, adot, addot]
  def deriv(self, state):
    m1 = self.model.CART.mass
    m2 = self.model.POLE.mass
    I2 = self.model.POLE.inertia
    h = self.model.POLE.height / 2
    
    #x, xdot, a, adot = state[0], state[1], state[2], state[3]
    x, xdot, a, adot = state[:,0], state[:,1], state[:,2], state[:,3]
    #x, xdot, a, adot = tf.unstack(state)
    c = tf.cos(a)
    s = tf.sin(a)
    
    coef = 1 / (h * h * m2 * (m1 + m2 * s * s) + (m1 + m2) * I2)
    b1 = -h * m2 * s * adot * adot + self.force
    b2 = h * m2 * self.model.GRAVITY * s
    
    xddot = coef * ((h * h * m2 + I2) * b1 + (h * m2 * c) * b2)
    addot = coef * ((h * m2 * c) * b1 + (m1 + m2) * b2)
    
    #return tf.stack([xdot, xddot, adot, addot])
    return tf.stack([xdot, xddot, adot, addot], axis=1)
