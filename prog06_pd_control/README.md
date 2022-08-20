https://user-images.githubusercontent.com/732920/185763491-f1fde0c7-ef82-4716-ac0f-7c2682142635.mp4

# PD Control

This is a simulation of an inverted pendulum with a manually adjusted PD controller.

The objective of the controller is to control the pendulum to arbitrary angles.
It works as shown in the animation above.
Obviously, the horizontal position of the cart ($x$) is left uncontrolled. Because this PD controller can control only a single variable.

You may also notice a small amount of residual error, I think it is due to the absence of the integral component in the controller.

![](figs/block_diagram.png)

### 2022-08-21
- Initial version
