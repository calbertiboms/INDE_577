# Gradient Descent Project

This project implements **Gradient Descent**, an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent as defined by the negative of the gradient.

![Gradient Descent](https://miro.medium.com/v2/resize:fit:1200/0*7VyOHNGxbSvHoKAP.jpg)
### Key Concepts:
- **Gradient Descent Algorithm**: The main idea is to update parameters (weights) iteratively to reduce the cost function. At each iteration, the algorithm calculates the gradient of the cost function with respect to the parameters, and then it updates the parameters in the direction that reduces the cost.
- **Learning Rate**: This is a hyperparameter that controls the size of the steps taken to reach the minimum. A higher learning rate means faster convergence, but if it's too high, it might overshoot the minimum.
- **Convergence**: Gradient Descent continues until the algorithm reaches a minimum or a sufficiently small gradient, indicating convergence.

### Highlights:
- This implementation demonstrates how Gradient Descent can be used to minimize the power dissipation in an electric circuit by adjusting the current.
- The project calculates the optimal current that minimizes power dissipation based on a mathematical model.

### Requirements:
- Python 3.x
- Libraries: NumPy, Matplotlib (for visualizing the optimization process)
