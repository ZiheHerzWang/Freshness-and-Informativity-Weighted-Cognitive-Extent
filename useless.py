import numpy as np
import matplotlib.pyplot as plt

# Define the function y(x)
def y(x):
    return (-3 + np.sqrt(65 - 8*np.exp(-x) - 8*np.exp(x))) / 4

# Create an array of x values within a reasonable range
x_values = np.linspace(-2, 2, 400)

# Calculate the corresponding y values
y_values = y(x_values)

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=r'$y(x) = \frac{-3 + \sqrt{65 - 8e^{-x} - 8e^x}}{4}$', color='orange')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Plot of the solution")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
