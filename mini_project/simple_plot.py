import matplotlib.pyplot as plt
import numpy as np

# Data
num_scenarios = [1, 3, 10]
train_route_completion = [0.38260415, 0.5922889, 0.5259787]
eval_route_completion = [0.5580398, 0.3991208, 0.3761546]

# Plot
plt.figure(figsize=(6, 3))
plt.plot(num_scenarios, train_route_completion, color = "darkturquoise", marker='o', label='Train')
plt.plot(num_scenarios, eval_route_completion, color = "violet", marker='o', label='Eval')

# Labels and title
plt.xlabel('Num Scenarios')
plt.ylabel('Final Route Completion')
plt.title('Final Route Completion vs Num Scenarios')
plt.legend()
plt.grid(True)

# Show plot
plt.show()