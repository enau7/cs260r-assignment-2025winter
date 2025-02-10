import torch

# Original tensors
tensor = torch.tensor([[1, 2], [3, 4]])
indices = torch.tensor([[1], [0]])

# Use gather to index the tensor along dimension 1
result = tensor.gather(1, indices)

# Transpose the result to get the desired shape
result = result.transpose(0, 1)

print(result)