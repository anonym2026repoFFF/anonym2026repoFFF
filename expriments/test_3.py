from datasets import load_dataset, load_from_disk

path = r"C:\Users\robor\.cache\huggingface\test\slim"
# Replace with your actual file path and format
dataset = load_dataset(
    path,  # Can also be a dict or list
)

# Display a few rows
print(dataset)
print(dataset['train'][0])  # First row of the training set
