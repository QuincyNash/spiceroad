import time
import torch
import torch.nn as nn

# Parameters
input_size = 218
num_runs = 100000

# Use all available CPU threads
torch.set_num_threads(torch.get_num_threads())


class TraditionalFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 209)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        y = torch.relu(self.fc2(x))
        z = torch.relu(self.fc3(x))
        return self.fc4(z)


def benchmark_model(model, input_tensor, runs):
    model.eval()

    scripted_model = torch.jit.script(model)

    # Warmup
    with torch.inference_mode():
        for _ in range(100):
            _ = scripted_model(input_tensor)

        start = time.time()
        for _ in range(runs):
            _ = scripted_model(input_tensor)
        end = time.time()

    fps = runs / (end - start)
    print(
        f"{model.__class__.__name__} on {input_tensor.device}: {fps:.2f} inferences/sec"
    )


def main():
    device = torch.device("cpu")
    input_tensor = torch.randn(1, input_size).to(device)

    traditional = TraditionalFF().to(device)
    benchmark_model(traditional, input_tensor, num_runs)


if __name__ == "__main__":
    main()
