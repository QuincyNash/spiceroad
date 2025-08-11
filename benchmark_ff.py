import time
import torch
import torch.backends
import torch.nn as nn
from fastfeedforward import FFF


# Parameters
input_size = 700
output_size = 200
num_runs = 100000


class TraditionalFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 200),
        )

    def forward(self, x):
        return self.net(x)


def benchmark_model(model, input_tensor, runs):
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)

        start = time.time()
        for _ in range(runs):
            _ = model(input_tensor)
        end = time.time()

    total = runs
    duration = end - start
    print(
        f"{model.__class__.__name__} on {input_tensor.device}: {total/duration:.2f} inferences/sec"
    )


def main():
    device = torch.device("cpu")

    input_tensor = torch.randn(1, input_size).to(device)

    traditional = TraditionalFF().to(device)

    print("Benchmarking models:")
    benchmark_model(traditional, input_tensor, num_runs)


if __name__ == "__main__":
    main()
