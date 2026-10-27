"""@flwrlabs/demo: A Flower Hub Demo app."""

import numpy as np
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from quickstart_numpy.task import get_dummy_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    print("👋 Hello from the ServerApp! This is a custom message that I added.")
    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    model = get_dummy_model()
    arrays = ArrayRecord(model)

    # Initialize FedAvg strategy
    strategy = FedAvg()

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    print("Goodbye from the ServerApp! This is my first custom Flower App 🚀!")