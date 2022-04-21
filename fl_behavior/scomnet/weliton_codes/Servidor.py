import flwr as fl

# Start Flower server for three rounds of federated learning
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
    min_available_clients=10,  # Minimum number of clients that need to be connected to the server before a training round can start
)
fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 100}, strategy=strategy) #Aumentar o numero de rounds
