import flwr as fl

# Start Flower server for three rounds of federated learning
fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 100}) #Aumentar o numero de rounds
