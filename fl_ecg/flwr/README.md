
# Instalação de dependências:

python3 -m pip install -r requirements.txt

Após a instalação das dependências, basta inicializar um servidor e dois ou mais clientes, de forma respectivas. Note que o dataset original não está contido nessa pasta devido ao fato de ser muito grande.

# Inicialização de um servidor:

python3 Servidor.py

# Inicialização de um cliente:
python3 Cliente.py <x_dataset.csv>, onde x é um número entre 0 e 20 incluindo os extremos caso o splitDataset.py tenha sido executado com o dataset original
