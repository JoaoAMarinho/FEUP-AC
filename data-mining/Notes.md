## Tabelas

### Account

- ID
- ID distrito
- Frequencia de declaração
- Data de criação da conta (YYMMDD)

### Client

- ID
- Dia de anos e sexo (YYMMDD Homem, YYMM+50DD Mulher)
- ID distrito

### Disposition

- ID
- ID do cliente
- ID da conta
- Tipo de associação

Fornece info:
- Número de associados a uma conta
- Quantas contas tem um cliente
- Grau de associação

### Transaction

- ID
- ID da conta
- Dia da transação
- Tipo (crédito/receber, widthrawal/retirar)
- Operação efetuada (depósito, troca de dinheiro entre contas, retirar em dinheiro, ...)
- Quantidade
- Dinheiro após transação (balanço)
- Caracterização da transaçao
- Banco do parceiro (se em questão)
- Conta do parceiro

### Loan

- ID
- Conta associada
- Dia que foi aceite
- Quantia
- Duração da loan (meses)
- Quantia de pagamento mensal
- Estado da loan (acabou nao acabou?)

### Credit Card

- ID
- ID da disposição
- Tipo de cartão (classic, gold, junior)
- Data de emissão

Fornece info:
- Quantos cartoes por conta

### Demographic data

- Código, Nome e Região
- Numero de habitantes
- Municipios com menos de x habitantes ...
- Numero de cidades
- Salario medio
- Taxa de desemprego
- Numero de entrepreneurs
- Numero de crimes