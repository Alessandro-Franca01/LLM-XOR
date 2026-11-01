"""XOR com pesos escolhidos à mão, sem treino e sem PyTorch (só a biblioteca
padrão do Python).

Mesma arquitetura de train_xor.py com --hidden 2:
    entrada (2) -> Linear(2, 2) -> tanh -> Linear(2, 1) -> sigmoid

Ideia:
    neurônio 1 ~ OR  : +1 se alguma entrada for 1
    neurônio 2 ~ AND : +1 se as duas entradas forem 1
    saída = OR e não AND = XOR
"""

import math

# Equivalente a 0.weight, 0.bias, 2.weight e 2.bias do modelo PyTorch.
W_HIDDEN = [[10.0, 10.0], [10.0, 10.0]]
B_HIDDEN = [-5.0, -15.0]
W_OUT = [10.0, -10.0]
B_OUT = -10.0

INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
EXPECTED = [0, 1, 1, 0]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def forward(x1: float, x2: float):
    """Retorna (ativações escondidas, saída)."""
    hidden = [
        math.tanh(w[0] * x1 + w[1] * x2 + b)
        for w, b in zip(W_HIDDEN, B_HIDDEN)
    ]
    logit = sum(w * h for w, h in zip(W_OUT, hidden)) + B_OUT
    return hidden, sigmoid(logit)


def main() -> None:
    print("x1 x2 |   n1      n2   | saída  | arredondada | esperado")
    for (x1, x2), esperado in zip(INPUTS, EXPECTED):
        (n1, n2), saida = forward(x1, x2)
        print(
            f" {x1}  {x2} | {n1:+.4f} {n2:+.4f} | {saida:.4f} |"
            f"      {round(saida)}      |    {esperado}"
        )


if __name__ == "__main__":
    main()
