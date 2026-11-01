"""Treina uma rede neural pequena para aprender uma porta lógica a partir
da tabela verdade e mostra os pesos aprendidos.

Exemplos:
    python train_xor.py                        # XOR com 4 neurônios escondidos
    python train_xor.py --gate and --hidden 0  # AND com um único neurônio
    python train_xor.py --gate xor --hidden 0  # XOR sem camada escondida: falha
    python train_xor.py --seed 1               # outra inicialização, outros pesos
    python train_xor.py --save xor.pt          # salva os pesos em arquivo
"""

import argparse

import torch
import torch.nn as nn

# As 4 combinações possíveis de duas entradas binárias.
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Saída esperada de cada porta, na mesma ordem de INPUTS.
GATES = {
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "nand": [1, 1, 1, 0],
    "nor": [1, 0, 0, 0],
    "xor": [0, 1, 1, 0],
}


def build_model(hidden: int) -> nn.Sequential:
    """Com hidden=0 cria um único neurônio; senão, uma camada escondida."""
    if hidden == 0:
        return nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    return nn.Sequential(
        nn.Linear(2, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


def train(model, X, y, epochs: int, lr: float, log_every: int) -> float:
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = None
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"época {epoch:5d}  perda {loss.item():.4f}")
    return loss.item()


def show_results(model, y) -> int:
    with torch.no_grad():
        out = model(torch.tensor(INPUTS, dtype=torch.float32))
    acertos = 0
    print("\nx1 x2 | esperado | saída  | arredondada")
    for (x1, x2), esperado, saida in zip(
        INPUTS, y.flatten().tolist(), out.flatten().tolist()
    ):
        previsto = round(saida)
        acertos += int(previsto == int(esperado))
        print(f" {x1}  {x2} | {int(esperado):^8} | {saida:.4f} | {previsto}")
    print(f"\nAcertos: {acertos}/4")
    return acertos


def show_weights(model) -> None:
    print("\n--- pesos ---")
    for nome, p in model.named_parameters():
        print(f"{nome} {tuple(p.shape)}")
        print(p.detach())
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal de parâmetros: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gate", choices=sorted(GATES), default="xor")
    parser.add_argument(
        "--hidden", type=int, default=4,
        help="neurônios na camada escondida (0 = sem camada escondida)",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", metavar="ARQUIVO", help="salva o state_dict")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    X = torch.tensor(INPUTS, dtype=torch.float32)
    y = torch.tensor(GATES[args.gate], dtype=torch.float32).unsqueeze(1)

    model = build_model(args.hidden)
    print(f"Porta: {args.gate.upper()} | escondidos: {args.hidden} | seed: {args.seed}\n")

    train(model, X, y, args.epochs, args.lr, log_every=max(1, args.epochs // 5))
    show_results(model, y)
    show_weights(model)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"\nPesos salvos em {args.save}")


if __name__ == "__main__":
    main()
