# xor-nn

Mini projeto de estudo: uma rede neural pequena que **aprende uma porta lógica
(AND, OR, NAND, NOR, XOR) a partir dos exemplos da tabela verdade**, em vez de
você escrever a regra. O foco é enxergar o que acontece por dentro: perda,
treino, camadas e, principalmente, **os pesos**.

## O que tem aqui

| Arquivo | Para quê |
|---|---|
| `train_xor.py` | Treina a rede com PyTorch, mostra a tabela de resultados e imprime os pesos |
| `hand_built.py` | XOR com pesos escolhidos à mão. Não usa PyTorch nem treino |
| `test_hand_built.py` | Testa a solução feita à mão |
| `requirements.txt` | Dependências (só `torch`) |

## Instalação

Requer Python 3.9 ou superior.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

O PyTorch só para CPU já basta: o problema tem 4 exemplos e treina em segundos.
Se a instalação padrão baixar muita coisa, veja o seletor de instalação em
<https://pytorch.org/get-started/locally/>.

`hand_built.py` e o teste **não precisam de PyTorch** e rodam em qualquer Python.

## Como usar

```bash
python train_xor.py                          # XOR com 4 neurônios escondidos
python train_xor.py --gate and --hidden 0    # AND com um único neurônio
python train_xor.py --gate xor --hidden 0    # XOR sem camada escondida (falha)
python train_xor.py --seed 1                 # outra inicialização
python train_xor.py --save xor.pt            # salva os pesos em arquivo

python hand_built.py                         # XOR com pesos feitos à mão
python -m unittest                           # roda o teste
```

Opções de `train_xor.py`: `--gate`, `--hidden`, `--epochs` (2000), `--lr` (0.05),
`--seed` (0) e `--save`.

### Na IDE

- **VS Code:** abra a pasta, `Ctrl+Shift+P` → *Python: Select Interpreter* →
  escolha o `.venv`. Rode o arquivo com o botão de play ou use o terminal.
- **PyCharm:** *Settings → Project → Python Interpreter → Add Interpreter →
  Existing* e aponte para o `.venv`. Para passar argumentos, edite a
  *Run Configuration* e preencha *Parameters* (por exemplo `--gate and --hidden 0`).

## O que a rede é

```
entrada (x1, x2) -> Linear(2, N) -> tanh -> Linear(N, 1) -> sigmoid -> saída
```

Com `--hidden 0` vira um único neurônio: `Linear(2, 1) -> sigmoid`.

## Como ler os pesos

O script imprime os parâmetros com `model.named_parameters()`. Para `--hidden 4`:

| Nome | Formato | Significado |
|---|---|---|
| `0.weight` | (4, 2) | 4 neurônios escondidos, 1 peso por entrada em cada |
| `0.bias` | (4) | um viés por neurônio escondido |
| `2.weight` | (1, 4) | o neurônio de saída combina os 4 escondidos |
| `2.bias` | (1) | viés da saída |

Os índices são 0 e 2 porque `Tanh` (1) e `Sigmoid` (3) não têm parâmetros.
São 17 parâmetros no total (4×2 + 4 + 4 + 1).

Você também pode acessar direto no código:

```python
model[0].weight   # camada escondida
model[2].weight   # camada de saída
model.state_dict()
```

**Os valores mudam a cada semente.** O treino parte de pesos aleatórios, então
duas execuções corretas podem ter números bem diferentes: neurônios trocados de
lugar, sinais invertidos, alguns redundantes. O que se repete é a *função*
aprendida, não os números.

## A solução feita à mão

`hand_built.py` usa 2 neurônios escondidos e estes pesos:

| Parâmetro | Valor |
|---|---|
| `0.weight` (2×2) | `[[10, 10], [10, 10]]` |
| `0.bias` | `[-5, -15]` |
| `2.weight` (1×2) | `[[10, -10]]` |
| `2.bias` | `[-10]` |

- Neurônio 1 = `tanh(10·(x1 + x2 − 0,5))`: vale ≈ +1 se **alguma** entrada for 1 (OR).
- Neurônio 2 = `tanh(10·(x1 + x2 − 1,5))`: vale ≈ +1 só se **as duas** forem 1 (AND).
- Saída = `sigmoid(10·(n1 − n2) − 10)`: dá 1 quando é OR mas não AND, ou seja, XOR.

Para conferir no PyTorch, treine com `--hidden 2`, copie esses valores para
`model[0].weight`, `model[0].bias`, `model[2].weight` e `model[2].bias` (dentro
de `torch.no_grad()`) e veja se as saídas dão 0, 1, 1, 0.

## Experimentos sugeridos

1. **AND e OR com `--hidden 0`:** um único neurônio resolve, porque os casos são
   separáveis por uma reta.
2. **XOR com `--hidden 0`:** não converge. Nenhuma reta separa o XOR, e é por
   isso que existem camadas escondidas.
3. **XOR com `--hidden 1`, `2`, `3`, `4`:** a partir de quantos neurônios treina
   de forma confiável? Teste várias sementes.
4. **Compare sementes** (`--seed 0..5`) e identifique, nos pesos, qual neurônio
   virou "OR", qual virou "AND" e quais são redundantes.
5. **Sem `Tanh`:** troque por identidade e veja que a rede volta a falhar no XOR
   (camadas lineares empilhadas continuam sendo uma função linear).
6. **Mais entradas:** generalize para 3 ou mais bits (paridade, somador binário)
   e treine com parte da tabela para ver quando a rede generaliza e quando só
   decora.
7. **Exportar para ONNX** com `torch.onnx.export` e rodar com `onnxruntime`.

## Observação

Para a lógica em si, a tabela verdade já é exata e mais barata que qualquer
rede. O valor deste projeto é didático: o problema é pequeno, a resposta certa
é conhecida e dá para verificar tudo.
