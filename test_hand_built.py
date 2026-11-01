import unittest

from hand_built import EXPECTED, INPUTS, forward


class HandBuiltXorTest(unittest.TestCase):
    def test_tabela_verdade(self):
        for (x1, x2), esperado in zip(INPUTS, EXPECTED):
            _, saida = forward(x1, x2)
            self.assertEqual(round(saida), esperado, f"entrada ({x1}, {x2})")

    def test_saidas_confiantes(self):
        # As saídas devem ficar bem perto de 0 ou de 1.
        for (x1, x2), esperado in zip(INPUTS, EXPECTED):
            _, saida = forward(x1, x2)
            self.assertLess(abs(saida - esperado), 0.01)


if __name__ == "__main__":
    unittest.main()
