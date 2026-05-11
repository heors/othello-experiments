"""Constants for MetaOthello setup."""

### Games
BLACK = -1
WHITE = 1
EMPTY = 0
DIRECTIONS = [[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
MAX_STEPS = 60
BOARD_DIM = 8

letters = "abcdefgh"
number = "12345678"

tuple2move = {(i, j): letters[j] + number[i] for i in range(BOARD_DIM) for j in range(BOARD_DIM)}

move2tuple = {letters[j] + number[i]: (i, j) for i in range(BOARD_DIM) for j in range(BOARD_DIM)}


SQUARES = list(move2tuple.keys())

### Tokenizer
