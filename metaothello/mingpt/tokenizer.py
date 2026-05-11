from metaothello.constants import SQUARES


class Tokenizer:
    """A simple tokenizer for MetaOthello move sequences."""

    # Special tokens (using standard naming conventions)
    PAD_TOKEN = "[PAD]"
    PASS_TOKEN = None  # Represents a pass move in Othello

    def __init__(self) -> None:
        """Initialize the tokenizer with vocabulary and mappings.

        Vocabulary structure:
        - Index 0: PAD_TOKEN ("[PAD]") for padding sequences
        - Index 1-64: Valid board squares (a1-h8)
        - Index 65: None (represents a pass move)
        """
        self.vocab: list[str | None] = [self.PAD_TOKEN, *SQUARES, self.PASS_TOKEN]
        self.stoi: dict[str | None, int] = {s: i for i, s in enumerate(self.vocab)}
        self.itos: dict[int, str | None] = dict(enumerate(self.vocab))

        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.stoi[self.PAD_TOKEN]

    def encode(self, sequence: list[str | None]) -> list[int]:
        """Encode a sequence of moves to token IDs."""
        return [self.stoi[s] for s in sequence]

    def decode(self, tokens: list[int]) -> list[str | None]:
        """Decode token IDs back to move sequences."""
        return [self.itos[t] for t in tokens]

    def encode_batch(self, sequences: list[list[str | None]]) -> list[list[int]]:
        """Encode multiple sequences at once."""
        return [self.encode(seq) for seq in sequences]

    def decode_batch(self, token_sequences: list[list[int]]) -> list[list[str | None]]:
        """Decode multiple token sequences at once."""
        return [self.decode(tokens) for tokens in token_sequences]

    def pad_sequence(
        self,
        tokens: list[int],
        max_length: int,
        truncate: bool = False,
    ) -> list[int]:
        """Pad (or truncate) a token sequence to a fixed length."""
        current_length = len(tokens)

        if current_length > max_length:
            if truncate:
                return tokens[:max_length]
            raise ValueError(
                f"Sequence length {current_length} exceeds max_length {max_length}. "
                f"Set truncate=True to allow truncation."
            )

        # Pad if shorter
        return tokens + [self.pad_token_id] * (max_length - current_length)

    def pad_batch(
        self,
        token_sequences: list[list[int]],
        max_length: int | None = None,
        truncate: bool = False,
    ) -> list[list[int]]:
        """Pad (or truncate) a batch of sequences to the same length."""
        if max_length is None:
            max_length = max(len(seq) for seq in token_sequences)

        return [self.pad_sequence(seq, max_length, truncate) for seq in token_sequences]
