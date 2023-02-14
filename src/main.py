import random

from dataclasses import dataclass, field
from typing import *

import numpy as np
import tqdm


@dataclass
class Config:
    n_players: int
    n_guess: int
    n_neutral: int
    n_black: int

    @property
    def n_words(self) -> int:
        return self.n_neutral + self.n_black + self.n_guess * self.n_players


@dataclass
class Game:
    player_words: list[list[str]]
    neutral_words: list[str]
    black_words: list[str]

    @property
    def game_words(self) -> list[str]:
        res = []
        for pw in self.player_words:
            res.extend(pw)
        res.extend(self.neutral_words)
        res.extend(self.black_words)
        return res

    def get_player_words_indices(self, player: int) -> list[int]:
        n_before = sum(map(len, self.player_words[:player]))
        return list(range(n_before, n_before + len(self.player_words[player])))


@dataclass
class Embeddings:
    words: list[str]
    rev_words: dict[str, int] = field(init=False)
    embeddings: np.ndarray
    _emb_words: list[str] = field(init=False)

    def __post_init__(self):
        assert len(self.words) == self.embeddings.shape[0]
        self.rev_words = dict([(w, i) for i, w in enumerate(self.words)])
        self._emb_words = self.words

    def __getitem__(self, key):
        return self.embeddings[self.rev_words[key], :]

    def intersect_dict(self, other_dict: set[str]):
        self.rev_words = {k: v for k, v in self.rev_words.items() if k in other_dict}
        self.words = [w for w in self.words if w in other_dict]


@dataclass
class ScoreMatrix:
    words: list[str]
    rev_words: dict[str, int] = field(init=False)
    scores: np.ndarray

    def __post_init__(self):
        assert len(self.words) == self.scores.shape[1]
        self.rev_words = dict([(w, i) for i, w in enumerate(self.words)])

    def __getitem__(self, key):
        return self.scores[:, self.rev_words[key]]


def score_playable_words(game_words: list[str], embeddings: Embeddings) -> ScoreMatrix:
    game_words_embeddings = np.stack([embeddings[w] for w in game_words])
    playable_words_embeddings = np.stack([embeddings[w] for w in embeddings.words])

    return ScoreMatrix(
        words=embeddings.words,
        scores=game_words_embeddings @ playable_words_embeddings.T,
    )


def base_count_points(indices: set[int], score: np.ndarray, order: np.ndarray) -> int:
    i = 0
    while order[-1 - i] in indices:
        i += 1
    return i


def threshold_count_points(
    indices: set[int], score: np.ndarray, order: np.ndarray, threshold: float = 0.3
) -> int:
    index_first_other = next(i for i in reversed(order) if i not in indices)
    biggest_other_score = score[index_first_other]

    i = 0
    while (player_index := order[-1 - i]) in indices and score[
        player_index
    ] > biggest_other_score + threshold:
        i += 1
    return i


def find_best_match(
    game: Game, player: int, score_matrix: ScoreMatrix, fn_count_points, *args
) -> tuple[list[str], int]:
    p_word_indices = set(game.get_player_words_indices(player))
    sorted_indices = np.argsort(score_matrix.scores, axis=0)

    best_points, best_word = 0, []
    for i, w in tqdm.tqdm(enumerate(score_matrix.words), total=len(score_matrix.words)):
        if w in game.game_words:
            continue
        local_order = sorted_indices[:, i]
        local_score = score_matrix.scores[:, i]

        # word_points = base_count_points(p_word_indices, local_score, local_order)
        word_points = fn_count_points(p_word_indices, local_score, local_order, *args)

        if word_points > best_points:
            best_points = word_points
            best_word = [w]
        if word_points == best_points:
            best_word.append(w)
    return best_word, best_points


def load_w2v(file: str):
    words = []
    with open(file, "r", encoding="utf8") as fp:
        lines = fp.readlines()
        n, m = map(int, lines[0].split())
        embeddings = np.zeros((n, m))
        for i, line in tqdm.tqdm(enumerate(lines[1:]), total=n):
            w, *emb = line.split()
            emb_vector = np.array([float(e) for e in emb], dtype=np.float16)
            words.append(w.strip().lower())
            embeddings[i, :] = emb_vector
    return Embeddings(words=words, embeddings=embeddings)


def load_codenames_words(lang: str) -> list[str]:
    with open(
        f"codenames/wordlist/{lang}/default/wordlist.txt", "r", encoding="utf8"
    ) as fp:
        return [s.strip().lower() for s in fp.readlines()]


def load_dict_words(lang: str) -> list[str]:
    with open(f"/usr/share/dict/{lang}", "r", encoding="utf8") as fp:
        return [s.strip().lower() for s in fp.readlines()]


def load_playable_words(w2v_file: str) -> tuple[list[str], Embeddings]:
    codenames_words = set(load_codenames_words("fr-FR"))
    dict_words = set(load_dict_words("french"))
    embeddings = load_w2v(w2v_file)
    embeddings.intersect_dict(dict_words)

    codenames_words.intersection_update(embeddings.words)
    return list(codenames_words), embeddings


def chunk_list(l: list, n: int) -> list[list]:
    assert len(l) % n == 0, "List length not compatible"
    chunk_len = len(l) // n
    res = []
    for j in range(n):
        res.append(l[j * chunk_len : (j + 1) * chunk_len])
    return res


def draw_game(game_config: Config, dictionay: list[str]) -> Game:
    n_words = game_config.n_words
    game_words = random.sample(dictionay, n_words)
    i = 0

    black_words = game_words[i : i + game_config.n_black]
    i += game_config.n_black

    neutral_words = game_words[i : i + game_config.n_neutral]
    i += game_config.n_neutral

    player_words = chunk_list(game_words[i:], game_config.n_players)

    return Game(
        player_words=player_words, neutral_words=neutral_words, black_words=black_words
    )


def main():
    words, embeddings = load_playable_words("data/cc.fr.300.vec")
    cfg = Config(n_players=2, n_guess=8, n_neutral=8, n_black=1)
    game = draw_game(cfg, words)
    print(score_playable_words(game.game_words, embeddings).shape)


if __name__ == "__main__":
    words, embeddings = load_playable_words("data/cc.fr.300.vec")
    cfg = Config(n_players=2, n_guess=8, n_neutral=8, n_black=1)
    game = draw_game(cfg, words)
    score_matrix = score_playable_words(game.game_words, embeddings)
