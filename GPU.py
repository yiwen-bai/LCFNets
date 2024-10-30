import numpy as np
import statistics


def calculate_score(card_position, total_cards, max_score, min_score):
    score = max_score - ((card_position - 1) / (total_cards - 1)) * (max_score - min_score)
    return score

# 假设按性能从低到高排序的显卡列表
graphics_cards = [
"Titan X", "GTX 1080 Ti", "Titan Xp", "RTX 2080 Ti", "RTX 3090 Ti" # ... 继续排列其他显卡直至最低性能的显卡
]

# 计算每个显卡的分数
for i, card in enumerate(graphics_cards, start=1):
    max_score = 0.8  # Titan X 的分数
    min_score = 0.2  # RTX 3090Ti 的分数
    score = calculate_score(i, len(graphics_cards), max_score, min_score)
    print(f"{card}: {score}")
