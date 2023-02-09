# -*- coding: utf-8 -*-

from prompt import Prompting

model_path = "bert-base-uncased"
prompting = Prompting(model=model_path)

template = "Savaş went to Laris to visit the parliament. "
vervlize_per = ["person", "man"]
vervlize_pos = ["location", "city", "place"]

pred_score1 = prompting.compute_tokens_prob(template + "Savaş is a type of [MASK].", token_list1=vervlize_per,
                                            token_list2=vervlize_pos)
print(pred_score1)

pred_score2 = prompting.compute_tokens_prob(template + "Laris is a type of [MASK].", token_list1=vervlize_per,
                                            token_list2=vervlize_pos)
print(pred_score2)
