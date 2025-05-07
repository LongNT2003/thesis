import pandas as pd  # type: ignore

df = pd.read_csv("review_labels_part7.csv")

total = df["label_correct"].to_list()
correct = len([x for x in total if x == "Correct"]) / len(total)

print(correct * 100)
