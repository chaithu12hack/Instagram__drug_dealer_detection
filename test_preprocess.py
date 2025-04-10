from utils.preprocess import load_and_preprocess

X, y = load_and_preprocess("dataset/posts.csv")

for i in range(len(X)):
    print(f"Cleaned Text: {X[i]}")
    print(f"Label: {y[i]}")
    print("-" * 40)
print("✅ Preprocessing complete.")