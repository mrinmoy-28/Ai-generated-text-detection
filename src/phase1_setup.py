# src/phase1_setup.py
import os
import pandas as pd
from datasets import load_dataset

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_prepare_hc3():
    print("📥 Loading HC3 dataset (may take a few minutes)...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")

    texts  = []
    labels = []

    for item in dataset['train']:

        # Human answers → label 0
        for ans in item['human_answers']:
            if ans and len(ans.strip()) > 50:
                texts.append(ans.strip())
                labels.append(0)

        # AI answers → label 1
        for ans in item['chatgpt_answers']:
            if ans and len(ans.strip()) > 50:
                texts.append(ans.strip())
                labels.append(1)

    # Build dataframe
    df = pd.DataFrame({'text': texts, 'label': labels})

    # Remove duplicates
    df = df.drop_duplicates(subset='text').reset_index(drop=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "hc3_cleaned.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to {output_path}")
    print(f"   Total samples : {len(df)}")
    print(f"   Human (0)     : {(df.label == 0).sum()}")
    print(f"   AI    (1)     : {(df.label == 1).sum()}")

    return df


def explore_data(df):
    print("\n📊 Dataset Overview:")
    print(f"   Shape         : {df.shape}")
    print(f"   Columns       : {df.columns.tolist()}")
    print(f"\n   Sample Human text:\n   → {df[df.label==0].iloc[0]['text'][:200]}")
    print(f"\n   Sample AI text:\n   → {df[df.label==1].iloc[0]['text'][:200]}")

    # Check average text length
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    print(f"\n   Avg word count (Human) : {df[df.label==0]['text_length'].mean():.1f}")
    print(f"   Avg word count (AI)    : {df[df.label==1]['text_length'].mean():.1f}")


if __name__ == "__main__":

    # Step 1 — download and clean dataset
    df = load_and_prepare_hc3()

    # Step 2 — explore and understand data
    explore_data(df)

    print("\n✅ Phase 1 Complete! Run phase2_statistical.py next.")