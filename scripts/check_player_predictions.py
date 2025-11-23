import pandas as pd

# Load data
file_path = "/workspace1/gonzalo.fuentes/BielsIA/reports/benchmark/transformer_graphormer/predictions_2024_transformer_graphormer.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Helper function to print player stats
def print_player_stats(player_name, df_row):
    print(f"\n--- {player_name} ({df_row['team']}) ---")
    cols = ['Goals', 'xG', 'Assists', 'Tackles', 'Minutes']
    for col in cols:
        true_val = df_row.get(f'True_{col}', 0)
        pred_val = df_row.get(f'Pred_{col}', 0)
        diff = pred_val - true_val
        print(f"{col}: Real={true_val:.2f}, Pred={pred_val:.2f}, Diff={diff:.2f}")

# 1. Haaland
haaland = df[df['player'].str.contains("Haaland", case=False, na=False)]
if not haaland.empty:
    print_player_stats(haaland.iloc[0]['player'], haaland.iloc[0])
else:
    print("\nHaaland not found.")

# 2. Arsenal Player
arsenal = df[df['team'].str.contains("Arsenal", case=False, na=False)]
if not arsenal.empty:
    # Try to find Saka, Odegaard, or Rice
    stars = ["Saka", "Odegaard", "Rice", "Martinelli"]
    found = False
    for star in stars:
        player = arsenal[arsenal['player'].str.contains(star, case=False, na=False)]
        if not player.empty:
            print_player_stats(player.iloc[0]['player'], player.iloc[0])
            found = True
            break
    if not found:
        print_player_stats(arsenal.iloc[0]['player'], arsenal.iloc[0])
else:
    print("\nArsenal team not found.")

# 4. Check specific outliers
print("\n--- OUTLIER ANALYSIS ---")
outliers = ["Gonzalo García", "Daniel Bentley", "Takehiro Tomiyasu"]
for name in outliers:
    player = df[df['player'].str.contains(name, case=False, na=False)]
    if not player.empty:
        print_player_stats(player.iloc[0]['player'], player.iloc[0])
    else:
        print(f"\n{name} not found.")
