import pandas as pd
import matplotlib.pyplot as pt

pt.rcParams['font.family'] = 'serif'
pt.rcParams['font.size'] = 12
pt.rcParams['pdf.fonttype'] = 42

csv = pd.read_csv("ruins_log_dir/PPO_log_ruins.csv")
print(csv)

trend_x = csv["timestep"].rolling(240).mean()
trend_y = csv["reward"].rolling(240).mean()

pt.figure(figsize=(3,3))
pt.plot(csv["timestep"], csv["reward"], '-', color=(.8,)*3)
pt.plot(trend_x, trend_y, 'k-')
pt.plot([8136000]*2, [csv["reward"].min(), csv["reward"].max()], 'k:')
pt.xlabel("Total Timesteps")
pt.ylabel("Episode Reward")
pt.tight_layout()
pt.savefig("ruins_ppo.pdf")
pt.show()
