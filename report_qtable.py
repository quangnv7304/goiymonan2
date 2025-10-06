import json
import os
import time
from statistics import mean, median

PATH = 'sarsa_table.json'

def main():
    if not os.path.exists(PATH):
        print('No Q-table found at', PATH)
        return
    size = os.path.getsize(PATH)
    mtime = os.path.getmtime(PATH)
    with open(PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    states = len(data)
    actions_counts = [len(v) for v in data.values()]
    avg_actions = mean(actions_counts) if actions_counts else 0
    med_actions = median(actions_counts) if actions_counts else 0

    # collect top Q-values
    xs = []
    for s, v in data.items():
        for a, val in v.items():
            try:
                xs.append((float(val), s, a))
            except Exception:
                pass
    xs.sort(reverse=True, key=lambda t: t[0])

    print('Q-table path:', PATH)
    print('File size (bytes):', size)
    print('Last modified:', time.ctime(mtime))
    print('Number of states:', states)
    print('Actions per state: avg={:.2f}, med={}'.format(avg_actions, med_actions))
    print('\nTop 10 Q-values:')
    for val, s, a in xs[:10]:
        print(f'  Q={val:.6f}  action={a}  state_preview="{s[:120]}"')

    print('\nSample 5 states:')
    i = 0
    for s, v in list(data.items())[:5]:
        print(' STATE:', s[:120])
        print('  actions:', {k: float(vv) for k, vv in v.items()})
        i += 1

if __name__ == '__main__':
    main()
