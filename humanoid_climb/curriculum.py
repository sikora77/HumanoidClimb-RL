import numpy as np
import pandas as pd

_GLOBAL_CURRICULUM = None


def get_curriculum(csv_path, **kwargs):
    global _GLOBAL_CURRICULUM
    if _GLOBAL_CURRICULUM is None:
        _GLOBAL_CURRICULUM = Curriculum(csv_path, **kwargs)
    return _GLOBAL_CURRICULUM


class Curriculum:
    def __init__(self, csv_path, initial_difficulty_percentile=10, max_difficulty_percentile=100, scaling_steps=5000000):
        """
        Loads the climbs CSV ONCE, pre-parses route data into lightweight python dicts,
        and deletes the DataFrame to save RAM and avoid redundant disk loads across threads/workers.
        """
        print(f"[Curriculum] Single-loading curriculum data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Filter for 0 angle routes
        df_zero = df[df['angle'] == 0].copy()
        df_zero = df_zero.sort_values(by=['difficulty', 'name']).reset_index(drop=True)
        
        self.scaling_steps = scaling_steps
        self.initial_percentile = initial_difficulty_percentile
        self.max_percentile = max_difficulty_percentile
        
        # Pre-parse into lightweight native dicts
        self.routes = []
        for _, row in df_zero.iterrows():
            try:
                hx = [int(x) for x in str(row['holes_x']).split(',')]
                hy = [int(y) for y in str(row['holes_y']).split(',')]
                roles = [int(r) for r in str(row['role_ids']).split(',')]
                self.routes.append({
                    'name': str(row['name']),
                    'difficulty': float(row['difficulty']),
                    'frames': str(row['frames']),
                    'holes_x': hx,
                    'holes_y': hy,
                    'role_ids': roles
                })
            except Exception:
                continue

        del df
        del df_zero
        print(f"[Curriculum] Successfully cached {len(self.routes)} routes in memory.")

    def get_route(self, cumulative_steps):
        """
        Samples a route based on the current curriculum progress.
        """
        progress = min(1.0, cumulative_steps / self.scaling_steps)
        current_percentile = self.initial_percentile + progress * (self.max_percentile - self.initial_percentile)
        
        max_idx = max(1, int((current_percentile / 100.0) * len(self.routes)))
        sampled_idx = np.random.randint(0, max_idx)
        return self.routes[sampled_idx]
