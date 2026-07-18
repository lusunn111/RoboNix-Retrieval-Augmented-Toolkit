
import tensorflow_datasets as tfds
import os
import numpy as np

# Point to the local dataset directory
db_dir = '/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0'
MIN_EPISODE_LENGTH = 5

print(f"Counting steps in dataset at: {db_dir}")

try:
    builder = tfds.builder_from_directory(db_dir)
    ds = builder.as_dataset(split='train')
    
    total_episodes = 0
    total_steps = 0
    skipped_episodes = 0
    
    print("Iterating through episodes...")
    for episode in ds:
        # RLDS datasets usually have a 'steps' dataset inside
        steps = episode['steps']
        # We need to iterate to count, or use cardinality if available/reliable
        # Converting to list is safe for counting as we are not loading heavy images into memory if we don't access them, 
        # but 'steps' is a dataset, so list(steps) might trigger loading.
        # Better to just iterate.
        
        step_count = 0
        for _ in steps:
            step_count += 1
            
        if step_count < MIN_EPISODE_LENGTH:
            skipped_episodes += 1
        else:
            total_episodes += 1
            total_steps += step_count
            
        if (total_episodes + skipped_episodes) % 50 == 0:
            print(f"Processed {total_episodes + skipped_episodes} episodes...")

    print("-" * 30)
    print(f"Dataset: {builder.name}")
    print(f"Total Episodes (Processed): {total_episodes}")
    print(f"Skipped Episodes (< {MIN_EPISODE_LENGTH} steps): {skipped_episodes}")
    print(f"Total Steps (Expected DB Records): {total_steps}")
    print("-" * 30)

except Exception as e:
    print(f"Error counting steps: {e}")
