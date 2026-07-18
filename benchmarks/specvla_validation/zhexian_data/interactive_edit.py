"""
interactive_edit.py

Interactive line chart data editor
- Left click and drag to move data points
- Press 's' to save images and data
- Press 'r' to reset data
- Press 'q' to quit
- Press '1/2/3' to switch views (radius/displacement/weighted)

Usage:
1. Put this script and .npy file in the same directory
2. Run: python interactive_edit.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# ================================
# Config
# ================================
DATA_FILE = "push_the_plate_to_the_front_of_the_stove_both_success_r3_s1_zhexian_data.npy"


class InteractiveEditor:
    def __init__(self, data_file):
        # Load data
        self.data = np.load(data_file, allow_pickle=True).item()
        self.data_file = data_file
        
        # Backup original data for reset
        self.original_data = {
            'r_radius_norm': self.data['r_radius_norm'].copy(),
            's_radius_norm': self.data['s_radius_norm'].copy(),
            'r_displacement_norm': self.data['r_displacement_norm'].copy(),
            's_displacement_norm': self.data['s_displacement_norm'].copy(),
            'r_weighted': self.data['r_weighted'].copy(),
            's_weighted': self.data['s_weighted'].copy(),
        }
        
        # Current view: 1=radius, 2=displacement, 3=weighted
        self.current_view = 1
        self.view_names = {1: 'Curvature Radius', 2: 'Displacement', 3: 'Weighted'}
        
        # Drag state
        self.dragging = False
        self.drag_line = None  # 'r' or 's'
        self.drag_idx = None
        
        # Start index (skip first 5 points)
        self.start_idx = 5
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Interactive Data Editor')
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial plot
        self.update_plot()
        
    def get_current_data(self):
        """Get data for current view"""
        if self.current_view == 1:
            return self.data['r_radius_norm'], self.data['s_radius_norm']
        elif self.current_view == 2:
            return self.data['r_displacement_norm'], self.data['s_displacement_norm']
        else:
            return self.data['r_weighted'], self.data['s_weighted']
    
    def set_current_data(self, r_data, s_data):
        """Set data for current view"""
        if self.current_view == 1:
            self.data['r_radius_norm'] = r_data
            self.data['s_radius_norm'] = s_data
        elif self.current_view == 2:
            self.data['r_displacement_norm'] = r_data
            self.data['s_displacement_norm'] = s_data
        else:
            self.data['r_weighted'] = r_data
            self.data['s_weighted'] = s_data
    
    def update_plot(self):
        """Update plot"""
        self.ax.clear()
        
        r_data, s_data = self.get_current_data()
        
        # Retrieval data
        steps_r = np.arange(len(r_data))
        valid_r = ~np.isnan(r_data) & (steps_r >= self.start_idx)
        self.line_r, = self.ax.plot(
            steps_r[valid_r], r_data[valid_r],
            'b-', linewidth=2, alpha=0.8,
            marker='o', markersize=8,
            markerfacecolor='blue', markeredgecolor='darkblue',
            label='Retrieval', picker=5
        )
        
        # SD data
        steps_s = np.arange(len(s_data))
        valid_s = ~np.isnan(s_data) & (steps_s >= self.start_idx)
        self.line_s, = self.ax.plot(
            steps_s[valid_s], s_data[valid_s],
            color='darkorange', linewidth=2, alpha=0.8,
            marker='o', markersize=8,
            markerfacecolor='darkorange', markeredgecolor='orangered',
            label='SD', picker=5
        )
        
        # Store valid indices mapping
        self.valid_r_indices = steps_r[valid_r]
        self.valid_s_indices = steps_s[valid_s]
        
        # Set figure properties
        self.ax.set_title(
            f'{self.view_names[self.current_view]} (Normalized)\n'
            f'Drag points | 1/2/3=switch view | s=save | r=reset | q=quit',
            fontsize=12
        )
        self.ax.set_xlabel('Time Step', fontsize=11)
        self.ax.set_ylabel('Normalized Value', fontsize=11)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_ylim(-0.1, 1.1)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def find_nearest_point(self, x, y):
        """Find nearest data point"""
        r_data, s_data = self.get_current_data()
        
        min_dist = float('inf')
        nearest_line = None
        nearest_idx = None
        
        # Check Retrieval line
        for i, real_idx in enumerate(self.valid_r_indices):
            val = r_data[real_idx]
            if not np.isnan(val):
                dist = np.sqrt((real_idx - x)**2 + (val - y)**2 * 100)
                if dist < min_dist:
                    min_dist = dist
                    nearest_line = 'r'
                    nearest_idx = real_idx
        
        # Check SD line
        for i, real_idx in enumerate(self.valid_s_indices):
            val = s_data[real_idx]
            if not np.isnan(val):
                dist = np.sqrt((real_idx - x)**2 + (val - y)**2 * 100)
                if dist < min_dist:
                    min_dist = dist
                    nearest_line = 's'
                    nearest_idx = real_idx
        
        # Distance threshold
        if min_dist < 5:
            return nearest_line, nearest_idx
        return None, None
    
    def on_press(self, event):
        """Mouse press event"""
        if event.inaxes != self.ax:
            return
        if event.button != MouseButton.LEFT:
            return
        
        line, idx = self.find_nearest_point(event.xdata, event.ydata)
        if line is not None:
            self.dragging = True
            self.drag_line = line
            self.drag_idx = idx
            print(f"Selected: {'Retrieval' if line == 'r' else 'SD'} point {idx}")
    
    def on_release(self, event):
        """Mouse release event"""
        if self.dragging:
            self.dragging = False
            if event.ydata is not None:
                print(f"Modified: point {self.drag_idx} -> {event.ydata:.4f}")
            self.drag_line = None
            self.drag_idx = None
    
    def on_motion(self, event):
        """Mouse motion event"""
        if not self.dragging:
            return
        if event.inaxes != self.ax:
            return
        if event.ydata is None:
            return
        
        # Clamp y value
        new_y = np.clip(event.ydata, 0, 1)
        
        # Update data
        r_data, s_data = self.get_current_data()
        if self.drag_line == 'r':
            r_data[self.drag_idx] = new_y
        else:
            s_data[self.drag_idx] = new_y
        self.set_current_data(r_data, s_data)
        
        # Update plot
        self.update_plot()
    
    def on_key(self, event):
        """Keyboard event"""
        print(f"Key pressed: {event.key}")  # Debug output
        
        if event.key == 'q':
            print("Exiting editor...")
            plt.close(self.fig)
        elif event.key == 's':
            print("Saving...")
            self.save_all()
        elif event.key == 'r':
            print("Resetting data...")
            self.reset_data()
        elif event.key == '1':
            self.current_view = 1
            self.update_plot()
            print("Switched to: Curvature Radius")
        elif event.key == '2':
            self.current_view = 2
            self.update_plot()
            print("Switched to: Displacement")
        elif event.key == '3':
            self.current_view = 3
            self.update_plot()
            print("Switched to: Weighted")
    
    def reset_data(self):
        """Reset data to original state"""
        for key in self.original_data:
            self.data[key] = self.original_data[key].copy()
        self.update_plot()
        print("Data reset complete!")
    
    def save_all(self):
        """Save data and images"""
        try:
            # Save data
            np.save("modified_data.npy", self.data)
            print("Data saved: modified_data.npy")
            
            # Save all three plots
            self.save_all_plots()
            print("All saves complete!")
        except Exception as e:
            print(f"Save error: {e}")
    
    def save_all_plots(self):
        """Save all three plot types (original style)"""
        # Curvature radius
        self._save_single_plot(
            self.data['r_radius_norm'],
            self.data['s_radius_norm'],
            "final_radius.png",
            marker='^'
        )
        print("Saved: final_radius.png")
        
        # Displacement
        self._save_single_plot(
            self.data['r_displacement_norm'],
            self.data['s_displacement_norm'],
            "final_displacement.png",
            marker='*',
            markersize_r=10,
            markersize_s=12
        )
        print("Saved: final_displacement.png")
        
        # Weighted
        self._save_single_plot(
            self.data['r_weighted'],
            self.data['s_weighted'],
            "final_weighted.png",
            marker='s'
        )
        print("Saved: final_weighted.png")
    
    def _save_single_plot(self, r_data, s_data, filename, marker='^', markersize_r=7, markersize_s=7):
        """Save single plot (original style, no labels)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        steps_r = np.arange(len(r_data))
        valid_r = ~np.isnan(r_data) & (steps_r >= self.start_idx)
        ax.plot(
            steps_r[valid_r], r_data[valid_r],
            'b-', linewidth=4, alpha=0.8,
            marker=marker, markersize=markersize_r,
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1
        )
        
        steps_s = np.arange(len(s_data))
        valid_s = ~np.isnan(s_data) & (steps_s >= self.start_idx)
        ax.plot(
            steps_s[valid_s], s_data[valid_s],
            color='darkorange', linewidth=4, alpha=0.8,
            marker=marker, markersize=markersize_s,
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1
        )
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def run(self):
        """Run editor"""
        print("=" * 60)
        print("Interactive Line Chart Editor")
        print("=" * 60)
        print(f"Data file: {self.data_file}")
        print()
        print("Controls:")
        print("  - Left click and drag to move points")
        print("  - Press '1' for Curvature Radius")
        print("  - Press '2' for Displacement")
        print("  - Press '3' for Weighted")
        print("  - Press 's' to SAVE data and images")
        print("  - Press 'r' to RESET all changes")
        print("  - Press 'q' to QUIT")
        print("=" * 60)
        print()
        print("NOTE: Click on the plot area first to enable keyboard!")
        print()
        
        plt.show()


def main():
    editor = InteractiveEditor(DATA_FILE)
    editor.run()


if __name__ == "__main__":
    main()
