import numpy as np
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def compute_entropy(y):

    prob = 0

    if len(y) == 0:
        return 0
    
    prob = sum(y[y==1])/len(y)
    
    if prob == 0 or prob == 1:
        return 0
    
    else:
        return -prob*np.log2(prob) - (1-prob)*np.log2(1-prob)


# ==========================================
# 1. THE ENTROPY PLOTTER
# ==========================================    
   
def plot_entropy(p_target):
    # 1. Generate the base curve
    prob = np.linspace(0.001, 0.999, 100) # Avoid 0 and 1 to prevent log errors
    entropy = -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
    
    # 2. Calculate entropy for the specific slider value
    target_entropy = -p_target * np.log2(p_target) - (1 - p_target) * np.log2(1 - p_target) if 0 < p_target < 1 else 0
    
    # 3. Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(prob, entropy, label='Entropy Curve', color='blue', alpha=0.6)
    
    # Add the interactive "Point"
    plt.scatter(p_target, target_entropy, color='red', s=100, zorder=5)
    plt.annotate(f'  H = {target_entropy:.3f}', (p_target, target_entropy), fontsize=12, fontweight='bold')
    
    plt.axvline(p_target, color='red', linestyle='--', alpha=0.3)
    plt.xlabel('Probability of Class 1 ($p$)')
    plt.ylabel('Entropy $H(p)$')
    plt.title(f'Entropy at p={p_target:.2f} is {target_entropy:.3f}')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.show()


# ==========================================
# 2. THE DECISION TREE BUILDER
# ========================================== 

# splitting function for binary features (0 or 1)
def split_indices(X, feature_index):
    """Splits data into left (1) and right (0) based on a feature."""
    left_indices = np.where(X[:, feature_index] == 1)[0]
    right_indices = np.where(X[:, feature_index] == 0)[0]
    return left_indices, right_indices

# ==========================================
#  information gain function, we need to calculate the weighted average of the child entropies and subtract it from the parent entropy  
# ==========================================

def information_gain(X, y, left_indices, right_indices):
    """Calculates the information gain of a split."""
    p_node = compute_entropy(y)
    w_left = len(left_indices) / len(y)
    w_right = len(right_indices) / len(y)
    p_left = compute_entropy(y[left_indices])
    p_right = compute_entropy(y[right_indices])
    return p_node - (w_left * p_left + w_right * p_right)

# ==========================================
# GET THE BEST FEATURE TO SPLIT ON
# ==========================================    
def get_best_split(X, y):
    """Loops through all features to find the one with the highest IG."""
    best_feature = -1
    max_gain = -1
    
    for i in range(X.shape[1]):
        left_indices, right_indices = split_indices(X, i)
        
        # Skip this split if it doesn't divide the data at all
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue
            
        gain = information_gain(X, y, left_indices, right_indices)
        
        if gain > max_gain:
            max_gain = gain
            best_feature = i
            
    return best_feature, max_gain

# ==========================================
# 2. THE RECURSIVE TREE BUILDER
# ==========================================
def build_tree(X, y, depth=0, max_depth=3, feature_names=None):
    """
    Recursively builds the decision tree.
    Returns a dictionary representing a node.
    """
    # BASE CASE 1: The node is 100% pure (only 1 class left)
    if len(np.unique(y)) == 1:
        return {'is_leaf': True, 'prediction': y[0]}
        
    # BASE CASE 2: We hit the maximum depth limit
    if depth >= max_depth:
        # Predict the majority class in this bucket
        prediction = np.bincount(y).argmax()
        return {'is_leaf': True, 'prediction': prediction}

    # RECURSIVE STEP 1: Find the best feature
    best_feature, max_gain = get_best_split(X, y)

    # BASE CASE 3: No feature provides any Information Gain
    if max_gain <= 0 or best_feature == -1:
        prediction = np.bincount(y).argmax()
        return {'is_leaf': True, 'prediction': prediction}

    # RECURSIVE STEP 2: Perform the actual split
    left_indices, right_indices = split_indices(X, best_feature)

    # RECURSIVE STEP 3: Call this exact same function on the children!
    # Notice we pass the subsets of X and y, and increase the depth by 1
    left_branch = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth, feature_names)
    right_branch = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth, feature_names)

    # Return the current node, linking it to its children
    feature_name = feature_names[best_feature] if feature_names else best_feature
    return {
        'is_leaf': False,
        'feature_index': best_feature,
        'feature_name': feature_name,
        'left': left_branch,
        'right': right_branch
    }

# A simple helper function to print the dictionary nicely
def print_tree(node, spacing=""):
    if node['is_leaf']:
        print(spacing + f"Predict: {node['prediction']}")
        return

    print(spacing + f"Split on: {node['feature_name']} (Index {node['feature_index']})")
    print(spacing + "--> True (Left):")
    print_tree(node['left'], spacing + "  ")
    print(spacing + "--> False (Right):")
    print_tree(node['right'], spacing + "  ")