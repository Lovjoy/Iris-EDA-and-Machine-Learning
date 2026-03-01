import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution=0.02,title="Decision Regions"):
    
    # Initialise the marker types and colors
    markers = ('o','s','D')
    colors = ('blue', 'orange', 'green')
    color_Map = ListedColormap(colors[:len(np.unique(y))])
    
    # Convert string labels to numeric if needed
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    
    # Parameters for the graph and decision surface
    x1_min = X[:,0].min() - 1
    x1_max = X[:,0].max() + 1
    x2_min = X[:,1].min() - 1
    x2_max = X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # Convert predicted labels to numeric

    Z_numeric = np.array([label_map[label] for label in Z])
    Z_numeric = Z_numeric.reshape(xx1.shape)
    
    plt.contourf(xx1,xx2,Z_numeric,alpha=0.4,cmap=color_Map)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx, cl in enumerate(unique_labels):
        plt.scatter(x = X[y_numeric == idx, 0], y = X[y_numeric == idx, 1],
                    alpha = 0.8, color = colors[idx],
                    marker = markers[idx], label = cl
                   )
    
    plt.title(title)
    plt.legend()
