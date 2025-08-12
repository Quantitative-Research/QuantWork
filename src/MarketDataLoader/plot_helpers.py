import os
import matplotlib.pyplot as plt

def show_plot():
    """
    Show the plot only if we're not running inside pytest.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        plt.show()
    else:
        plt.close()
