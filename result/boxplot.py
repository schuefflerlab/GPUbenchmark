
import numpy as np
import matplotlib.pyplot as plt


# Function to read numpy array from file
def read_numpy_array(file_path):
    return np.load(file_path)


# Function to plot boxplot
def plot_and_save_boxplot(data1, data2, save_path):
    fig, ax = plt.subplots()
    ax.boxplot([data1, data2])

    ax.set_xticklabels(['RTX3090', 'A4500'])
    ax.set_ylabel('Time / [s]')
    ax.set_title('Training Time on PatchCam per GPU')
    bot,top = ax.get_ylim()
    #print(bot)
    ax.set_ylim(bot,top,auto=True)

    # Save the boxplot to the specified path
    plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    # Replace 'path_to_array1.npy' and 'path_to_array2.npy' with the actual file paths
    array1_path = '3090_run=10.npy'
    array2_path = 'A4500_run=10.npy'

    # Read numpy arrays from files
    array1 = read_numpy_array(array1_path)
    array2 = read_numpy_array(array2_path)
    #print(array2.shape)

    # Ensure that the arrays have the correct shape [1, 10]
    if array1.shape == (10,) and array2.shape == (10,):
        save_path = 'PatchCam_4500vs3090.png'
        # Plot and save the boxplot
        plot_and_save_boxplot(array1[:], array2[:], save_path)
    else:
        print("Error: Arrays should have shape [10,].")