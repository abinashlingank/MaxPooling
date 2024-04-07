import numpy as np

def max_pooling(matrix, window_size):
    m, n = matrix.shape
    k = window_size
    
    # Initialize an empty result matrix
    result = np.zeros((m-k+1, n-k+1))
    
    # Loop through each position of the moving window
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            # Extract the submatrix/window
            window = matrix[i:i+k, j:j+k]
            # Calculate the maximum value in the window
            max_value = np.max(window)
            # Assign the maximum value to the corresponding position in the result matrix
            result[i, j] = max_value
    
    return result

# Test the function with user input
def test_max_pooling():
    # Initialize an empty list to store rows
    rows = []
    print("Enter no of rows:")
    n = int(input())
    # Get input for each row from the user
    for i in range(n):
        row = input(f"Enter values for row {i + 1} separated by spaces: ").split()
        # Convert the input values to integers
        row = [int(val) for val in row]
        rows.append(row)

    # Convert the list of lists to a numpy array
    matrix = np.array(rows)

    print("Matrix entered by the user:")
    print(matrix)

    # Get the window size from the user
    print("Enter window size:")
    window_size = int(input())

    print("Original matrix:")
    print(matrix)
    print()

    print("Result using max_pooling:")
    print(max_pooling(matrix, window_size).astype(int))
    
test_max_pooling()
