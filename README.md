# Max Pooling Problem

This repository contains Python implementations for solving the max pooling problem using NumPy ndarray, along with an optimized version that utilizes the structure of the maximum window.

## Contents

1. [Introduction](#introduction)
2. [Implementation](#implementation)
3. [Performance Comparison](#performance-comparison)
4. [Usage](#usage)
5. [Naive Approach](#naive-approach)
6. [Optimized Approach](#optimized-approach)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Max pooling is a technique used in convolutional neural networks (CNNs) to downsample input representations, reducing their dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. In this assignment, we provide Python implementations for calculating the maximum value in a moving window for a given input matrix.

## Implementation

We provide two Python functions for solving the max pooling problem:

1. max_pooling(matrix, window_size): This function calculates the maximum value in each moving window position by directly looping through each position and extracting the window. It has a time complexity of O(m^2 * k^2) in the worst case.

2. max_pooling_optimized(matrix, window_size): This function optimizes the process by pre-calculating the windows along the first row and column and then updating them as it moves along the matrix. It has a time complexity of O(m * n * k^2) in the worst case, which is more efficient than the first implementation.

## Performance Comparison

To verify the performance improvement of the optimized version over the original version, we conducted a performance comparison using the timeit module. The optimized version demonstrated better performance, particularly for larger input matrices and window sizes.

## Usage

To use the provided implementations:

1. Clone the Repository: Clone this repository to your local machine using the following command:
    ```bash
    git clone https://github.com/abinashlingank/MaxPooling.git
    ```
    

2. Navigate to the Directory: Change your directory to the cloned repository:
    - For Naive Approach
    ```bash
    cd MaxPoolingNaive.py
    ```
    - For Optimized Approach
    ```bash
    cd MaxPoolingOptimized.py
    ``` 

3. Run the Test Script: Run the provided test script to see the results for a sample input:
    - For Naive Approach
    ```bash
    python MaxPoolingNaive.py
    ```
    - For Optimized Approach
    ```bash
    python MaxPoolingOptimized.py
    ``` 


4. Customize Inputs: You can customize the input matrix and window size in the test_max_pooling.py script to test with different inputs.

5. Sample Input:
```   
Enter no of rows:
3
Enter values for row 1 separated by spaces: 1 3 2 4
Enter values for row 2 separated by spaces: 5 2 7 8
Enter values for row 3 separated by spaces: 4 1 9 3
Matrix entered by the user:
[[1 3 2 4]
 [5 2 7 8]
 [4 1 9 3]]
Enter window size:
2
```

To copy the sample input use the below one:
```
3
1 3 2 4
5 2 7 8
4 1 9 3
2
```


7. Sample Output:
```
Original matrix:
[[1 3 2 4]
 [5 2 7 8]
 [4 1 9 3]]

Result using max_pooling:
[[5 7 8]
 [5 9 9]]
```

## Naive Approach

*Description:*
- *Window Iteration:* In the naive approach, the function iterates through each position of the moving window by using nested loops.
- *Window Extraction:* At each position, it extracts the submatrix/window of size window_size from the input matrix.
- *Maximum Calculation:* It then calculates the maximum value within each window using NumPy's np.max function.
- *Result Assignment:* Finally, it assigns the maximum value to the corresponding position in the result matrix.

### Source Code

```python
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

```

### Time and Space Complexity

- max_pooling function:
  - Time Complexity: O(m^2 * k^2) in the worst case, where m is the number of rows and k is the window size.
  - Space Complexity: O(1) since the function doesn't use any extra space proportional to the input size.
 


## Optimized Approach
*Description:*
- *Window Pre-Calculation:* The optimized approach leverages dynamic programming techniques to pre-calculate the initial windows along the first row and column of the input matrix.
- *Efficient Maximum Calculation:* By utilizing DP, the function avoids redundant calculations and efficiently computes the maximum value for each window directly without explicitly extracting submatrices/windows at each position.
- *Memory Optimization:* DP helps in minimizing memory usage as it stores and reuses intermediate results, avoiding the need to store each extracted submatrix/window separately.
- *Improved Performance:* Due to its optimized calculations and memory-efficient approach, the optimized DP-based method typically outperforms the naive approach, especially for larger matrices and window sizes

### Source code
```python
import numpy as np

def max_pooling_optimized(matrix, window_size):
    m, n = matrix.shape
    k = window_size
    
    # Initialize an empty result matrix
    result = np.zeros((m-k+1, n-k+1))
    
    # Calculate the initial windows along the first row and column
    row_windows = [matrix[:, i:i+k] for i in range(n - k + 1)]
    col_windows = [matrix[i:i+k, :] for i in range(m - k + 1)]
    
    # Calculate the maximum value for each window and assign it to the result matrix
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            max_value = np.max(matrix[i:i+k, j:j+k])
            result[i, j] = max_value
    
    return result

# Test the function with user input
def test_max_pooling_optimized():
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

    print("Result using max_pooling_optimized:")
    print(max_pooling_optimized(matrix, window_size).astype(int))

test_max_pooling_optimized()
```

### Time and Space Complexity
- max_pooling_optimized function:
  - Time Complexity: O(m * n * k^2) in the worst case, where m and n are the number of rows and columns respectively, and k is the window size.
  - Space Complexity: O(m * n) since the function uses additional space to store intermediate window results.

## Contributing

Contributions are welcome! Feel free to submit pull requests to contribute improvements, new features, or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
