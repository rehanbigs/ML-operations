# test_cases.py
import numpy as np
import matplotlib.pyplot as plt

def test_initialize_parameters_deep(func):
    layer_dims = [4, 3, 2, 1]
    params = func(layer_dims)
    assert params["W1"].shape == (3, 4), "W1 shape incorrect"
    assert params["W2"].shape == (2, 3), "W2 shape incorrect"
    assert params["W3"].shape == (1, 2), "W3 shape incorrect"
    print("Test passed!")

def test_normalize_array(func):
    arr = np.array([1, 2, 3, 4, 5])
    norm_arr = func(arr)
    assert np.all((norm_arr >= 0) & (norm_arr <= 1)), "Normalization failed"
    print("Test passed!")

def test_leaky_relu(func):
    Z = np.array([[-1, 2, -3], [4, -5, 6]])
    A = func(Z)
    expected_output = np.array([[-0.01, 2, -0.03], [4, -0.05, 6]])  # Assuming alpha = 0.01
    assert np.array_equal(A, expected_output), "Leaky ReLU incorrect"
    print("Test passed!")

def test_linear_activation_forward(func):
    A_prev = np.random.randn(4, 5)
    W = np.random.randn(3, 4) * 0.01
    b = np.zeros((3, 1))
    A, Z = func(A_prev, W, b, "relu")
    assert A.shape == (3, 5), "Forward activation incorrect"
    print("Test passed!")

def test_gradient_descent(func):
    w = np.array([0.5, -0.3])
    grad = np.array([0.1, -0.2])
    updated_w = func(w, grad)
    assert np.all(updated_w == w - 0.01 * grad), "Gradient descent update incorrect"
    print("Test passed!")

def test_logistic_cost(func):
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    expected_cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(func(y_true, y_pred), expected_cost), "Cost function calculation incorrect"
    print("Test passed!")

def test_elementwise_multiply(func):
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    assert np.array_equal(func(arr1, arr2), arr1 * arr2), "Multiplication incorrect"
    print("Test passed!")

def test_plot_decision_boundary(func):
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)
    assert isinstance(func(X, y), plt.Figure), "Plot function incorrect"
    print("Test passed!")

def test_identity_matrix(func):
    assert np.array_equal(func(), np.eye(5)), "Identity matrix incorrect"
    print("Test passed!")

def test_dot_product(func):
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    assert np.array_equal(func(mat1, mat2), np.dot(mat1, mat2)), "Dot product incorrect"
    print("Test passed!")
