from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
print(digits.data)
print(digits.images[0])
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
