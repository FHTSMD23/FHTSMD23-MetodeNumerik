import numpy as np

# Membuat Matriks koefisien dan vektor konstanta
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)

b = np.array([6, 2, 10], dtype=float)

# Membuat sistem metode eliminasi Gauss
def gauss_elimination(A, b):
    n = len(b)
    # menggabungkan A dan b ke dalam satu matriks
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Melakukan proses eliminasi
    for i in range(n):
        # Buat elemen diagonal menjadi 1 dan mengeliminasi dibawahnya
        for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    # Proses Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Eliminasi Gauss-Jordan
def gauss_jordan(A, b):
    n = len(b)
    # Gabungkan A dan b ke dalam satu matriks
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Proses eliminasi
    for i in range(n):
        # Buat elemen diagonal menjadi 1
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]

    # Solusi
    return Ab[:, -1]

# Menjalankan kedua metode Gauss
solution_gauss = gauss_elimination(A, b)
solution_gauss_jordan = gauss_jordan(A, b)

# Menampilkan hasil dari kedua metode Eliminasi
print("Solusi dengan Eliminasi Gauss:")
print(f"x1 = {solution_gauss[0]}, x2 = {solution_gauss[1]}, x3 = {solution_gauss[2]}")

print("\nSolusi dengan Eliminasi Gauss-Jordan:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")
