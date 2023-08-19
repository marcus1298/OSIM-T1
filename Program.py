import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -(1.4 - 3.0 * x) * np.sin(18.0 * x)

def f_prime(x):
    return (1.4 - 3.0 * x) * 18.0 * np.cos(18.0 * x) - 3.0 * np.sin(18.0 * x)

def newton_raphson(initial_guess, tolerance, max_iterations):
    x = initial_guess
    points = [(x, f(x))]  # Lista para armazenar os pontos (x, f(x))
    for i in range(max_iterations):
        x_new = x - f(x) / f_prime(x)
        points.append((x_new, f(x_new)))  # Adicionar ponto à lista
        if abs(x_new - x) < tolerance:
            return x_new, points  # Retorna o zero encontrado e os pontos calculados
        x = x_new
    return None, points  # Retorna None se o método não convergir

def plot_near_zero_points(points, max_points=8):
    # Ordenar pontos por distância crescente de x
    points.sort(key=lambda point: abs(point[0] - round(point[0])))
    # Pegar os primeiros max_points pontos
    selected_points = points[:max_points]
    for x, y in selected_points:
        plt.plot(x, y, 'ro')  # Pintar o ponto de vermelho

r_min, r_max = 0.0, 1.2

# Encontrar os 8 zeros a partir de diversos pontos iniciais
num_zeros = 8
initial_guesses = np.linspace(r_min, r_max, num_zeros * 50)  # Aumente a quantidade de palpites
roots = []

for guess in initial_guesses:
    root, _ = newton_raphson(guess, 1e-6, 1000)
    if root is not None and r_min <= root <= r_max:
        roots.append(root)

print("Zeros encontrados:")
for root in roots:
    print(f"  Zero em x = {root:.6f}")

# Plotagem do gráfico com pontos vermelhos nos zeros e próximos de zero
inputs = np.arange(r_min, r_max, 0.01)
results = f(inputs)
points = [(x, f(x)) for x in inputs]  # Calcular todos os pontos (x, f(x)) para plotagem

plt.plot(inputs, results)  # Plotar a função f(x)
plt.axhline(y=0, ls='--', color='black')  # Adicionar linha horizontal em y = 0
plt.plot(roots, [0] * len(roots), 'ro')  # Plotar os pontos vermelhos nos zeros encontrados

near_zero_points = []
for x in np.arange(r_min, r_max, 0.001):
    y = f(x)
    if -0.01 <= y <= 0.01:
        near_zero_points.append((x, y))  # Adicionar ponto à lista

# Plotar os pontos próximos de zero selecionados
plot_near_zero_points(near_zero_points, max_points=8)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gráfico da Função e Zeros')
plt.show()

"""  
  Ponto próximo de zero: x = 0.000000, f(x) = -0.000000
  Ponto próximo de zero: x = 0.175000, f(x) = 0.007356 
  Ponto próximo de zero: x = 0.349000, f(x) = 0.000418 
  Ponto próximo de zero: x = 0.466000, f(x) = -0.001722
  Ponto próximo de zero: x = 0.524000, f(x) = -0.001242
  Ponto próximo de zero: x = 0.698000, f(x) = -0.001645
  Ponto próximo de zero: x = 0.873000, f(x) = -0.007359
  Ponto próximo de zero: x = 1.047000, f(x) = -0.006191
  """