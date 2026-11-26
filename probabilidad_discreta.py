"""
UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO
FACULTAD DE INGENIERÍA 2027-1
ESTRUCTURAS DISCRETAS
PROYECTO: PROBABILIDAD DISCRETA
ANZURES LOPEZ JOSHUA GAEL
318057561
"""
import random
import json
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Lanza un dado y reistra la f relativa acumulada de obtener águila
def simular_convergencia_dado(n_max=20000, cara_objetivo=6, semilla=42):
    random.seed(semilla)
    aciertos = 0
    frecuencias = []
    paso = 50
    conteos_x = []
    for i in range(1, n_max + 1):
        resultado = random.randint(1, 6)
        if resultado == cara_objetivo:
            aciertos += 1
        if i % paso == 0:
            frecuencias.append(aciertos / i)
            conteos_x.append(i)
    return conteos_x, frecuencias


def simular_convergencia_moneda(n_max=20000, semilla=7):
    random.seed(semilla)
    aguilas = 0
    frecuencias = []
    paso = 50
    conteos_x = []
    for i in range(1, n_max + 1):
        resultado = random.randint(0, 1)  # 1 = aguila
        if resultado == 1:
            aguilas += 1
        if i % paso == 0:
            frecuencias.append(aguilas / i)
            conteos_x.append(i)
    return conteos_x, frecuencias


def graficar_convergencia(conteos_dado, frec_dado, conteos_moneda, frec_moneda,
                           archivo_salida):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(conteos_dado, frec_dado, color="#3D5A80", linewidth=1.2,
                 label="Frecuencia empírica P(sacar 6)")
    axes[0].axhline(1 / 6, color="#E0A458", linestyle="--", linewidth=1.5,
                     label="Probabilidad teórica = 1/6")
    axes[0].set_title("Convergencia: dado (cara = 6)")
    axes[0].set_xlabel("Número de lanzamientos")
    axes[0].set_ylabel("Frecuencia relativa")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(conteos_moneda, frec_moneda, color="#3D5A80", linewidth=1.2,
                 label="Frecuencia empírica P(águila)")
    axes[1].axhline(0.5, color="#E0A458", linestyle="--", linewidth=1.5,
                     label="Probabilidad teórica = 1/2")
    axes[1].set_title("Convergencia: moneda (águila)")
    axes[1].set_xlabel("Número de lanzamientos")
    axes[1].set_ylabel("Frecuencia relativa")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=150)
    plt.close(fig)

#Calcula la funcion de masa de probabilidad teórica de la suma de 2 dados
def pmf_suma_dos_dados():
    conteos = Counter()
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            conteos[d1 + d2] += 1
    total = 36
    pmf = {x: conteos[x] / total for x in range(2, 13)}
    return pmf


def esperanza_varianza(pmf):
    esperanza = sum(x * p for x, p in pmf.items())
    segundo_momento = sum((x ** 2) * p for x, p in pmf.items())
    varianza = segundo_momento - esperanza ** 2
    return esperanza, varianza

#Simulación de MonteCarlo
def simular_suma_dos_dados(n=100000, semilla=123):
    random.seed(semilla)
    muestras = [random.randint(1, 6) + random.randint(1, 6) for _ in range(n)]
    conteo = Counter(muestras)
    pmf_empirica = {x: conteo[x] / n for x in range(2, 13)}
    media_empirica = statistics.mean(muestras)
    varianza_empirica = statistics.pvariance(muestras)
    return pmf_empirica, media_empirica, varianza_empirica


def graficar_pmf(pmf_teorica, pmf_empirica, archivo_salida):
    xs = list(range(2, 13))
    teo = [pmf_teorica[x] for x in xs]
    emp = [pmf_empirica[x] for x in xs]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ancho = 0.35
    ax.bar([x - ancho / 2 for x in xs], teo, width=ancho, color="#3D5A80",
           label="pmf teórica")
    ax.bar([x + ancho / 2 for x in xs], emp, width=ancho, color="#E0A458",
           label="Frecuencia empírica (100,000 simulaciones)")
    ax.set_title("Suma de dos dados: pmf teórica vs. simulada")
    ax.set_xlabel("x (suma de los dos dados)")
    ax.set_ylabel("P(X = x)")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=150)
    plt.close(fig)

#Problema teórico de la enfermedad ficticia y la prueba médica 99% preceisa
def teorema_bayes_prueba_medica(prevalencia=0.01, sensibilidad=0.99,
                                 falsos_positivos=0.05):
    p_b = prevalencia
    p_no_b = 1 - prevalencia
    p_a_dado_b = sensibilidad
    p_a_dado_no_b = falsos_positivos

    p_a = p_a_dado_b * p_b + p_a_dado_no_b * p_no_b 
    p_b_dado_a = (p_a_dado_b * p_b) / p_a

    return {
        "P(B) [prevalencia]": p_b,
        "P(A|B) [sensibilidad]": p_a_dado_b,
        "P(A|no B) [falsos positivos]": p_a_dado_no_b,
        "P(A) [probabilidad total]": p_a,
        "P(B|A) [resultado de Bayes]": p_b_dado_a,
    }

#Simula 1,000,000 de personas para verificar el resultado teórico por conteo directo de f
def verificar_bayes_con_simulacion(n=1_000_000, prevalencia=0.01,
                                    sensibilidad=0.99, falsos_positivos=0.05,
                                    semilla=99):
    random.seed(semilla)
    positivos = 0
    positivos_y_enfermos = 0
    for _ in range(n):
        enfermo = random.random() < prevalencia
        if enfermo:
            dio_positivo = random.random() < sensibilidad
        else:
            dio_positivo = random.random() < falsos_positivos
        if dio_positivo:
            positivos += 1
            if enfermo:
                positivos_y_enfermos += 1
    return positivos_y_enfermos / positivos if positivos else None

#Saca las impresiones legibles para la consola
def main():
    resultados = {}
    conteos_dado, frec_dado = simular_convergencia_dado()
    conteos_moneda, frec_moneda = simular_convergencia_moneda()
    graficar_convergencia(conteos_dado, frec_dado, conteos_moneda, frec_moneda,
                           "convergencia.png")
    resultados["dado_frecuencia_final"] = frec_dado[-1]
    resultados["moneda_frecuencia_final"] = frec_moneda[-1]
    resultados["dado_n_lanzamientos"] = conteos_dado[-1]
    resultados["moneda_n_lanzamientos"] = conteos_moneda[-1]

    pmf_teorica = pmf_suma_dos_dados()
    esperanza_teo, varianza_teo = esperanza_varianza(pmf_teorica)
    pmf_empirica, media_emp, varianza_emp = simular_suma_dos_dados()
    graficar_pmf(pmf_teorica, pmf_empirica, "pmf_suma_dados.png")

    resultados["pmf_teorica"] = pmf_teorica
    resultados["esperanza_teorica"] = esperanza_teo
    resultados["varianza_teorica"] = varianza_teo
    resultados["esperanza_empirica"] = media_emp
    resultados["varianza_empirica"] = varianza_emp
    
    bayes = teorema_bayes_prueba_medica()
    bayes_simulado = verificar_bayes_con_simulacion()
    resultados["bayes_teorico"] = bayes
    resultados["bayes_simulado"] = bayes_simulado

    print("=" * 70)
    print("LEY DE LOS GRANDES NUMEROS")
    print("=" * 70)
    print(f"Dado  - tras {resultados['dado_n_lanzamientos']:,} lanzamientos: "
          f"frecuencia de '6' = {resultados['dado_frecuencia_final']:.4f} "
          f"(teórico = {1/6:.4f})")
    print(f"Moneda - tras {resultados['moneda_n_lanzamientos']:,} lanzamientos: "
          f"frecuencia de 'águila' = {resultados['moneda_frecuencia_final']:.4f} "
          f"(teórico = {0.5:.4f})")

    print("\n" + "=" * 70)
    print("pmf DE X = SUMA DE DOS DADOS")
    print("=" * 70)
    print(f"{'x':>3} | {'p(x) teórica':>13} | {'p(x) empírica':>13}")
    for x in range(2, 13):
        print(f"{x:>3} | {pmf_teorica[x]:>13.4f} | {pmf_empirica[x]:>13.4f}")
    print(f"\nE[X] teórico   = {esperanza_teo:.4f}   |  E[X] empírico   = {media_emp:.4f}")
    print(f"Var(X) teórico = {varianza_teo:.4f}   |  Var(X) empírico = {varianza_emp:.4f}")

    print("\n" + "=" * 70)
    print("TEOREMA DE BAYES: PRUEBA MEDICA")
    print("=" * 70)
    for k, v in bayes.items():
        print(f"{k:35s} = {v:.6f}")
    print(f"\nVerificación por simulación (1,000,000 personas): "
          f"P(enfermo | positivo) ≈ {bayes_simulado:.4f}")

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\nArchivos generados: convergencia.png, pmf_suma_dados.png, resultados.json")


if __name__ == "__main__":
    main()