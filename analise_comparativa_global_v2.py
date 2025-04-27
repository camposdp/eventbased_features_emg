import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import wilcoxon, ttest_rel
from scipy.stats import wilcoxon, ttest_rel
import openpyxl
# ===================== Parâmetros =====================
exercises = ["E1", "E2", "E3"]
methods = ["Sliding", "Event-Based"]

# Onde estão os arquivos CSVs
folder = "."

# Para armazenar
results_folds = []
results_ffs = []

# Nome da pasta de saída
output_folder = "resultados_analise"

# Cria a pasta se não existir
os.makedirs(output_folder, exist_ok=True)

# ===================== Carregar Dados =====================

for exercise in exercises:
    for method in methods:
        if method == "Sliding":
            folds_file = os.path.join(folder, f"folds_result_{exercise}.csv")
            ffs_file = os.path.join(folder, f"ffs_history_{exercise}.csv")
        else:
            folds_file = os.path.join(folder, f"folds_result_eventbased_{exercise}.csv")
            ffs_file = os.path.join(folder, f"ffs_history_eventbased_{exercise}.csv")

        if not os.path.exists(folds_file) or not os.path.exists(ffs_file):
            print(f"[AVISO] Arquivos não encontrados para {method} {exercise}")
            continue

        # Carregar folds
        df_folds = pd.read_csv(folds_file)
        df_folds["Exercise"] = exercise
        df_folds["Method"] = method
        results_folds.append(df_folds)

        # Carregar FFS
        df_ffs = pd.read_csv(ffs_file)
        df_ffs["Exercise"] = exercise
        df_ffs["Method"] = method
        results_ffs.append(df_ffs)

print("[INFO] Todos os arquivos carregados com sucesso.")

# ===================== Organizar DataFrames =====================

# Todos folds
df_folds_all = pd.concat(results_folds, ignore_index=True)

# Histórico do Forward Feature Selection
df_ffs_all = pd.concat(results_ffs, ignore_index=True)

# Salvar versão bruta
df_folds_all.to_csv(os.path.join(output_folder,"comparativo_folds_global.csv"), index=False)
df_ffs_all.to_csv(os.path.join(output_folder,"comparativo_ffs_global.csv"), index=False)

# ===================== Criar Tabela de Acurácia Incremental =====================

# Para cada combinação Método-Exercício
tabela_ffs = []

for (method, exercise), group in df_ffs_all.groupby(["Method", "Exercise"]):
    for idx, row in group.iterrows():
        tabela_ffs.append({
            "Exercise": exercise,
            "Method": method,
            "Step": row['Step'],
            "Feature_Added": row['Added_Feature'],
            "Accuracy": row['Accuracy']
        })

df_tabela_ffs = pd.DataFrame(tabela_ffs)

# Salvar tabela
df_tabela_ffs.to_csv(os.path.join(output_folder,"comparativo_ffs_table.csv"), index=False)

print("[INFO] Tabela de acurácia incremental criada e salva.")


# ===================== Análise Estatística Atualizada =====================


# Guardar resultados dos testes
stat_results = []

print("\n===== Análise Estatística (Wilcoxon entre métodos) =====")

for exercise in exercises:
    acc_sliding = df_folds_all[(df_folds_all['Exercise'] == exercise) & (df_folds_all['Method'] == "Sliding")]['Accuracy'] * 100
    acc_event = df_folds_all[(df_folds_all['Exercise'] == exercise) & (df_folds_all['Method'] == "Event-Based")]['Accuracy'] * 100

    if len(acc_sliding) > 0 and len(acc_event) > 0:
        stat, p = wilcoxon(acc_sliding, acc_event)
        print(f"Exercise {exercise}: p-value (Wilcoxon) = {p:.4f}")
        stat_results.append({
            "Exercise": exercise,
            "Test": "Wilcoxon",
            "p-value": p,
            "Significant (p<0.05)": "Yes" if p < 0.05 else "No"
        })

# ===================== Salvar Resultados Estatísticos =====================

df_stat = pd.DataFrame(stat_results)
df_stat.to_excel(os.path.join(output_folder,"resultado_estatistica.xlsx"), index=False)

print("\n[INFO] Resultados estatísticos salvos em 'resultado_estatistica.xlsx'.")

# ===================== Gráficos Configurações =====================
sns.set_context("talk")
sns.set_style("whitegrid")

colors = {
    "Sliding": "#0072B2",      # Azul
    "Event-Based": "#E69F00"   # Laranja
}

# Função para erro padrão da média
def sem(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

# ===================== Gráfico de barras com erro padrão e estrelas =====================

plt.figure(figsize=(10,6))

order = ["Sliding", "Event-Based"]
ax = sns.barplot(
    data=df_folds_all,
    x="Exercise",
    y=df_folds_all["Accuracy"] * 100,
    hue="Method",
    hue_order=order,
    ci=None,
    palette=colors,
    capsize=0.2
)

# Calcular médias e erro padrão manualmente
for i, (exercise, method) in enumerate([(e, m) for e in exercises for m in order]):
    y_vals = df_folds_all[(df_folds_all['Exercise'] == exercise) & (df_folds_all['Method'] == method)]['Accuracy'] * 100
    mean = np.mean(y_vals)
    error = sem(y_vals)
    ax.errorbar(i, mean, yerr=error, fmt='none', c='black', capsize=5)

# Adicionar asterisco se diferença significativa
for idx, row in df_stat.iterrows():
    if row['Significant (p<0.05)'] == "Yes":
        x1 = exercises.index(row['Exercise']) * 2
        x2 = x1 + 1
        y_max = max(
            df_folds_all[(df_folds_all['Exercise'] == row['Exercise'])]['Accuracy'] * 100
        ) + 5
        plt.plot([x1, x2], [y_max, y_max], color='black')
        plt.text((x1+x2)/2, y_max+1, "*", ha='center', va='bottom', color='black', fontsize=20)

plt.title('Mean Accuracy per Exercise (with Standard Error)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xlabel('Exercise', fontsize=14)
plt.legend(title='Method', fontsize=12, title_fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"grafico_comparativo_barras_atualizado.svg"))
plt.savefig(os.path.join(output_folder,"grafico_comparativo_barras_atualizado.pdf"))
plt.close()

# ===================== Gráfico de Boxplot =====================

plt.figure(figsize=(10,6))
sns.boxplot(
    data=df_folds_all,
    x="Exercise",
    y=df_folds_all["Accuracy"] * 100,
    hue="Method",
    palette=colors
)
plt.title('Accuracy Dispersion per Exercise', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xlabel('Exercise', fontsize=14)
plt.legend(title='Method', fontsize=12, title_fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"grafico_comparativo_boxplot_atualizado.svg"))
plt.savefig(os.path.join(output_folder,"grafico_comparativo_boxplot_atualizado.pdf"))
plt.close()

# ===================== Gráfico de Violinplot =====================

plt.figure(figsize=(10,6))
sns.violinplot(
    data=df_folds_all,
    x="Exercise",
    y=df_folds_all["Accuracy"] * 100,
    hue="Method",
    split=True,
    inner="quartile",
    palette=colors
)
plt.title('Accuracy Distribution per Exercise (Violin Plot)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xlabel('Exercise', fontsize=14)
plt.legend(title='Method', fontsize=12, title_fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"grafico_comparativo_violinplot.svg"))
plt.savefig(os.path.join(output_folder,"grafico_comparativo_violinplot.pdf"))
plt.close()

# ===================== Gráficos de Acurácia Incremental (1 gráfico por exercício) =====================

for exercise in exercises:
    plt.figure(figsize=(12,6))

    for method in methods:
        df_sub = df_tabela_ffs[(df_tabela_ffs['Exercise'] == exercise) & (df_tabela_ffs['Method'] == method)]
        if len(df_sub) == 0:
            continue
        steps = df_sub['Step']
        accs = df_sub['Accuracy'] * 100
        feats = df_sub['Feature_Added']

        plt.plot(steps, accs, marker='o', label=method, color=colors[method])

        for i, txt in enumerate(feats):
            plt.annotate(txt, (steps.iloc[i], accs.iloc[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

    plt.title(f'Accuracy vs Number of Features - {exercise}', fontsize=16)
    plt.xlabel('Number of Features Added', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, title='Method', title_fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,f"grafico_incremental_{exercise}_atualizado.svg"))
    plt.savefig(os.path.join(output_folder,f"grafico_incremental_{exercise}_atualizado.pdf"))
    plt.close()

print("\n[INFO] Todos os gráficos atualizados e salvos em SVG/PDF!")
