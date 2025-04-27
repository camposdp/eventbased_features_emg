import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

# ===================== Funções =====================

def load_exercise_data(base_dir, exercise="E1"):
    all_signals, all_labels = [], []
    subject_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("s")]

    for subj_dir in subject_dirs:
        mat_files = glob(os.path.join(subj_dir, "**", f"*_{exercise}_*.mat"), recursive=True)
        if not mat_files:
            print(f"[AVISO] Nenhum arquivo encontrado para o exercício {exercise} em {subj_dir}")
            continue
        for mat_file in mat_files:
            print(f"Carregando {mat_file}...")
            mat = loadmat(mat_file, squeeze_me=True)
            signals = mat['emg']
            if signals.shape[0] == 12 and signals.ndim == 2:
                signals = signals.T
            restimulus = mat['restimulus'].astype(int)
            all_signals.append(signals)
            all_labels.append(restimulus)

    if not all_signals:
        raise ValueError(f"Nenhum dado carregado para o exercício {exercise}.")

    return np.vstack(all_signals), np.concatenate(all_labels)

def extract_feature(signal_window, feature_type):
    if feature_type == 'MAV':
        return np.mean(np.abs(signal_window), axis=0)
    elif feature_type == 'RMS':
        return np.sqrt(np.mean(signal_window**2, axis=0))
    elif feature_type == 'WL':
        return np.sum(np.abs(np.diff(signal_window, axis=0)), axis=0)
    elif feature_type == 'ZC':
        return np.sum(np.diff(np.sign(signal_window), axis=0) != 0, axis=0)
    elif feature_type == 'SSC':
        d1 = np.diff(signal_window, axis=0)
        d2 = np.sign(d1)
        d3 = np.diff(d2, axis=0)
        return np.sum(d3 != 0, axis=0)
    elif feature_type == 'IAV':
        return np.sum(np.abs(signal_window), axis=0)
    elif feature_type == 'LD':
        return np.exp(np.mean(np.log(np.abs(signal_window) + 1e-6), axis=0))
    elif feature_type == 'DASDV':
        return np.sqrt(np.mean(np.diff(signal_window, axis=0)**2, axis=0))
    elif feature_type == 'STD':
        return np.std(signal_window, axis=0)
    else:
        raise ValueError("Feature não reconhecida")

def segment_movements(signals, labels):
    """
    Detecta início e fim de cada movimento (labels != 0) para segmentação por evento.
    """
    segments = []
    n_samples = len(labels)
    current_label = 0
    start_idx = None

    for i in range(n_samples):
        if labels[i] != 0 and current_label == 0:
            start_idx = i
            current_label = labels[i]
        elif (labels[i] == 0 or labels[i] != current_label) and current_label != 0:
            end_idx = i
            segments.append((start_idx, end_idx, current_label))
            current_label = 0

    # Caso o movimento vá até o final
    if current_label != 0 and start_idx is not None:
        segments.append((start_idx, n_samples-1, current_label))

    return segments

def process_data_event_based(signals, labels, feature_list):
    """
    Processa o sinal extraindo features agrupadas por evento (não sliding window).
    """
    X_all_features = {feat: [] for feat in feature_list}
    y_labels = []

    segments = segment_movements(signals, labels)

    for (start_idx, end_idx, label) in segments:
        segment = signals[start_idx:end_idx, :]
        if segment.shape[0] < 10:  # Ignorar movimentos muito curtos
            continue

        for feat_type in feature_list:
            feats = extract_feature(segment, feat_type)
            X_all_features[feat_type].append(feats)

        y_labels.append(label)

    # Transformar listas em arrays
    for feat in X_all_features:
        X_all_features[feat] = np.array(X_all_features[feat])

    return X_all_features, np.array(y_labels)

# ===================== Início do Main =====================

def main():
    base_dir = "."
    feature_types = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'IAV', 'LD', 'DASDV', 'STD']
    exercises = ["E1", "E2", "E3"]
    n_splits = 10
    for exercise in exercises:
        print(f"\n======================== EVENT-BASED - EXERCÍCIO {exercise} ========================")

        signals, labels = load_exercise_data(base_dir, exercise)
        X_all_features, y = process_data_event_based(signals, labels, feature_types)

        if len(np.unique(y)) < 2:
            print("[AVISO] Dados insuficientes para classificação")
            continue

        all_features_names = list(X_all_features.keys())
        available_features = all_features_names.copy()
        selected_features = []
        history = []

        # Padronizar cada feature
        scaler_dict = {}
        for feat in all_features_names:
            scaler = StandardScaler()
            X_all_features[feat] = scaler.fit_transform(X_all_features[feat])
            scaler_dict[feat] = scaler

        # Split correto: apenas índices
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

        model_base = SVC(kernel='linear', C=1.0, probability=True)

        print("[INFO] Iniciando Forward Feature Selection agrupado...")
        accuracies_per_step = []
        for step_idx in tqdm(range(len(available_features)), desc=f"Forward Selection {exercise}"):
            best_acc = 0
            best_feat = None

            for feat in available_features:
                feats_to_use = selected_features + [feat]
                X_train_feats = np.hstack([X_all_features[f][train_idx] for f in feats_to_use])
                X_test_feats  = np.hstack([X_all_features[f][test_idx] for f in feats_to_use])

                model_base.fit(X_train_feats, y[train_idx])
                y_pred = model_base.predict(X_test_feats)
                acc = accuracy_score(y[test_idx], y_pred)

                print(f"[DEBUG] Testando adição de {feat}: Acurácia = {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_feat = feat

            if best_feat is not None:
                selected_features.append(best_feat)
                available_features.remove(best_feat)

                history.append({
                    'Step': step_idx + 1,
                    'Added_Feature': best_feat,
                    'Accuracy': best_acc
                })
                accuracies_per_step.append(best_acc)

                print(f"[INFO] Feature escolhida: {best_feat} -> Acurácia acumulada: {best_acc:.4f}")
            else:
                break

        # Salvar CSV histórico
        df_history = pd.DataFrame(history)
        df_history.to_csv(f"ffs_history_eventbased_{exercise}.csv", index=False)
        print(f"\n[RESULTADO] Forward Selection para {exercise} finalizado.")

        # Plotar gráfico incremental
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(accuracies_per_step)+1), accuracies_per_step, marker='o')
        for i, feat in enumerate([h['Added_Feature'] for h in history]):
            plt.annotate(feat, (i+1, accuracies_per_step[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        plt.xlabel('Nº de Features Adicionadas')
        plt.ylabel('Acurácia de Teste')
        plt.title(f'Forward Feature Selection (Event-Based) - {exercise}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"grafico_ffs_eventbased_{exercise}.png")
        plt.close()

        # ===================== Validação Cruzada 10-Fold =====================
        print(f"\n[INFO] Iniciando validação cruzada 10-Fold para {exercise}...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold_idx, (train_fold_idx, test_fold_idx) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
            X_train_fold_feats = np.hstack([X_all_features[f][train_fold_idx] for f in selected_features])
            X_test_fold_feats  = np.hstack([X_all_features[f][test_fold_idx] for f in selected_features])
            y_train_fold = y[train_fold_idx]
            y_test_fold = y[test_fold_idx]

            model_fold = SVC(kernel='linear', C=1.0, probability=True)
            model_fold.fit(X_train_fold_feats, y_train_fold)
            y_pred_fold = model_fold.predict(X_test_fold_feats)
            acc = accuracy_score(y_test_fold, y_pred_fold)

            fold_results.append({
                'Fold': fold_idx + 1,
                'Accuracy': acc
            })

            print(f"[INFO] Fold {fold_idx+1}: Acurácia = {acc:.4f}")

        df_folds = pd.DataFrame(fold_results)
        df_folds.to_csv(f"folds_result_eventbased_{exercise}.csv", index=False)
        print(f"\n[RESULTADO] Validação cruzada para {exercise} concluída. Resultados salvos.")

        # Gráfico da validação cruzada
        plt.figure(figsize=(8,5))
        accs = df_folds['Accuracy']
        plt.bar(range(1, len(accs)+1), accs)
        plt.xlabel('Fold')
        plt.ylabel('Acurácia')
        plt.title(f'Validação Cruzada 10-Fold (Event-Based) - {exercise}')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"grafico_folds_eventbased_{exercise}.png")
        plt.close()

        # Salvar modelo final
        X_final_feats = np.hstack([X_all_features[f] for f in selected_features])
        final_model = SVC(kernel='linear', C=1.0, probability=True)
        final_model.fit(X_final_feats, y)

        joblib.dump(final_model, f"modelo_eventbased_{exercise}.pkl")
        print(f"[INFO] Modelo final salvo para {exercise}.")

if __name__ == "__main__":
    main()
