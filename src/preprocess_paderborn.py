import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder

WINDOW_SIZE = 1024

CLASSES = ["BallFault", "InnerRace", "Normal", "OuterRace"]

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)


def extract_signal_from_mat(mat_path):
    import numpy as np
    from scipy.io import loadmat

    mat = loadmat(mat_path)
    mat = {k: v for k, v in mat.items() if not k.startswith("__")}

    data = list(mat.values())[0]
    X = data['X'][0, 0]

    data_cells = X['Data'][0]

    # --------------------------------------------------
    # Automatically select a valid vibration channel
    # --------------------------------------------------
    for idx, channel in enumerate(data_cells):
        try:
            signal = np.array(channel).squeeze()
            if signal.ndim == 1 and signal.size > 5000:
                # Found a valid vibration signal
                return signal.astype(np.float32)
        except Exception:
            continue

    raise ValueError(f"No valid vibration signal found in {mat_path}")




def create_windows(signal, window_size):
    n_windows = len(signal) // window_size
    windows = []

    for i in range(n_windows):
        window = signal[i * window_size:(i + 1) * window_size]

        # --------------------------------------------------
        # PER-WINDOW STANDARDIZATION (CRITICAL FIX)
        # --------------------------------------------------
        mean = np.mean(window)
        std = np.std(window)

        if std < 1e-8:
            std = 1e-8

        window = (window - mean) / std

        windows.append(window)

    return np.array(windows, dtype=np.float32)


def infer_label_from_folder(folder_name):
    """
    Paderborn → CWRU label mapping
    """
    if folder_name.startswith("K00"):
        return "Normal"
    elif folder_name.startswith("KI"):
        return "InnerRace"
    elif folder_name.startswith("KA"):
        return "OuterRace"
    elif folder_name.startswith("KB"):
        return "BallFault"
    else:
        raise ValueError(f"Unknown Paderborn class folder: {folder_name}")


def preprocess_paderborn(root_dir):
    X_all, y_all = [], []

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)

        if not os.path.isdir(class_path):
            continue

        label = infer_label_from_folder(class_folder)
        print(f"Processing folder: {class_folder} → {label}")

        for file in os.listdir(class_path):
            if not file.endswith(".mat"):
                continue

            file_path = os.path.join(class_path, file)

            signal = extract_signal_from_mat(file_path)

            # --------------------------------------------------
            # GLOBAL MAX-ABS NORMALIZATION (same as CWRU)
            # --------------------------------------------------
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val

            windows = create_windows(signal, WINDOW_SIZE)

            X_all.append(windows)
            y_all.extend([label] * len(windows))

    X = np.vstack(X_all)
    X = X[..., np.newaxis]   # (N, 1024, 1)

    y = label_encoder.transform(y_all)

    print("\nPaderborn preprocessing complete")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label distribution:", np.unique(y, return_counts=True))

    return X, y
