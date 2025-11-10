import numpy as np

def select_top_m_people(pose_data, M_target=4):
    T, M_current, V, C = pose_data.shape

    energy = np.zeros(M_current)
    for m in range(M_current):
        person = pose_data[:, m, :, :2]
        valid_mask = person.sum(axis=-1).sum(axis=-1) != 0
        person_valid = person[valid_mask]
        if len(person_valid) > 1:
            energy[m] = np.var(person_valid, axis=0).sum()
        else:
            energy[m] = 0

    sorted_idx = np.argsort(-energy)

    if M_current >= M_target:
        pose_trimmed = pose_data[:, sorted_idx[:M_target], :, :]
    else:
        pad = np.zeros((T, M_target - M_current, V, C), dtype=pose_data.dtype)
        pose_trimmed = np.concatenate([pose_data, pad], axis=1)

    return pose_trimmed