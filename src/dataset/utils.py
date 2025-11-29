import numpy as np
import logging


def graph_processing(data, graph, processing):
    C, T, V, M = data.shape
    num_person = 1 if len(graph.split('-')) == 1 else int(graph.split('-')[1])

    if num_person > 1:
        if processing == 'default':
            multi_person_data = np.zeros([C, T, V * M, 1], dtype=data.dtype)
            for i in range(M):
                multi_person_data[:, :, V * i:V * i + V, 0] = data[:, :, :, i]
        elif processing == 'two-group':
            multi_person_data = np.zeros([C, T, V * M // 2, 2], dtype=data.dtype)
            for i in range(M // 2):
                multi_person_data[:, :, V * i:V * i + V, 0] = data[:, :, :, i]
                multi_person_data[:, :, V * i:V * i + V, 1] = data[:, :, :, i + M // 2]
        else:
            logging.info('')
            logging.error('Error: Wrong in loading processing configs')
            raise ValueError()
        return multi_person_data
    return data


def multi_input(data, conn, inputs, centers):
    C, T, V, M = data.shape
    joint = np.zeros((C * 2, T, V, M), dtype=data.dtype)
    joint_motion = np.zeros((C * 2, T, V, M), dtype=data.dtype)
    bone = np.zeros((C * 2, T, V, M), dtype=data.dtype)
    bone_motion = np.zeros((C * 2, T, V, M), dtype=data.dtype)
    joint[:C, :, :, :] = data
    for i in range(V):
        center = centers[i]
        if center >= 0:
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, center, :]

    for i in range(len(conn)):
        if conn[i] >= 0:
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i, :, :, :] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C + i, :, :, :] = np.arccos(np.clip(bone[i, :, :, :] / bone_length, -1.0, 1.0))

    # Joint motion: J(t+1)-J(t), pad first frame with zeros
    joint_motion[:C, 1:, :, :] = joint[:C, 1:, :, :] - joint[:C, :-1, :, :]
    # replicate to the second half
    joint_motion[C:, :, :, :] = joint_motion[:C, :, :, :]

    # Bone motion: B(t+1)-B(t), pad first frame with zeros
    bone_motion[:C, 1:, :, :] = bone[:C, 1:, :, :] - bone[:C, :-1, :, :]
    bone_motion[C:, :, :, :] = bone_motion[:C, :, :, :]

    streams = {
        "J": joint,
        "JM": joint_motion,
        "B": bone,
        "BM": bone_motion,
    }
    alias_map = {
        "JOINT": "J",
        "JOINT-MOTION": "JM",
        "BONE": "B",
        "BONE-MOTION": "BM",
    }

    if isinstance(inputs, str):
        normalized_inputs = inputs.upper()
        normalized_inputs = alias_map.get(normalized_inputs, normalized_inputs)
        if normalized_inputs == "JVBM":
            order = ["J", "JM", "B", "BM"]
        elif normalized_inputs in streams:
            order = [normalized_inputs]
        else:
            logging.error("Unsupported inputs value: %s", inputs)
            raise ValueError(f"Unsupported inputs value: {inputs}")
    elif isinstance(inputs, (list, tuple)):
        order = [str(item).upper() for item in inputs]
        order = [alias_map.get(item, item) for item in order]
    else:
        order = ["J", "JM", "B", "BM"]

    data_new = []
    for key in order:
        if key not in streams:
            logging.error("Unknown stream key '%s' in inputs %s", key, inputs)
            raise ValueError(f"Unknown stream key '{key}' in inputs {inputs}")
        data_new.append(streams[key])

    return np.stack(data_new, axis=0)
