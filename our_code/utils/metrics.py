import networkx as nx
import pandas as pd
from scipy import stats
import random
import torch
import numpy as np
import torch.nn as nn
from our_code.models.proto_model import ProtoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def get_confusion_matrix(output_sizes, ys, predicted_ys, true_tree):
    """
    Calculate the average cost between the tree of the predictions and the ground truth tree.

    Params:
        output_sizes: list of integers
        ys: true labels
        predicted_ys: predicted labels
        true_tree: ground truth tree

    Return:
        average_cost: float
    """
    tree, class_labels = true_tree
    tree = nx.DiGraph.to_undirected(tree)
    for idx in range(len(output_sizes)):
        true_data = ys[idx]
        if len(true_data.shape) > 1:
            true_data = true_data.flatten()
        data = {"actual": true_data, "predicted": predicted_ys[idx]}
        df = pd.DataFrame(data, columns=["actual", "predicted"])
        confusion_matrix = pd.crosstab(
            df["actual"], df["predicted"], rownames=["Actual"], colnames=["Predicted"]
        )

        # Set diagonal of conf_np to 0 so only measure cost of the actual errors.
        for col in confusion_matrix.columns:
            confusion_matrix.at[col, col] = 0
        conf_np = confusion_matrix.to_numpy()

        cost_matrix = np.zeros_like(conf_np)
        for row_idx, row_label in enumerate(class_labels):
            for col_idx, col_label in enumerate(class_labels):
                # print("Finding path from", row_label, "to", col_label)
                cost_matrix[row_idx, col_idx] = nx.algorithms.shortest_path_length(
                    tree, row_label, col_label
                )
        total_cost = np.multiply(cost_matrix, conf_np)
        average_cost = np.sum(total_cost) / np.sum(conf_np)
        # Set the diagonals to 0.
        for col in confusion_matrix.columns:
            confusion_matrix.at[col, col] = 0

        return average_cost

def get_deep_mst(model, add_origin=True, plot=False, labels=None):
        nx_graph = nx.Graph()
        tuple_encs = []
        encodings = model.get_prototypes()
        all_labels = []
        for group_id, label_group in enumerate(labels):
            duplication_factor = model.duplication_factors[group_id]
            assert duplication_factor == 1
            added_labels = [label_group for _ in range(duplication_factor)]
            flattened_added = [item for sublist in added_labels for item in sublist]
            all_labels.extend(flattened_added)
        if add_origin:
            encodings.append(np.zeros_like(encodings[0]))
            all_labels.append("origin")
        for label_id, encoding in enumerate(encodings):
            label = all_labels[label_id]
            tuple_enc = tuple(encoding)
            nx_graph.add_node(label)
            for other_id, other_enc in enumerate(tuple_encs):
                other_label = all_labels[other_id]
                nx_graph.add_edge(label, other_label,
                                  weight=np.linalg.norm(np.asarray(other_enc) - np.asarray(tuple_enc)))
            tuple_encs.append(tuple_enc)
        undirected = nx_graph.to_undirected()
        tree = nx.minimum_spanning_tree(undirected)
        # Create a directed version of the tree via the ordering of prototypes.
        ordered_tree = nx.DiGraph(tree)
        for start_id in range(len(encodings)):
            start_label = all_labels[start_id]
            for end_id in range(len(encodings)):
                end_label = all_labels[end_id]
                # Small to big vs. big to small
                # In this case, only allow big to small
                if ordered_tree.has_edge(end_label, start_label) and end_id < start_id:
                    ordered_tree.remove_edge(end_label, start_label)
                    # print("Removing edge from", end_label, "to", start_label)
        # Set node attributes in tree, which is needed for tree equality comparison.
        for node in ordered_tree.nodes:
            ordered_tree.add_node(node, name=str(node))
        return ordered_tree

def get_disparity_metric(predictions, ys, protected_idx=1):
    """
    Calculate the disparate impact and demographic disparity metrics from the predictions and the true labels.

    Params:
        predictions: 2D array
        ys: true labels
        protected_idx: integer

    Return:
        disparity_impact: float
        dem_disparity: float
        (max_key, min_key): tuple of integers
    """

    protected_totals = {}
    protected_positive = {}
    protected_vals = ys[protected_idx]
    for i, protected_val in enumerate(protected_vals):
        if protected_val not in protected_totals.keys():
            protected_totals[protected_val] = 0
            protected_positive[protected_val] = 0
        protected_totals[protected_val] += 1
        if predictions[0][i] % 2 == 1:
            protected_positive[protected_val] += 1
        # print('end of iter')
    print("Protected positives", protected_positive)
    print("Protected totals", protected_totals)
    print("All fractions")
    for key in protected_totals.keys():
        print("For key", key, protected_positive[key] / protected_totals[key])
    sorted_keys = sorted(protected_positive.keys())
    max_key = max(
        protected_totals,
        key=lambda x: protected_positive.get(x) / protected_totals.get(x),
    )
    min_key = min(
        protected_totals,
        key=lambda x: protected_positive.get(x) / protected_totals.get(x),
    )
    random.shuffle(sorted_keys)
    sorted_keys = [max_key, min_key]

    key0 = max_key
    key1 = min_key
    fraction_0 = protected_positive.get(key0) / protected_totals.get(key0)
    fraction_1 = protected_positive.get(key1) / protected_totals.get(key1)
    print("Fraction0", fraction_0)
    print("Fraction1", fraction_1)
    disparity_impact = min([fraction_0 / fraction_1, fraction_1 / fraction_0])
    print("Min disparity", disparity_impact)

    # From the Wasserstein paper, compute the "demographic disparity"
    mean_prob = sum(protected_positive.values()) / sum(protected_totals.values())
    dem_disparity = 0
    for key in sorted_keys:
        fraction = protected_positive.get(key) / protected_totals.get(key)
        dem_disparity += abs(fraction - mean_prob)

    print("Dem disparity", dem_disparity)
    return disparity_impact, dem_disparity, (max_key, min_key)


def get_mst(model, add_origin=True, plot=False, labels=None):
    """
    Create minimum spanning tree.

    Params:
        model: trained model
        add_origin: boolean
        plot: boolean
        labels: list of integers

    Return:
        ordered_tree: Directed Graph abject
    """
    nx_graph = nx.Graph()
    tuple_encs = []
    if len(model.output_sizes) != 2:
        print("WARNING: assume 2 outputs when building MST.")
    encodings = model.get_prototypes()[: model.output_sizes[0] + model.output_sizes[1]]
    all_labels = []
    for group_id, label_group in enumerate(labels):
        duplication_factor = model.duplication_factors[group_id]
        added_labels = [label_group for _ in range(duplication_factor)]
        flattened_added = [item for sublist in added_labels for item in sublist]
        all_labels.extend(flattened_added)
    if add_origin:
        encodings.append(np.zeros_like(encodings[0]))
        all_labels.append("origin")
    for label_id, encoding in enumerate(encodings):
        label = all_labels[label_id]
        tuple_enc = tuple(encoding)
        nx_graph.add_node(label)
        for other_id, other_enc in enumerate(tuple_encs):
            assert other_id < len(encodings)
            other_label = all_labels[other_id]
            nx_graph.add_edge(
                label,
                other_label,
                weight=np.linalg.norm(np.asarray(other_enc) - np.asarray(tuple_enc)),
            )
        tuple_encs.append(tuple_enc)
    undirected = nx_graph.to_undirected()
    tree = nx.minimum_spanning_tree(undirected)
    # Create a directed version of the tree via the ordering of prototypes.
    ordered_tree = nx.DiGraph(tree)
    for start_id in range(len(encodings)):
        start_label = all_labels[start_id]
        for end_id in range(len(encodings)):
            end_label = all_labels[end_id]
            # Small to big vs. big to small
            # In this case, only allow big to small
            if ordered_tree.has_edge(end_label, start_label) and end_id < start_id:
                assert end_id < len(encodings)
                ordered_tree.remove_edge(end_label, start_label)
                # print("Removing edge from", end_label, "to", start_label)

    # if plot:
    #     plot_mst(ordered_tree)
    # Set node attributes in tree, which is needed for tree equality comparison.
    for node in ordered_tree.nodes:
        ordered_tree.add_node(node, name=str(node))
    return ordered_tree


def predict_protected(diffs_to_protos, feature_enc, ys, protected_idx=1, keys=None):
    """
    Calculate the accuracy when trying to predict the protected field.

    Params:
        diffs_to_protos: 2D array
        feature_enc: 1D array
        ys: true labels
        protected_idx: integer
        keys: list

    Return:
        score: float, protected field accuracy
        difference from random chance: float
    """

    in_plane_point = feature_enc - torch.mean(diffs_to_protos, axis=2)
    from sklearn.linear_model import LogisticRegression

    filtered_ys = ys
    filtered_plane_points = in_plane_point
    if keys is not None:
        filtered_ys = [[], [], []]
        filtered_plane_points = []
        for idx in range(len(ys[0])):
            p_val = ys[protected_idx][idx]
            if p_val in keys:
                filtered_ys[0].append(ys[0][idx])
                filtered_ys[1].append(ys[1][idx])
                filtered_plane_points.append(in_plane_point[idx])
                if p_val == keys[0]:
                    filtered_ys[2].append(0)
                elif p_val == keys[1]:
                    filtered_ys[2].append(1)
        filtered_ys = [np.stack(elt) for elt in filtered_ys]
        # filtered_plane_points = tf.stack(filtered_plane_points)
        filtered_plane_points = torch.stack(filtered_plane_points, dim=0)
    # relevant_subspace = keras.backend.eval(filtered_plane_points)
    relevant_subspace = filtered_plane_points.detach().cpu().numpy()
    regression_model = LogisticRegression()
    train_frac = 0.5
    num_points = relevant_subspace.shape[0]
    x_train = relevant_subspace[: int(train_frac * num_points)]
    y_train = filtered_ys[protected_idx][: int(train_frac * num_points)]
    regression_model.fit(x_train, y_train)
    # Use score method to get accuracy of model
    x_test = relevant_subspace[int(train_frac * num_points) :]
    y_test = filtered_ys[protected_idx][int(train_frac * num_points) :]
    score = regression_model.score(x_test, y_test)
    print("S Acc", score)
    random_chance = stats.mode(y_test)[1] / (num_points - int(train_frac * num_points))
    print("Random baseline", random_chance)
    return score, score - random_chance


def get_grad_orthogonality(dataloader, model):
    """
    Calculate rho metric.

    Params:
        dataloader: dataloader
        model: model

    Return:
        prob_updates: list of lists
        other_updates: list of lists
        slopes: list of float
    """
    encodings = []
    ys = [[] for i in range(len(model.output_sizes))]
    for x, _, y, parity_y in dataloader:
        x = x.to(device)
        _, _, tmp_feature_enc = model.encoder(x)
        encodings.extend(tmp_feature_enc)
        ys[0].extend(y)
        ys[1].extend(parity_y)

    prob_updates = []
    other_updates = []
    slopes = []
    optimizer = torch.optim.Adam(model.parameters())

    for predictor_idx, predictor in enumerate(
        model.proto_layers
    ):  #  predictor is proto_layer, classifier_layer
        if predictor_idx >= 1:
            print("Only do this analysis for first case.")
            continue

        for target_classification in range(model.output_sizes[predictor_idx]):
            for magnitude in [1.0]:
                updated_probs = []
                other_class_probs = []
                num_examples_generated = 0
                for i in range(len(encodings)):  # for every test example
                    if num_examples_generated >= 10:
                        break
                    encoding = encodings[i]  # current z in torch
                    correct_class = ys[predictor_idx][i]
                    if correct_class != target_classification:
                        continue
                    num_examples_generated += 1

                    encoding = torch.reshape(encoding, (1, -1))
                    encoding = encoding.to(device)

                    tmp_dist_to_protos, _, _ = predictor(encoding)
                    confidence = model.classifier_layers[predictor_idx](
                        [tmp_dist_to_protos, model.label_mask_layers[predictor_idx]]
                    )
                    correct_class = correct_class.to(device)
                    confidence = confidence.to(device)
                    loss = nn.CrossEntropyLoss()(
                        confidence, torch.unsqueeze(correct_class, dim=0)
                    )

                    optimizer.zero_grad()
                    encoding.retain_grad()  # retain grad for non-leaf tensor

                    loss.backward(retain_graph=True)  # calculate the gradients
                    grads = encoding.grad.data

                    new_enc = encoding - magnitude * grads
                    new_enc = torch.reshape(new_enc, (1, -1))
                    dist_to_protos, _, _ = predictor(new_enc)
                    new_confidence = model.classifier_layers[predictor_idx](
                        [dist_to_protos, model.label_mask_layers[predictor_idx]]
                    )

                    updated_probs.append(
                        new_confidence.detach().cpu().numpy()
                        - confidence.detach().cpu().numpy()
                    )

                    if len(model.proto_layers) > 1:
                        other_idx = (
                            0 if predictor_idx != 0 else len(model.output_sizes) - 1
                        )

                        other_predictor = model.proto_layers[other_idx]

                        dist_to_protos, _, _ = other_predictor(encoding)
                        old_other_confidence = model.classifier_layers[predictor_idx](
                            [dist_to_protos, model.label_mask_layers[other_idx]]
                        )

                        dist_to_protos, _, _ = other_predictor(new_enc)
                        new_other_confidence = model.classifier_layers[predictor_idx](
                            [dist_to_protos, model.label_mask_layers[other_idx]]
                        )

                        other_class_probs.append(
                            new_other_confidence.detach().cpu().numpy()
                            - old_other_confidence.detach().cpu().numpy()
                        )
                # print("Along gradient for predictor", predictor_idx, "towards classification", target_classification, "with magnitude", magnitude)
                mean_updated_probs = np.mean(updated_probs, axis=0)
                main_update_diff = max(abs(mean_updated_probs[0]))
                prob_updates.append(mean_updated_probs[0].tolist())
                if len(model.proto_layers) > 1:
                    mean_other_update = np.mean(other_class_probs, axis=0)
                    other_updated_diff = max(abs(mean_other_update[0]))
                    slope = other_updated_diff / main_update_diff
                    print("Ratio r = ", slope)
                    # print("Mean confidence change for other predictor", mean_other_update)
                    other_updates.append(mean_other_update[0].tolist())
                    slopes.append(slope)
    return prob_updates, other_updates, slopes


def eval_proto_diffs(model, concept_idx=0):
    """
    Calculate prototypes alignment.

    Params:
        model: model
        concept_idx: integer

    Return:
        best_align: integer
    """
    true_protos = model.proto_layers[concept_idx].prototypes
    true_protos = true_protos.detach().cpu().numpy()
    true_diffs = ProtoModel.get_vector_differences(true_protos)
    mean_diffs = []
    mean_sames = []
    for i, other_protos in enumerate(model.proto_layers):
        if i == concept_idx:
            continue
        # For each of the true diffs, find how well it aligns with going from one other to another
        alignments = np.zeros(
            (
                len(true_diffs),
                len(true_diffs),
                other_protos.num_prototypes,
                other_protos.num_prototypes,
            )
        )
        other_protos = other_protos.prototypes.detach().cpu().numpy()
        other_diffs = ProtoModel.get_vector_differences(other_protos)

        for true_idx1, true_proto1_diffs in enumerate(true_diffs):
            for true_idx2, true_p1_to_p2 in enumerate(true_proto1_diffs):
                if true_idx1 == true_idx2:
                    continue
                for j, other_p1_diffs in enumerate(other_diffs):
                    for k, other_p1_to_p2 in enumerate(other_p1_diffs):
                        if j == k:
                            continue
                        cos_alignment = np.square(np.dot(true_p1_to_p2, other_p1_to_p2))
                        alignments[true_idx1, true_idx2, j, k] = cos_alignment

        # Now condense down to say how well aligned is
        best_align = np.max(alignments, axis=(0, 1))

        return best_align


def eval_proto_diffs_fashion(model, concept_idx=0):
    """
    Calculate prototypes alignment for MNIST Fashion dataset.

    Params:
        model: model
        concept_idx: integer

    Return:
        mean_diffs: list of float
        mean_sames: list of float
    """
    best_align = eval_proto_diffs(model, concept_idx=concept_idx)
    diff_groups = []
    same_groups = []
    class_to_group_mapping = {
        0: 1,
        1: 2,
        2: 1,
        3: 2,
        4: 1,
        5: 0,
        6: 1,
        7: 0,
        8: 2,
        9: 0,
    }
    for j, best_row in enumerate(best_align):
        for k, entry in enumerate(best_row):
            if j == k:
                continue
            if class_to_group_mapping.get(j) == class_to_group_mapping.get(k):
                same_groups.append(entry)
                continue
            diff_groups.append(entry)
    print("Diff groups", diff_groups)
    print("Same groups", same_groups)
    print("Mean diff", np.mean(diff_groups))
    print("Mean same", np.mean(same_groups))
    print("Median diff", np.median(diff_groups))
    print("Median same", np.median(same_groups))

    mean_diffs = []
    mean_sames = []
    mean_diffs.append(np.mean(diff_groups))
    mean_sames.append(np.mean(same_groups))
    return mean_diffs, mean_sames


def trees_match(tree1, tree2):
    """
    Check if two trees are the same.

    Params:
        tree1: Graph
        tree2: Graph

    Return:
        iso: boolean
    """

    def labels_match(node1, node2):
        return node1["name"] == node2["name"]

    iso = nx.is_isomorphic(tree1, tree2, node_match=labels_match)
    return iso


def graph_edit_dist(gold_tree, pred_tree):
    """
    Calculate edit distance between two trees.

    Params:
        gold_tree: Graph
        pred_tree: Graph

    Return:
        integer distance
    """

    def labels_match(node1, node2):
        return node1["name"] == node2["name"]

    return nx.graph_edit_distance(
        gold_tree, pred_tree, node_match=labels_match, timeout=120
    )
