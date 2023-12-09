"""
Created on Thu Dec  7 00:09:27 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

import pandas as pd
import numpy as np
import random

class DecisionTree:
    
    def __init__(self, max_depth=float('inf'), min_samples=3):
            self.max_depth = max_depth
            self.min_samples = min_samples
            self.tree = None
            self.feature_importances = {}
            
    def fit(self, data):
        data['label'] = pd.Categorical(data['label'])
        self.tree = self._build_tree(data, depth=self.max_depth)

    def _split_data(self, data, feature, threshold):
        left = data[data[feature] <= threshold]
        right = data[data[feature] > threshold]
        return left, right

    def _calculate_gini(self, labels):
        if len(labels) == 0:
            return 0
        proportions = labels.value_counts() / len(labels)
        return 1 - sum(proportions ** 2)

    def _find_best_split(self, data):
        features = data.columns[:-1]
        best_gini = float('inf')
        best_split = None
    
        for feature in features:
            sorted_values = data[feature].sort_values().unique()
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2  # Puntos medios entre valores únicos adyacentes
    
            for threshold in thresholds:
                splits = self._split_data(data, feature, threshold)
                gini_left = self._calculate_gini(splits[0]['label'])
                gini_right = self._calculate_gini(splits[1]['label'])
                gini = (len(splits[0]) * gini_left + len(splits[1]) * gini_right) / len(data)
    
                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature': feature, 'threshold': threshold, 'gini': gini, 'left': splits[0],
                                  'right': splits[1]}
        if best_split is not None:
            feature_name = best_split['feature']
            self.feature_importances[feature_name] = self.feature_importances.get(feature_name, 0) + best_gini

        return best_split


    def _build_tree(self, data, depth):
        if depth == 0 or len(data['label'].unique()) == 1 or len(data) <= self.min_samples:
            return {'node_type': 'leaf', 'class_distribution': data['label'].value_counts().to_dict()}
    
        best_split = self._find_best_split(data)
    
        left = self._build_tree(best_split['left'], depth - 1)
        right = self._build_tree(best_split['right'], depth - 1)
    
        return {'node_type': 'decision', 'feature': best_split['feature'], 'threshold': best_split['threshold'],
                'left': left, 'right': right}


    def print_tree(self, level=0, direction='NA'):
        tree_data = pd.DataFrame(columns=['Level', 'Type', 'Feature', 'Threshold', 'Direction', 'Class_Distribution'])

        def traverse_tree(tree, level, direction):
            nonlocal tree_data
            if tree['node_type'] == 'leaf':
                leaf_data = pd.DataFrame(
                    [[level, 'Leaf', 'NA', 'NA', direction, str(tree['class_distribution'])]],
                    columns=['Level', 'Type', 'Feature', 'Threshold', 'Direction', 'Class_Distribution'])
                tree_data = pd.concat([tree_data, leaf_data], ignore_index=True)
            else:
                node_type = 'Root' if level == 0 else 'Decision'
                decision_data = pd.DataFrame(
                    [[level, node_type, tree['feature'], tree['threshold'], direction, 'NA']],
                    columns=['Level', 'Type', 'Feature', 'Threshold', 'Direction', 'Class_Distribution'])
                tree_data = pd.concat([tree_data, decision_data], ignore_index=True)
                traverse_tree(tree['left'], level + 1, 'Left')
                traverse_tree(tree['right'], level + 1, 'Right')

        traverse_tree(self.tree, level, direction)
        tree_data = tree_data.sort_values(by='Level').reset_index(drop=True)
        print(tree_data.to_string(index=False))

    def predict(self, data):
        predictions = []

        def predict_instance(tree, instance):
            if tree['node_type'] == 'leaf':
                return max(tree['class_distribution'], key=tree['class_distribution'].get)
            else:
                if instance[tree['feature']] <= tree['threshold']:
                    return predict_instance(tree['left'], instance)
                else:
                    return predict_instance(tree['right'], instance)

        for i in range(len(data)):
            instance = data.iloc[i]
            prediction = predict_instance(self.tree, instance)
            predictions.append(prediction)

        return predictions


class RandomForest:
    def __init__(self, n_trees=10, max_depth=float('inf'), min_samples=3, random_state=None):
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.used_indices = []
        self.random_seed = random_state
        self.feature_importances = {}

    def fit(self, data, max_features=None):
        np.random.seed(self.random_seed)
    
        for _ in range(self.n_trees):
            # Selección aleatoria de características
            if max_features is not None:
                subset = data.sample(frac=1, replace=True, axis=0).sample(frac=max_features, replace=True, axis=1)
            else:
                subset = data.sample(frac=1, replace=True, axis=0).sample(frac=1, replace=True, axis=1)
    
            indices = subset.index
            self.used_indices.extend(indices)
            sampled_data = data.loc[indices]  # Bootstrap 
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(sampled_data)
            self.trees.append(tree)
            for feature, importance in tree.feature_importances.items():
                self.feature_importances[feature] = self.feature_importances.get(feature, 0) + importance

    
        self.used_indices = list(set(self.used_indices))


    def predict(self, data):
        
        predictions = []

        for i in range(len(data)):
            instance = data.iloc[i]
            tree_predictions = [tree.predict(pd.DataFrame([instance]))[0] for tree in self.trees]
            majority_vote = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(majority_vote)

        return predictions

    def print_trees(self, n=1, random_state=None):
       
        if random_state is not None:
            random.seed(random_state)  

        trees_to_print = n if n is not None else len(self.trees)

        # Obtener índices aleatorios de los árboles a imprimir
        trees_indices_to_print = random.sample(range(len(self.trees)), trees_to_print)

        for i in trees_indices_to_print:
            print(f"\nTree {i + 1}:")
            self.trees[i].print_tree()
            print("\n" + "_" * 90 + "\n")

    def get_used_indices(self):
        
        return self.used_indices