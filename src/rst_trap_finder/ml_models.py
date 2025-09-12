"""Machine learning integration for trap prediction and optimization."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import classification_report, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from . import TRAP_LETTERS
from .graph import Graph
from .scores import (
    one_step_rst_prob, escape_hardness, k_step_rst_prob, 
    minimax_topm, biased_pagerank, composite
)


class FeatureExtractor:
    """Extract features from graph nodes for ML models."""
    
    def __init__(self, graph: Graph, trap_letters: set = None):
        self.graph = graph
        self.trap_letters = trap_letters or TRAP_LETTERS
        self._pagerank = None
    
    @property
    def pagerank(self):
        """Cached PageRank computation."""
        if self._pagerank is None:
            self._pagerank = biased_pagerank(self.graph, self.trap_letters)
        return self._pagerank
    
    def extract_node_features(self, node: str) -> Dict[str, float]:
        """Extract comprehensive features for a single node."""
        if node not in self.graph:
            return self._get_zero_features()
        
        neighbors = self.graph[node]
        total_weight = sum(neighbors.values())
        
        # Basic graph features
        features = {
            'out_degree': len(neighbors),
            'total_out_weight': total_weight,
            'avg_out_weight': total_weight / len(neighbors) if neighbors else 0.0,
            'max_out_weight': max(neighbors.values()) if neighbors else 0.0,
            'min_out_weight': min(neighbors.values()) if neighbors else 0.0,
        }
        
        # Trap-specific features
        trap_neighbors = [v for v in neighbors if v and v[0] in self.trap_letters]
        trap_weight = sum(neighbors[v] for v in trap_neighbors)
        
        features.update({
            'trap_out_degree': len(trap_neighbors),
            'trap_weight_ratio': trap_weight / total_weight if total_weight > 0 else 0.0,
            'strong_trap_ratio': self._strong_trap_ratio(node),
        })
        
        # Text features
        features.update(self._extract_text_features(node))
        
        # Scoring features
        features.update({
            'one_step_prob': one_step_rst_prob(node, self.graph, self.trap_letters),
            'escape_hardness': escape_hardness(node, self.graph, self.trap_letters),
            'k2_step_prob': k_step_rst_prob(node, self.graph, self.trap_letters, k=2),
            'minimax_score': minimax_topm(node, self.graph, self.trap_letters),
            'pagerank': self.pagerank.get(node, 0.0),
        })
        
        # Neighborhood features
        features.update(self._extract_neighborhood_features(node))
        
        return features
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero-filled feature dict for missing nodes."""
        return {
            'out_degree': 0.0, 'total_out_weight': 0.0, 'avg_out_weight': 0.0,
            'max_out_weight': 0.0, 'min_out_weight': 0.0, 'trap_out_degree': 0.0,
            'trap_weight_ratio': 0.0, 'strong_trap_ratio': 0.0, 'word_length': 0.0,
            'vowel_ratio': 0.0, 'consonant_clusters': 0.0, 'starts_with_trap': 0.0,
            'ends_with_vowel': 0.0, 'one_step_prob': 0.0, 'escape_hardness': 0.0,
            'k2_step_prob': 0.0, 'minimax_score': 0.0, 'pagerank': 0.0,
            'neighbor_avg_degree': 0.0, 'neighbor_trap_ratio': 0.0, 'clustering_local': 0.0,
        }
    
    def _strong_trap_ratio(self, node: str) -> float:
        """Ratio of strong edges leading to trap letters."""
        neighbors = self.graph.get(node, {})
        if not neighbors:
            return 0.0
        
        max_weight = max(neighbors.values())
        threshold = 0.05 * max_weight
        strong_edges = [v for v, w in neighbors.items() if w >= threshold]
        
        if not strong_edges:
            return 0.0
        
        trap_strong = sum(1 for v in strong_edges if v and v[0] in self.trap_letters)
        return trap_strong / len(strong_edges)
    
    def _extract_text_features(self, node: str) -> Dict[str, float]:
        """Extract linguistic features from word."""
        vowels = set('aeiou')
        
        return {
            'word_length': len(node),
            'vowel_ratio': sum(1 for c in node if c in vowels) / len(node) if node else 0.0,
            'consonant_clusters': self._count_consonant_clusters(node),
            'starts_with_trap': 1.0 if node and node[0] in self.trap_letters else 0.0,
            'ends_with_vowel': 1.0 if node and node[-1] in vowels else 0.0,
        }
    
    def _count_consonant_clusters(self, word: str) -> float:
        """Count consonant clusters in word."""
        vowels = set('aeiou')
        clusters = 0
        in_cluster = False
        
        for char in word:
            if char not in vowels:
                if not in_cluster:
                    clusters += 1
                    in_cluster = True
            else:
                in_cluster = False
        
        return clusters / len(word) if word else 0.0
    
    def _extract_neighborhood_features(self, node: str) -> Dict[str, float]:
        """Extract features based on node's neighborhood."""
        neighbors = self.graph.get(node, {})
        
        if not neighbors:
            return {
                'neighbor_avg_degree': 0.0,
                'neighbor_trap_ratio': 0.0,
                'clustering_local': 0.0,
            }
        
        # Neighbor degree statistics
        neighbor_degrees = [len(self.graph.get(v, {})) for v in neighbors]
        avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0.0
        
        # Neighbor trap ratio
        trap_neighbors = sum(1 for v in neighbors if v and v[0] in self.trap_letters)
        neighbor_trap_ratio = trap_neighbors / len(neighbors)
        
        # Local clustering approximation
        clustering_local = self._approximate_local_clustering(node)
        
        return {
            'neighbor_avg_degree': avg_neighbor_degree,
            'neighbor_trap_ratio': neighbor_trap_ratio,
            'clustering_local': clustering_local,
        }
    
    def _approximate_local_clustering(self, node: str) -> float:
        """Approximate local clustering coefficient."""
        neighbors = list(self.graph.get(node, {}).keys())
        if len(neighbors) < 2:
            return 0.0
        
        # Count edges between neighbors
        edges_between = 0
        for i, n1 in enumerate(neighbors):
            for j, n2 in enumerate(neighbors[i+1:], i+1):
                if n2 in self.graph.get(n1, {}):
                    edges_between += 1
        
        possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
        return edges_between / possible_edges if possible_edges > 0 else 0.0
    
    def extract_features_batch(self, nodes: List[str]) -> pd.DataFrame:
        """Extract features for multiple nodes efficiently."""
        features_list = []
        for node in nodes:
            features = self.extract_node_features(node)
            features['node'] = node
            features_list.append(features)
        
        return pd.DataFrame(features_list)


class TrapPredictor:
    """ML model for predicting trap effectiveness."""
    
    def __init__(self, model_type: str = 'random_forest'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML features")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_training_data(self, graph: Graph, 
                            scoring_function: str = 'composite') -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from graph."""
        extractor = FeatureExtractor(graph)
        nodes = list(graph.keys())
        
        # Extract features
        features_df = extractor.extract_features_batch(nodes)
        
        # Generate labels based on scoring function
        if scoring_function == 'composite':
            pr = biased_pagerank(graph)
            labels = [composite(node, graph, TRAP_LETTERS, pr) for node in nodes]
        elif scoring_function == 'one_step':
            labels = [one_step_rst_prob(node, graph, TRAP_LETTERS) for node in nodes]
        else:
            raise ValueError(f"Unknown scoring function: {scoring_function}")
        
        # Convert to binary classification if needed
        if self.model_type == 'random_forest':
            threshold = np.percentile(labels, 75)  # Top 25% as positive class
            labels = (np.array(labels) >= threshold).astype(int)
        
        return features_df, np.array(labels)
    
    def train(self, graph: Graph, scoring_function: str = 'composite', 
              test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model and return performance metrics."""
        features_df, labels = self.prepare_training_data(graph, scoring_function)
        
        # Remove non-numeric columns
        feature_cols = [col for col in features_df.columns if col != 'node']
        X = features_df[feature_cols]
        self.feature_columns = feature_cols
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels if self.model_type == 'random_forest' else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        predictions = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self._get_feature_importance())),
        }
        
        if self.model_type == 'random_forest':
            from sklearn.metrics import classification_report
            metrics['classification_report'] = classification_report(y_test, predictions, output_dict=True)
        else:
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics
    
    def predict(self, graph: Graph, nodes: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict trap scores for nodes."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if nodes is None:
            nodes = list(graph.keys())
        
        extractor = FeatureExtractor(graph)
        features_df = extractor.extract_features_batch(nodes)
        
        X = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'random_forest':
            # Return probability of positive class
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            return dict(zip(nodes, probabilities))
        else:
            predictions = self.model.predict(X_scaled)
            return dict(zip(nodes, predictions))
    
    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            return np.zeros(len(self.feature_columns))
    
    def save_model(self, path: Union[str, Path]):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: Union[str, Path]):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']


class ParameterOptimizer:
    """Automated parameter optimization using Optuna."""
    
    def __init__(self, graph: Graph):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required for parameter optimization")
        
        self.graph = graph
        self.study = None
    
    def optimize_composite_weights(self, n_trials: int = 100) -> Dict[str, float]:
        """Optimize composite scoring weights."""
        def objective(trial):
            # Suggest weight parameters
            l1 = trial.suggest_float('l1', 0.0, 1.0)
            l2 = trial.suggest_float('l2', 0.0, 1.0)
            l3 = trial.suggest_float('l3', 0.0, 1.0)
            l4 = trial.suggest_float('l4', 0.0, 1.0)
            l5 = trial.suggest_float('l5', 0.0, 1.0)
            
            # Normalize weights
            total = l1 + l2 + l3 + l4 + l5
            if total == 0:
                return 0.0
            
            weights = (l1/total, l2/total, l3/total, l4/total, l5/total)
            
            # Compute scores
            pr = biased_pagerank(self.graph)
            scores = []
            trap_scores = []
            non_trap_scores = []
            
            for node in self.graph:
                score = composite(node, self.graph, TRAP_LETTERS, pr, weights)
                scores.append(score)
                
                if node and node[0] in TRAP_LETTERS:
                    trap_scores.append(score)
                else:
                    non_trap_scores.append(score)
            
            # Objective: maximize separation between trap and non-trap words
            if not trap_scores or not non_trap_scores:
                return 0.0
            
            trap_mean = np.mean(trap_scores)
            non_trap_mean = np.mean(non_trap_scores)
            
            return trap_mean - non_trap_mean
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        best_params = self.study.best_params
        total = sum(best_params.values())
        
        return {k: v/total for k, v in best_params.items()}
    
    def optimize_pagerank_alpha(self, n_trials: int = 50) -> float:
        """Optimize PageRank bias parameter."""
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1.0, 5.0)
            
            pr = biased_pagerank(self.graph, TRAP_LETTERS, alpha=alpha)
            
            # Compute trap vs non-trap PageRank separation
            trap_pr = [pr.get(node, 0) for node in pr if node and node[0] in TRAP_LETTERS]
            non_trap_pr = [pr.get(node, 0) for node in pr if not (node and node[0] in TRAP_LETTERS)]
            
            if not trap_pr or not non_trap_pr:
                return 0.0
            
            return np.mean(trap_pr) - np.mean(non_trap_pr)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params['alpha']


class WordEmbeddingAnalyzer:
    """Analyze word relationships using embeddings."""
    
    def __init__(self, embedding_dim: int = 100):
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required for word embeddings")
        
        self.embedding_dim = embedding_dim
        self.model = None
        self.is_trained = False
    
    def train_embeddings(self, graph: Graph, min_count: int = 1, window: int = 5) -> Word2Vec:
        """Train Word2Vec embeddings from graph structure."""
        # Generate walks through the graph
        walks = self._generate_random_walks(graph, num_walks=100, walk_length=20)
        
        # Train Word2Vec
        self.model = Word2Vec(
            walks, 
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        self.is_trained = True
        return self.model
    
    def _generate_random_walks(self, graph: Graph, num_walks: int, walk_length: int) -> List[List[str]]:
        """Generate random walks through the graph."""
        walks = []
        nodes = list(graph.keys())
        
        for _ in range(num_walks):
            for start_node in nodes:
                walk = self._random_walk(graph, start_node, walk_length)
                if len(walk) > 1:
                    walks.append(walk)
        
        return walks
    
    def _random_walk(self, graph: Graph, start: str, length: int) -> List[str]:
        """Perform a single random walk."""
        walk = [start]
        current = start
        
        for _ in range(length - 1):
            neighbors = graph.get(current, {})
            if not neighbors:
                break
            
            # Weighted random choice
            weights = list(neighbors.values())
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            next_node = np.random.choice(list(neighbors.keys()), p=probabilities)
            walk.append(next_node)
            current = next_node
        
        return walk
    
    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get words most similar to given word."""
        if not self.is_trained:
            raise ValueError("Embeddings must be trained first")
        
        try:
            return self.model.wv.most_similar(word, topn=top_k)
        except KeyError:
            return []
    
    def compute_trap_embedding_features(self, graph: Graph) -> Dict[str, Dict[str, float]]:
        """Compute embedding-based features for trap analysis."""
        if not self.is_trained:
            raise ValueError("Embeddings must be trained first")
        
        features = {}
        
        for node in graph:
            if node not in self.model.wv:
                features[node] = {
                    'trap_similarity': 0.0,
                    'embedding_centrality': 0.0,
                    'semantic_isolation': 0.0
                }
                continue
            
            # Similarity to trap letters
            trap_similarities = []
            for trap_word in graph:
                if trap_word and trap_word[0] in TRAP_LETTERS and trap_word in self.model.wv:
                    sim = self.model.wv.similarity(node, trap_word)
                    trap_similarities.append(sim)
            
            avg_trap_similarity = np.mean(trap_similarities) if trap_similarities else 0.0
            
            # Embedding centrality (average similarity to all other words)
            all_similarities = []
            for other_node in graph:
                if other_node != node and other_node in self.model.wv:
                    sim = self.model.wv.similarity(node, other_node)
                    all_similarities.append(sim)
            
            embedding_centrality = np.mean(all_similarities) if all_similarities else 0.0
            
            # Semantic isolation (inverse of centrality)
            semantic_isolation = 1.0 - embedding_centrality
            
            features[node] = {
                'trap_similarity': avg_trap_similarity,
                'embedding_centrality': embedding_centrality,
                'semantic_isolation': semantic_isolation
            }
        
        return features