# モデルアーキテクチャ仕様書
# CMI BFRB Detection - Model Architecture Specification

## 1. 全体アーキテクチャ概要

```
┌──────────────────────────────────────────────────────────┐
│                  Hierarchical Ensemble System             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Level 1: Feature Extraction & Base Models               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │Deep Learning│  │   LightGBM  │  │   XGBoost   │     │
│  │   Models    │  │   Models    │  │   Models    │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                 │                 │            │
│  Level 2: Task-Specific Heads                            │
│  ┌──────▼─────────────────▼─────────────────▼──────┐    │
│  │  Binary Head  │  BFRB Head  │  Full Head        │    │
│  │  (2 classes)  │  (8 classes) │  (18 classes)    │    │
│  └──────┬─────────────────┬─────────────────┬──────┘    │
│         │                 │                 │            │
│  Level 3: Meta-Learning & Optimization                   │
│  ┌──────▼─────────────────▼─────────────────▼──────┐    │
│  │           Stacking Meta-Learner                  │    │
│  └───────────────────────────────────────────────────┘    │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## 2. Deep Learning Models

### 2.1 Enhanced Two-Branch Architecture

```python
class EnhancedTwoBranchModel(tf.keras.Model):
    """
    改善された2ブランチアーキテクチャ
    - IMU専用ブランチ（深層）
    - TOF/Thermal専用ブランチ（軽量）
    - マルチヘッド出力
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # IMU Branch Components
        self.imu_cnn = self._build_imu_cnn()
        self.imu_rnn = self._build_imu_rnn()
        
        # TOF/Thermal Branch Components  
        self.tof_cnn = self._build_tof_cnn()
        self.tof_attention = self._build_tof_attention()
        
        # Fusion Layer
        self.fusion = self._build_fusion_layer()
        
        # Multi-Head Outputs
        self.binary_head = self._build_binary_head()
        self.bfrb_head = self._build_bfrb_head()
        self.full_head = self._build_full_head()
```

#### 2.1.1 IMU Branch Architecture

```python
def _build_imu_cnn(self):
    """
    Multi-Scale Residual CNN with SE-blocks
    """
    return tf.keras.Sequential([
        # Multi-scale feature extraction
        MultiScaleConv1D(filters=[32, 64, 128], 
                         kernel_sizes=[3, 5, 7]),
        
        # Residual blocks with SE attention
        ResidualSEBlock(filters=128, kernel_size=3),
        ResidualSEBlock(filters=256, kernel_size=3),
        ResidualSEBlock(filters=512, kernel_size=3),
        
        # Pooling
        tf.keras.layers.GlobalAveragePooling1D()
    ])

def _build_imu_rnn(self):
    """
    Bidirectional LSTM + GRU with Skip Connections
    """
    return tf.keras.Sequential([
        # First BiLSTM layer
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True,
                                dropout=0.3, recurrent_dropout=0.3)
        ),
        
        # Skip connection wrapper
        SkipConnection(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(128, return_sequences=True,
                                   dropout=0.3, recurrent_dropout=0.3)
            )
        ),
        
        # Attention layer
        MultiHeadAttention(num_heads=8, key_dim=64)
    ])
```

#### 2.1.2 TOF/Thermal Branch Architecture

```python
def _build_tof_cnn(self):
    """
    Lightweight CNN for TOF/Thermal data
    """
    return tf.keras.Sequential([
        # Simple convolutions
        tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        
        # Dropout for regularization
        tf.keras.layers.Dropout(0.3)
    ])

def _build_tof_attention(self):
    """
    Self-attention for TOF spatial patterns
    """
    return SpatialAttention(reduction_ratio=8)
```

### 2.2 Temporal Convolutional Network (TCN)

```python
class TCNModel(tf.keras.Model):
    """
    TCN with dilated convolutions for long-range dependencies
    """
    
    def __init__(self, config):
        super().__init__()
        
        # TCN blocks with exponentially increasing dilation
        self.tcn_blocks = [
            TCNBlock(filters=64, kernel_size=3, dilation_rate=1),
            TCNBlock(filters=128, kernel_size=3, dilation_rate=2),
            TCNBlock(filters=256, kernel_size=3, dilation_rate=4),
            TCNBlock(filters=512, kernel_size=3, dilation_rate=8)
        ]
        
        # Global pooling
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        
        # Classification heads
        self.classifier = self._build_classifier()
```

### 2.3 Transformer-based Model

```python
class TransformerModel(tf.keras.Model):
    """
    Transformer encoder for sequence modeling
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=256)
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=256,
                num_heads=8,
                dff=1024,
                dropout_rate=0.3
            ) for _ in range(4)
        ]
        
        # Pooling and classification
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = self._build_classifier()
```

## 3. Gradient Boosting Models

### 3.1 LightGBM Configuration

```python
class LightGBMPipeline:
    """
    Optimized LightGBM pipeline for tabular features
    """
    
    def __init__(self):
        self.params = {
            'objective': 'multiclass',
            'num_class': 18,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_child_samples': 20,
            'max_depth': -1,
            'class_weight': 'balanced'
        }
        
        # Separate models for different tasks
        self.binary_model = None  # BFRB vs Non-BFRB
        self.bfrb_model = None    # 8-class BFRB
        self.full_model = None    # 18-class full
```

### 3.2 XGBoost Configuration

```python
class XGBoostPipeline:
    """
    XGBoost with GPU acceleration
    """
    
    def __init__(self):
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 18,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 1,
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
            'random_state': 42
        }
```

### 3.3 CatBoost Configuration

```python
class CatBoostPipeline:
    """
    CatBoost for handling categorical features
    """
    
    def __init__(self):
        self.params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'class_weights': self._calculate_class_weights(),
            'random_seed': 42,
            'verbose': False,
            'task_type': 'GPU' if GPU_AVAILABLE else 'CPU'
        }
```

## 4. Hierarchical Classification Strategy

### 4.1 Binary Classifier (Level 1)

```python
class BinaryClassifier:
    """
    BFRB vs Non-BFRB binary classification
    """
    
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
```

### 4.2 BFRB Subtype Classifier (Level 2)

```python
class BFRBClassifier:
    """
    8-class BFRB subtype classification
    """
    
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=FocalLoss(gamma=2.0, alpha=self._get_class_weights()),
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
```

### 4.3 Full Classifier (Level 3)

```python
class FullClassifier:
    """
    18-class full classification with hierarchical constraints
    """
    
    def __init__(self):
        self.model = self._build_hierarchical_model()
        
    def _build_hierarchical_model(self):
        """
        Hierarchical softmax with constraint layers
        """
        inputs = tf.keras.layers.Input(shape=(feature_dim,))
        
        # Shared layers
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Binary branch
        binary_out = tf.keras.layers.Dense(1, activation='sigmoid', name='binary')(x)
        
        # BFRB branch (conditional on binary)
        bfrb_features = tf.keras.layers.Dense(256, activation='relu')(x)
        bfrb_out = tf.keras.layers.Dense(8, activation='softmax', name='bfrb')(bfrb_features)
        
        # Non-BFRB branch
        non_bfrb_features = tf.keras.layers.Dense(256, activation='relu')(x)
        non_bfrb_out = tf.keras.layers.Dense(10, activation='softmax', name='non_bfrb')(non_bfrb_features)
        
        # Combine outputs with hierarchical constraints
        final_out = HierarchicalCombine()([binary_out, bfrb_out, non_bfrb_out])
        
        return tf.keras.Model(inputs=inputs, outputs=final_out)
```

## 5. Meta-Learning & Stacking

### 5.1 Stacking Architecture

```python
class StackingMetaLearner:
    """
    Meta-learner for combining base model predictions
    """
    
    def __init__(self, base_models, meta_model='neural_network'):
        self.base_models = base_models
        self.meta_model_type = meta_model
        self.meta_model = self._build_meta_model()
        
    def _build_meta_model(self):
        if self.meta_model_type == 'neural_network':
            return tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(18, activation='softmax')
            ])
        elif self.meta_model_type == 'linear':
            return LogisticRegression(multi_class='multinomial')
        elif self.meta_model_type == 'xgboost':
            return xgb.XGBClassifier(n_estimators=100, max_depth=3)
```

### 5.2 Weighted Averaging

```python
class WeightedAverageEnsemble:
    """
    Optimized weighted averaging based on validation scores
    """
    
    def __init__(self, models, optimization_metric='combined_score'):
        self.models = models
        self.weights = None
        self.optimization_metric = optimization_metric
        
    def optimize_weights(self, val_predictions, val_labels):
        """
        Optimize weights using differential evolution
        """
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.zeros_like(val_predictions[0])
            
            for pred, w in zip(val_predictions, weights):
                ensemble_pred += w * pred
            
            score = self._calculate_metric(ensemble_pred, val_labels)
            return -score  # Minimize negative score
        
        # Optimization
        result = differential_evolution(
            objective,
            bounds=[(0, 1)] * len(self.models),
            seed=42,
            maxiter=100
        )
        
        self.weights = result.x / np.sum(result.x)
        return self.weights
```

## 6. Training Strategies

### 6.1 Progressive Training

```python
class ProgressiveTrainer:
    """
    段階的な学習戦略
    """
    
    def train(self, data, labels):
        # Stage 1: Binary classification
        print("Stage 1: Training binary classifier...")
        binary_labels = (labels < 8).astype(int)
        self.binary_model.fit(data, binary_labels, epochs=50)
        
        # Stage 2: BFRB subtype classification
        print("Stage 2: Training BFRB classifier...")
        bfrb_mask = labels < 8
        bfrb_data = data[bfrb_mask]
        bfrb_labels = labels[bfrb_mask]
        self.bfrb_model.fit(bfrb_data, bfrb_labels, epochs=100)
        
        # Stage 3: Full model fine-tuning
        print("Stage 3: Fine-tuning full model...")
        self.full_model.fit(data, labels, epochs=150)
```

### 6.2 Curriculum Learning

```python
class CurriculumLearning:
    """
    Easy to hard sample ordering
    """
    
    def __init__(self, difficulty_metric='confidence'):
        self.difficulty_metric = difficulty_metric
        
    def order_samples(self, data, labels, model):
        """
        Order samples by difficulty
        """
        predictions = model.predict(data)
        
        if self.difficulty_metric == 'confidence':
            # Use prediction confidence as difficulty
            confidences = np.max(predictions, axis=1)
            difficulty_order = np.argsort(confidences)[::-1]  # Easy to hard
            
        elif self.difficulty_metric == 'loss':
            # Use individual sample loss
            losses = categorical_crossentropy(labels, predictions)
            difficulty_order = np.argsort(losses)  # Low to high loss
            
        return difficulty_order
```

## 7. Loss Functions

### 7.1 Custom Competition Loss

```python
class CompetitionLoss(tf.keras.losses.Loss):
    """
    カスタム損失関数：Binary F1とMacro F1の最適化
    """
    
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Binary weight
        self.beta = beta   # Macro weight
        
    def call(self, y_true, y_pred):
        # Binary loss (BFRB vs Non-BFRB)
        binary_true = tf.cast(y_true < 8, tf.float32)
        binary_pred = tf.reduce_sum(y_pred[:, :8], axis=1)
        binary_loss = tf.keras.losses.binary_crossentropy(binary_true, binary_pred)
        
        # Macro loss (per-class)
        macro_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Combined loss
        total_loss = self.alpha * binary_loss + self.beta * macro_loss
        
        return total_loss
```

### 7.2 Focal Loss

```python
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for handling class imbalance
    """
    
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        
        if self.alpha is not None:
            alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss
            
        return tf.reduce_mean(focal_loss)
```

## 8. Model Optimization

### 8.1 Learning Rate Scheduling

```python
class CustomLRScheduler(tf.keras.callbacks.Callback):
    """
    カスタム学習率スケジューラ
    """
    
    def __init__(self, initial_lr=1e-3, decay_steps=10, decay_rate=0.5):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 30:
            # Warmup
            lr = self.initial_lr * (epoch + 1) / 30
        else:
            # Cosine annealing
            lr = self.initial_lr * (1 + np.cos(np.pi * (epoch - 30) / 100)) / 2
            
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
```

### 8.2 Model Pruning

```python
class ModelPruning:
    """
    モデルの枝刈りと量子化
    """
    
    def prune_model(self, model, target_sparsity=0.5):
        """
        Magnitude-based pruning
        """
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params
        )
        
        return pruned_model
    
    def quantize_model(self, model):
        """
        INT8 quantization for inference
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        quantized_model = converter.convert()
        return quantized_model
```

## 9. 推論パイプライン

### 9.1 Ensemble Inference

```python
class EnsembleInference:
    """
    アンサンブル推論パイプライン
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
    def predict(self, sequence, demographics):
        """
        単一シーケンスの予測
        """
        # Feature extraction
        features = self.extract_features(sequence, demographics)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            if isinstance(model, DeepLearningModel):
                pred = model.predict_sequence(features['time_series'])
            else:  # Gradient Boosting
                pred = model.predict_proba(features['tabular'])
            predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        # Apply hierarchical constraints
        final_pred = self.apply_hierarchical_constraints(ensemble_pred)
        
        return final_pred
```

### 9.2 Test Time Augmentation (TTA)

```python
class TestTimeAugmentation:
    """
    推論時のデータ拡張
    """
    
    def __init__(self, n_augmentations=5):
        self.n_augmentations = n_augmentations
        
    def predict_with_tta(self, model, sequence):
        """
        TTA with multiple augmentations
        """
        predictions = []
        
        # Original prediction
        predictions.append(model.predict(sequence))
        
        # Augmented predictions
        for _ in range(self.n_augmentations - 1):
            aug_sequence = self.augment_sequence(sequence)
            predictions.append(model.predict(aug_sequence))
        
        # Average predictions
        final_prediction = np.mean(predictions, axis=0)
        
        return final_prediction
    
    def augment_sequence(self, sequence):
        """
        Apply random augmentation
        """
        augmentations = [
            lambda x: x + np.random.normal(0, 0.01, x.shape),  # Noise
            lambda x: x * np.random.uniform(0.95, 1.05),       # Scale
            lambda x: np.roll(x, np.random.randint(-5, 5), axis=0)  # Shift
        ]
        
        aug_func = np.random.choice(augmentations)
        return aug_func(sequence)
```

---

**実装上の注意点**:
- GPU/TPUの利用を前提とした実装
- メモリ効率を考慮したバッチ処理
- 推論速度の最適化（量子化、蒸留）
- エラーハンドリングとフォールバック機構の実装