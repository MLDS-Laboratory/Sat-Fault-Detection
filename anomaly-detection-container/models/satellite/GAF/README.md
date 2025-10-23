# Satellite Anomaly Detection with Gramian Angular Fields (GAF)

This module implements satellite telemetry anomaly detection using Gramian Angular Field (GAF) image encoding of time series data with Convolutional Neural Networks (CNNs). It's particularly well-suited for detecting anomalies in spacecraft telemetry data that may indicate failures or unusual behaviors.

## Overview

Traditional anomaly detection in time series can be challenging. The GAF approach offers a clever workaround by transforming one-dimensional time series data into two-dimensional image-like representations, which unlocks the power of image classification techniques for anomaly detection:

1. **Time Series to Images**: Satellite telemetry data is converted to Gramian Angular Field images, which encode temporal correlations in a visual format
2. **CNN Classification**: These images are fed to a CNN which learns to distinguish visual patterns associated with normal vs. anomalous behavior
3. **Performance Evaluation**: Models are evaluated using multiple metrics (accuracy, F1 score, and confusion matrix) to ensure reliable anomaly detection

Why this approach works well: CNNs excel at finding spatial patterns in images, and the GAF transformation preserves the temporal relationships in the data in a format CNNs can understand.

## Getting Started

### Dependencies

- PyTorch (deep learning framework)
- torchvision (computer vision utilities)
- NumPy (numerical computing)
- Pillow (image processing)
- scikit-learn (evaluation metrics and sampling)
- matplotlib (visualization)
- tqdm (progress bars)

### Basic Usage

```python
# Train and evaluate a model
python gaf_main.py

# Visualize GAF images
python visualize_gaf.py --mode single --segment_idx 42
```

## Data Loading and Preprocessing

### Supported Data Loaders

This system works with multiple satellite data sources. You'll need to modify `gaf_main.py` to use the appropriate data loader:

1. **ESA Mission Data Loader** (default - better for large-scale analysis)
   ```python
   # In gaf_main.py (lines 32-33), keep these uncommented:
   mission_dir = os.path.abspath(os.path.join(__file__, "../../../../data/ESA-Anomaly/ESA-Mission1"))
   loader = ESAMissionDataLoader(mission_dir=mission_dir)
   
   # And comment out any OPS-SAT loader code (lines 27-29)
   ```

2. **OPS-SAT Data Loader** (good for smaller datasets with more balanced classes)
   ```python
   # In gaf_main.py, uncomment lines 27-29:
   dataset_csv = os.path.abspath(os.path.join(__file__, "../../../../data/OPS-SAT/dataset.csv"))
   segment_csv = os.path.abspath(os.path.join(__file__, "../../../../data/OPS-SAT/segments.csv"))
   loader = OpsSatDataLoader(dataset_csv, segment_csv)
   
   # And comment out ESA loader code (lines 32-33)
   ```

Choose your dataloader based on:
- Dataset size: ESA datasets tend to be much larger than OPS-SAT
- Class balance: ESA datasets typically have fewer anomalies (more imbalanced)
- Memory constraints: Larger datasets may require sampling strategies

### Expected Data Format

The data loaders expect data to be structured as a list of segment dictionaries. Each segment should contain:

- `ts`: numpy array of time series values (the actual telemetry measurements)
- `label`: binary class (0=normal, 1=anomaly)
- `channel`: telemetry channel identifier (which sensor/measurement the data comes from)
- `segment`: segment identifier (unique ID for this data chunk)
- `sampling`: sampling rate information (how frequently measurements were taken)

This format allows the system to handle multi-channel telemetry data while tracking segment origins and timing information.

### Sampling Options

Real-world satellite data presents two major challenges:
1. **Size**: Datasets can be enormous (millions of segments)
2. **Imbalance**: Anomalies are rare (often <1% of the data)

To address these challenges, the code includes sampling strategies:

1. **Stratified Sampling** (enabled by default for ESA on line 41)
   ```python
   # This limits dataset size while preserving class distribution
   # Modify max_train_samples and max_test_samples based on your compute resources
   train_segs, test_segs = stratified_sample(
       train_segs, test_segs, 
       max_train_samples=100000,  # Adjust based on your RAM/GPU memory
       max_test_samples=20000
   )
   ```
   
   Why use this? It dramatically reduces training time while maintaining representative data distribution. It's especially important for large datasets that won't fit in memory.

2. **Anomaly Oversampling** (useful for highly imbalanced datasets like ESA)
   ```python
   # This ensures a minimum percentage of anomalies through smart oversampling
   # Find this in gaf_data_loader.py around line 22
   train_segs, test_segs = stratified_sample(
       train_segs, test_segs, 
       max_train_samples=100000, 
       max_test_samples=20000,
       oversample_anomaly=True,  # Set to True to enable oversampling
       min_anomaly_pct=0.15      # Target minimum anomaly percentage
   )
   ```
   
   This is done because the models may struggle with highly imbalanced data. If your dataset has <1% anomalies (common in ESA data), the model may just predict "normal" for everything and still achieve >99% accuracy. Oversampling ensures the model sees enough anomalies during training to learn meaningful patterns.

   To disable oversampling, set `oversample_anomaly=False` when calling `stratified_sample()`.

## GAF Transformation: How It Works

The Gramian Angular Field transforms time series data into images that capture temporal patterns in a visual format:

1. **Normalization**: First, the time series is scaled to range [-1,1], which makes the data suitable for the angular transformation (think of this as mapping values to positions on a unit circle)

2. **Angular Mapping**: Values are converted to angles using arccos, which maps the normalized time series from linear values to angles on a circle

3. **Pairwise Calculation**: The GAF matrix is built by calculating the cosine of the sum of angles for each pair of time points, essentially encoding how different time points relate to each other

The resulting "image" has some fascinating properties:
- The main diagonal captures the original series
- Off-diagonal elements represent the correlation between points at different times
- Temporal patterns like trends and seasonality become visible as textures and patterns
- Anomalies often appear as distinct visual artifacts

This transformation enables us to treat time series anomaly detection as an image classification problem, leveraging the power of CNNs that excel at identifying visual patterns.

## Model Options

### Available Models

The system offers two model architectures, each with its own strengths:

1. **Pre-trained ResNet-18** (default - recommended for most cases)
   ```python
   # In gaf_main.py (line 138), uncomment:
   model = get_pretrained_resnet(num_classes=2, freeze_early=True)
   model_name = "pretrained"
   
   # And comment out the scratch CNN (line 136)
   ```

   Why use this? Transfer learning from pre-trained models gives you a head start - the model has already learned to extract useful features from images. This typically leads to faster training and better performance, especially with limited training data.

2. **Custom CNN from Scratch** (useful for experimentation or when your data is very different from natural images)
   ```python
   # In gaf_main.py, uncomment line 136:
   model = CNNFromScratch(num_classes=2, input_size=224)
   model_name = "scratch"
   
   # And comment out the pretrained model (line 138)
   ```

   Why use this? A custom CNN may perform better for very specialized data patterns that differ significantly from those in natural images. It's also helpful for learning purposes or when you need a lighter model.

### Hyperparameters

Fine-tune model performance by adjusting the hyperparameters in `gaf_main.py` (around line 133):

```python
hp = dict(
    epochs=10,                # Number of training epochs (increase for better performance, but watch for overfitting)
    batch_size=32,            # Batch size (larger = faster training but more memory)
    lr=5e-3,                  # Learning rate (controls how quickly the model adapts to the data)
    loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.01),  # Loss function with label smoothing to reduce overfitting
    loss_name="CrossEntropy"  # Loss function name for logging
)
```

Tips for adjustment:
- For difficult datasets, try increasing epochs (15-20) and reducing learning rate (1e-4 to 1e-3)
- If training is slow, increase batch size (64 or 128) if your GPU has enough memory
- If you see overfitting (validation loss increases while training loss decreases), try more label smoothing (0.05-0.1)

For systematic hyperparameter tuning, use `hyperparam_search.py`, which will test multiple combinations and find the best settings.

## Caching

The GAF transformation is computationally expensive since it involves multiple matrix operations for each time series. To speed up repeated runs, the system includes a caching mechanism:

```python
# In gaf_main.py, modify GAFDataset initialization around line 82:
full_train = GAFDataset(
    train_segs, 
    transform=tfms['train'],
    cache_dir="/path/to/cache"   # Add this line to enable caching
)
test_ds = GAFDataset(
    test_segs,  
    transform=tfms['val'],
    cache_dir="/path/to/cache"   # Same cache directory for test data
)
```

**When to use caching:**
- When running multiple experiments on the same dataset (saves hours of preprocessing time)
- For large datasets where GAF computation is a bottleneck
- When you have sufficient disk space (GAF images can take up significant space)

**When to avoid caching:**
- For one-time experiments (initial cache creation still takes time)
- When disk space is limited (each GAF image is ~150-300KB, so 100K segments = ~15-30GB)
- For small datasets where transformation is already fast
- When testing modifications to the GAF transformation itself (cached results won't reflect changes)

How it works: The first time each segment is processed, its GAF representation is saved to disk. On subsequent runs, the system checks if a cached version exists and loads it instead of recomputing.

## Visualization

GAF images can be challenging to interpret at first. The visualization tools help you understand what patterns the model is looking for:

```bash
# Single GAF image with time series - great for detailed inspection
python visualize_gaf.py --mode single --segment_idx 5

# Multiple images in a grid - useful for comparing patterns across segments
python visualize_gaf.py --mode grid --grid_size 8 8 --save gaf_grid.png

# Detailed grid with both time series and GAF images
python visualize_gaf.py --mode detailed_grid --grid_size 4 4

# Focus on anomalies to study their visual signatures
python visualize_gaf.py --mode grid --show_anomalies_only
```

These visualizations are invaluable for:
- Understanding what normal vs. anomalous patterns look like
- Debugging model performance (why certain anomalies are missed)
- Explaining results to stakeholders
- Gaining intuition about how the GAF transformation works

## Prediction

Once trained, your model can predict anomalies in new telemetry data:

```bash
python gaf_predict.py --model_path /path/to/model.pth --model_type pretrained --segment_path /path/to/segment.npy
```

This outputs both the binary prediction (normal/anomaly) and the probability scores, giving you a measure of the model's confidence.

For integration into larger systems, you can also import the prediction function:

```python
from gaf_predict import predict_gaf_image

# Your time series data
telemetry_data = [1.2, 1.3, 1.1, 0.9, 1.4, 5.6, 1.2, 1.1]  

# Make prediction
prediction, probabilities = predict_gaf_image(model, telemetry_data)
```

## Outputs and Results

The system generates several types of outputs:

1. **Model Checkpoints**: The best model during training is automatically saved to `CNNs/saved_models/`
   - Format: `{ModelName}.pth` (e.g., `ResNet.pth`)
   - Contains model weights that can be loaded for inference
   
2. **Training History**: During training, you'll see detailed metrics:
   - Per-epoch loss, accuracy, and F1 score for both training and validation sets
   - Evaluation metrics on the test set after training
   - Runtime statistics to help optimize performance

3. **Visualizations**: When using visualization tools with the `--save` flag:
   - Individual GAF images with corresponding time series
   - Grids of multiple GAF images for comparison
   - Detailed analysis views that help with interpretation

These outputs help you assess model performance, understand the data, and diagnose any issues.

## Advanced Usage

### Mixed Precision Training

Modern GPUs support mixed precision, which can speed up training by 2-3x with minimal accuracy impact:

```python
# In gaf_main.py, modify the ModelTrainer initialization (around line 104):
trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device, mixed_precision=True)
```

This is highly recommended for NVIDIA GPUs with Tensor Cores (RTX or Tesla series).

### Saving and Loading Models

For long-term projects, you'll want to save and reuse trained models:

```python
# Save model (typically done automatically during training)
trainer.save_model("/path/to/model.pth", save_entire_model=True)

# Load model for prediction (in gaf_predict.py or your own code)
from CNNs.pretrained_resnet import get_pretrained_resnet
model = get_pretrained_resnet(num_classes=2)
model.load_state_dict(torch.load("/path/to/model.pth"))
```

Setting `save_entire_model=True` saves the entire model architecture along with weights, which is more portable but results in larger files.

## Customization

The system is designed to be adapted to different datasets and requirements:

1. **Image Size**: Change the resolution of GAF images based on your needs
   ```python
   # In gaf_data_loader.py, modify GAFDataset initialization:
   GAFDataset(segments, transform=tfms['train'], image_size=224)
   ```
   
   Smaller sizes (128, 160) are faster but may lose detail
   Larger sizes (256, 320) capture more detail but require more memory and computation

2. **Normalization**: The default normalization matches ImageNet statistics, but you can customize:
   ```python
   # In gaf_main.py, modify transforms (around line 70):
   tfms = {
       'train': transforms.Compose([
           transforms.ToTensor(),
           # These mean/std values are from ImageNet - you may want to calculate
           # dataset-specific values for potentially better performance
           transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
       ])
   }
   ```

3. **Data Augmentation**: For limited datasets, add data augmentation to improve generalization:
   ```python
   # Add these transforms before normalization:
   transforms.RandomRotation(10),
   transforms.RandomHorizontalFlip(),
   ```
   
   Data augmentation works surprisingly well with GAF images despite their different nature from natural photos.