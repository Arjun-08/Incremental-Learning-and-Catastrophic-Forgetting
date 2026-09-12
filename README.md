# Incremental Learning and Catastrophic Forgetting

An experimental study of incremental learning using ResNet18 on the Oxford-IIIT Pet dataset.

The objective is to investigate whether a neural network can learn new classes without losing its ability to recognize previously learned classes. The experiment is divided into three stages:

1. Train a model on an initial set of 20 classes.
2. Expand the classifier to 37 classes and fine-tune it using only the 17 new classes.
3. Use exemplar replay and knowledge distillation to reduce the forgetting of the original 20 classes.

The assignment asks whether models can continuously learn new classes while retaining previously acquired knowledge, and specifically evaluates the effect of fine-tuning and distillation on the original classes.

---

## 1. Problem Statement

Traditional neural networks are generally trained assuming that all classes are available during training.

In an incremental learning setting, classes arrive in stages.

For this experiment, the 37 classes are divided into:

* **Old classes:** 20 classes
* **New classes:** 17 classes

The model first learns the 20 old classes. Later, the classifier is expanded to recognize all 37 classes and the model is trained using only the new 17 classes.

The key question is:

> Can the model learn the 17 new classes without forgetting what it learned about the original 20 classes?

This phenomenon is known as **catastrophic forgetting**.

---

## 2. Dataset

The experiment uses the **Oxford-IIIT Pet dataset**, which contains 37 pet classes.

The dataset is divided into two incremental learning tasks:

| Task     | Number of Classes | Purpose               |
| -------- | ----------------: | --------------------- |
| Old task |                20 | Initial knowledge     |
| New task |                17 | Incremental knowledge |

The implementation creates separate datasets and dataloaders for the old classes, new classes, and the complete 37-class test set.

```text
37 classes
     |
     +-------------------+
     |                   |
  20 old              17 new
  classes              classes
     |                   |
 Stage 1              Stage 2
     |                   |
     +---------+---------+
               |
        37-class model
               |
       Distillation +
       exemplar replay
```

The class split is explicitly implemented using the first 20 class IDs as the old task and the remaining classes as the new task.

---

## 3. Image Preprocessing

Two different transformations are used for training and evaluation.

### Training transformations

Training images are:

* Resized
* Randomly cropped
* Randomly horizontally flipped
* Augmented using ColorJitter
* Converted to tensors
* Normalized using ImageNet mean and standard deviation

```python
RandomResizedCrop(224)
RandomHorizontalFlip()
ColorJitter()
ToTensor()
Normalize()
```

### Evaluation transformations

For evaluation, deterministic preprocessing is used:

```python
Resize(224, 224)
ToTensor()
Normalize()
```

This ensures that random augmentation does not affect test results.

---

# 4. Model Architecture

The backbone used in the experiment is **ResNet18**, trained from scratch.

The assignment allows either ResNet18 or ResNet50.

The implementation removes the original ResNet fully connected layer and replaces it with a new classifier.

```text
Input Image
     |
     v
 ResNet18
     |
     v
Feature Extractor F
     |
     v
Feature Vector
     |
     v
Linear Classifier G
     |
     v
Class Scores
```

The feature extractor produces a feature vector of dimension `f`, while the classifier maps that feature vector to the required number of classes.

For the initial model:

$$
G_0 \in \mathbb{R}^{20 \times f}
$$

After adding the new classes:

$$
G_1 \in \mathbb{R}^{37 \times f}
$$

This corresponds directly to the formulation in the assignment.

The implementation creates the feature extractor by replacing the original ResNet FC layer with an identity layer and then adds a new linear classifier.

---

# 5. Training Strategy

The experiment follows three main stages.

## Stage 1: Learning the Initial 20 Classes

The first model is trained only on the 20 old classes.

```text
20 classes
    |
    v
ResNet18
    |
    v
20-class classifier
```

The model is trained using cross-entropy loss with label smoothing.

### Cross-Entropy Loss

For a classification problem with \(C\) classes:

$$
\mathcal{L}_{CE}
=
-\sum_{c=1}^{C} y_c \log(p_c)
$$

where:

* \(y_c\) is the target label
* \(p_c\) is the predicted probability for class \(c\)
* \(C\) is the number of classes

The implementation uses:

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

Training uses SGD with momentum, weight decay, Nesterov acceleration, and cosine annealing learning-rate scheduling.

After training, the resulting feature extractor and classifier are saved conceptually as:

$$
F_0,\ G_0
$$

These represent the model's knowledge before learning the new classes.

---

# 6. Stage 2: Naive Incremental Learning

The next step is to introduce the 17 new classes.

Instead of training a completely new model, the original classifier is expanded.

### Before expansion

$$
G_0 \in \mathbb{R}^{20 \times f}
$$

### After expansion

$$
G_1 \in \mathbb{R}^{37 \times f}
$$

The weights corresponding to the original 20 classes are copied into the new classifier, while the additional 17 output neurons are newly initialized.

```text
             Old classifier
              20 outputs
                  |
                  v
            Expand FC layer
                  |
        +---------+---------+
        |                   |
   Old 20 outputs      New 17 outputs
        |                   |
        +---------+---------+
                  |
            37 outputs
```

The implementation explicitly copies the old classifier weights and biases into the first 20 rows of the new classifier.

---

## Naive Fine-Tuning

The expanded model is then fine-tuned using only the training data belonging to the 17 new classes.

This is the simplest incremental learning strategy.

```text
Old knowledge
     |
     v
Expanded ResNet18
     |
     v
Train only on new 17 classes
     |
     v
37-class model
```

The model is evaluated separately on:

* All 37 classes
* Old 20 classes
* New 17 classes

The implementation calculates the decrease in old-class performance as:

$$
\text{Forgetting}
=
Acc_{old}^{before}
-
Acc_{old}^{after}
$$

where:

* \(Acc_{old}^{before}\) = accuracy of the original 20-class model
* \(Acc_{old}^{after}\) = accuracy on the same 20 classes after learning the new classes

This directly measures catastrophic forgetting.

---

# 7. Why Does Catastrophic Forgetting Occur?

During Stage 2, the model is optimized using only examples from the 17 new classes.

Therefore, there is no direct training signal telling the model:

> "Do not change the representations and predictions that were useful for the original 20 classes."

Gradient updates therefore modify the feature extractor and classifier toward the new classes.

Conceptually:

$$
\theta_{new}
=
\theta_{old}
-
\eta\nabla_{\theta}\mathcal{L}_{new}
$$

where:

* \(\theta\) represents model parameters
* \(\eta\) is the learning rate
* \(\mathcal{L}_{new}\) is the loss calculated on the new classes

Because the optimization objective contains information about the new classes but not the old classes, parameters useful for old-class recognition can change.

This causes the model's performance on previously learned classes to decrease.

That decrease is the **catastrophic forgetting effect** the assignment asks us to investigate.

---

# 8. Stage 3: Exemplar Replay

To reduce forgetting, a small number of examples from the old classes are retained.

The assignment specifies:

> Store 5 examples per class from the original 20 classes.

Therefore:

$$
20 \times 5 = 100
$$

old examples are retained as exemplars.

These examples are denoted as:

$$
E_0
$$

The implementation uses **herding** to select representative examples from each old class rather than simply selecting random images.

---

# 9. Herding-Based Exemplar Selection

The purpose of exemplar selection is to choose a small set of images that represents each old class well.

First, the images are passed through the original feature extractor.

The features are normalized:

$$
z_i =
\frac{F_0(x_i)}
{\|F_0(x_i)\|}
$$

For each class, the mean feature vector is calculated:

$$
\mu_c
=
\frac{1}{N_c}
\sum_{i=1}^{N_c} z_i
$$

The algorithm then greedily selects examples whose running feature mean is closest to the class mean.

The implementation therefore attempts to choose representative examples rather than arbitrary samples.

---

# 10. Knowledge Distillation

Exemplar replay alone provides a small amount of old-class data.

Knowledge distillation adds another constraint:

> The updated model should preserve the predictions produced by the original model on the old examples.

The original model acts as a **teacher**:

$$
G_0(F_0(x))
$$

The updated model acts as a **student**:

$$
G_1(F_1(x))
$$

The assignment specifies a distillation objective encouraging the new model to preserve the old model's outputs on the exemplars.

---

# 11. Distillation Loss Used in the Implementation

The implementation uses KL divergence between the teacher and student distributions.

First, the logits are softened using a temperature \(T\).

Teacher distribution:

$$
p_T
=
softmax\left(\frac{z_T}{T}\right)
$$

Student distribution:

$$
p_S
=
softmax\left(\frac{z_S}{T}\right)
$$

The KL-divergence loss is:

$$
\mathcal{L}_{KD}
=
T^2
D_{KL}(p_T \parallel p_S)
$$

or:

$$
\mathcal{L}_{KD}
=
T^2
\sum_i
p_T(i)
\log
\frac{p_T(i)}
{p_S(i)}
$$

The factor \(T^2\) is used to maintain an appropriate gradient scale when temperature scaling is applied.

In the implementation:

```python
temperature = 2.0
alpha = 1.0
```

and the final optimization objective is:

$$
\mathcal{L}
=
\mathcal{L}_{classification}
+
\alpha\mathcal{L}_{KD}
$$

The classification loss ensures that the model learns the new classes, while the distillation loss encourages it to preserve the old model's behavior on the exemplars.

---

# 12. Combined Incremental Learning Approach

The complete approach can therefore be summarized as:

```text
                    Dataset
                       |
             +---------+---------+
             |                   |
          Old 20              New 17
          classes              classes
             |                   |
             v                   v
        Train ResNet18      New-class data
             |
             v
          F0 + G0
             |
             +-------------------+
             |                   |
       Store exemplars       Expand classifier
       5/class               20 -> 37
             |                   |
             +---------+---------+
                       |
                       v
             Fine-tune on new data
                       +
             Exemplar replay
                       +
             Knowledge distillation
                       |
                       v
                  F1 + G1
                       |
                       v
                Evaluate 37 classes
```

---

# 13. Experimental Comparison

Three models are effectively evaluated:

| Model                   | Training Data          | Purpose                         |
| ----------------------- | ---------------------- | ------------------------------- |
| Base model              | Old 20 classes         | Establish initial performance   |
| Naive incremental model | New 17 classes         | Measure catastrophic forgetting |
| Distillation model      | New 17 + old exemplars | Reduce catastrophic forgetting  |

The most important comparison is between the naive model and the model using exemplar replay plus knowledge distillation.

The code reports:

* Total accuracy on 37 classes
* Accuracy on the old 20 classes
* Accuracy on the new 17 classes
* Forgetting relative to the original model

---

## Results

### Base Model

The ResNet18 model was first trained from scratch on the initial 20 classes.

| Metric                |     Result |
| --------------------- | ---------: |
| Old 20-class accuracy | **49.47%** |

This serves as the baseline for measuring catastrophic forgetting after incremental learning.

### Incremental Learning Comparison

| Metric                      |  No Distillation | With Distillation |
| --------------------------- | ---------------: | ----------------: |
| Total accuracy (37 classes) |       **27.42%** |        **33.47%** |
| Old 20-class accuracy       |        **0.00%** |        **17.31%** |
| New 17-class accuracy       |       **59.60%** |        **52.43%** |
| Forgetting vs. base         | **49.47 points** |  **32.16 points** |

### Interpretation

The results clearly demonstrate the catastrophic forgetting problem.

After expanding the classifier from 20 to 37 classes and fine-tuning only on the 17 new classes, the **old-class accuracy dropped from 49.47% to 0.00%**. This corresponds to **49.47 percentage points of forgetting**.

The model performs reasonably well on the new classes, achieving **59.60% accuracy**, but completely loses its ability to correctly classify the original 20 classes. This shows that naive fine-tuning strongly favors the newly introduced classes when no old-class examples are available during training.

Adding **exemplar replay and knowledge distillation** improves retention of the previously learned knowledge. Old-class accuracy increases from **0.00% to 17.31%**, reducing forgetting from **49.47 to 32.16 percentage points**.

The total 37-class accuracy also improves from **27.42% to 33.47%**, an improvement of **6.05 percentage points**.

However, the new-class accuracy decreases from **59.60% to 52.43%**. This trade-off is expected: the distillation objective constrains the updated model to preserve the behavior of the original model while it is simultaneously learning the new classes.

Overall, the experiment shows that **exemplar replay combined with knowledge distillation mitigates catastrophic forgetting**, although it does not completely eliminate it.

### Key Observation

The most important comparison is the old-class performance:

$$
49.47\% \rightarrow 0.00\%
$$

with naive fine-tuning, compared with:

$$
49.47\% \rightarrow 17.31\%
$$

when exemplar replay and knowledge distillation are introduced.

Thus, the proposed mitigation strategy preserves substantially more of the original knowledge than naive incremental fine-tuning.


---

# 15. Comparison Plot

The implementation generates the following comparison plot:

```text
distillation_comparison.png
```

The plot compares:

* Total 37-class accuracy
* Old 20-class accuracy
* New 17-class accuracy

between:

* No Distillation
* With Distillation

A horizontal reference line also represents the original 20-class base-model accuracy.

The plotting code saves the figure using:

```python
plt.savefig("distillation_comparison.png", dpi=150)
```

Add the generated plot to the repository and display it in the README using:

<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/ceb9ce44-4f55-4e62-9556-91d6bfc9cff7" />


---

# 16. What Was Tried

The experiment follows a progressively stronger approach rather than directly applying distillation.

### Approach 1: Train from scratch

A ResNet18 is trained on the initial 20 classes.

Purpose:

* Establish a baseline
* Measure how well the model learns the initial task
* Obtain the original feature extractor \(F_0\) and classifier \(G_0\)

---

### Approach 2: Naive incremental fine-tuning

The classifier is expanded:

$$
20 \rightarrow 37
$$

The model is then fine-tuned using only the 17 new classes.

Purpose:

* Simulate a realistic incremental learning scenario
* Measure how much old knowledge is lost

Expected issue:

$$
\text{New knowledge} \uparrow
\quad
\text{Old knowledge} \downarrow
$$

This demonstrates catastrophic forgetting.

---

### Approach 3: Exemplar replay + knowledge distillation

Five representative examples are retained for every old class.

$$
5 \times 20 = 100 \text{ exemplars}
$$

These examples are combined with new-class training batches.

The model is trained using:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+
\alpha\mathcal{L}_{KD}
$$

Purpose:

* Continue learning the new 17 classes
* Replay a small amount of old data
* Preserve the behavior of the original model
* Reduce catastrophic forgetting

---

# 17. Hyperparameters

| Parameter                      |              Value |
| ------------------------------ | -----------------: |
| Architecture                   |           ResNet18 |
| Pretrained                     |                 No |
| Old classes                    |                 20 |
| New classes                    |                 17 |
| Total classes                  |                 37 |
| Batch size                     |                 32 |
| Initial training epochs        |                 30 |
| Incremental training epochs    |                 15 |
| Initial learning rate          |               0.01 |
| Incremental learning rate      |              0.005 |
| Optimizer                      |                SGD |
| Momentum                       |                0.9 |
| Weight decay                   | \(5\times10^{-4}\) |
| Label smoothing                |                0.1 |
| LR scheduler                   |   Cosine Annealing |
| Exemplars per old class        |                  5 |
| Total exemplars                |                100 |
| Distillation weight \(\alpha\) |                1.0 |
| Temperature \(T\)              |                2.0 |
| Random seed                    |                 42 |

These values are taken directly from the implementation.

---

# 18. Key Concepts

### Incremental Learning

Incremental learning is the process of updating a model as new classes or data become available without completely retraining the model from the beginning.

In this experiment:

$$
20\ classes
\rightarrow
37\ classes
$$

---

### Catastrophic Forgetting

Catastrophic forgetting occurs when learning new information causes a neural network to lose previously acquired knowledge.

In this experiment:

$$
Acc_{old}^{before}
>
Acc_{old}^{after}
$$

indicates forgetting.

---

### Exemplar Replay

Exemplar replay stores a small number of representative samples from previously learned classes.

Here:

$$
5\ examples/class
$$

are stored for the original 20 classes.

---

### Knowledge Distillation

Knowledge distillation transfers knowledge from a previously trained teacher model to a newly updated student model.

Here:

$$
Teacher = (F_0,G_0)
$$

and

$$
Student = (F_1,G_1)
$$

The student is encouraged to preserve the teacher's output distribution on the old exemplars.

---

### Herding

Herding is used to select representative samples from each old class based on their feature representations.

Instead of randomly storing five images, the implementation selects samples whose feature mean is close to the overall class feature mean.

---

# 19. Final Takeaway

This experiment demonstrates the central challenge of incremental learning.

Simply expanding the classifier from 20 to 37 classes and fine-tuning on the new classes allows the model to learn the new classes, but the absence of old training examples can cause the model to forget previously learned classes.

The second approach addresses this problem using a small memory of representative old examples and knowledge distillation:

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{CE}
+
\alpha\mathcal{L}_{KD}
}
$$

The classification loss promotes learning of the new classes, while the distillation loss constrains the updated model to remain consistent with the original model on the retained exemplars.

The experiment therefore illustrates an important principle of continual/incremental learning:

> Learning new knowledge should not require completely sacrificing previously acquired knowledge.



