# **Neuroimaging (ADNI Brain MRI) Using ConvNeXt**
**Author:** Lawrence Tiang  
**Student Number:** 48922104  
**Implementation Difficulty:** Hard


## **1. Overview**
### **1.1 Problem & Motivation**
Describe the problem being addressed, why it is important.
- refer to tasksheet for problem description

### **1.2 Proposed Algorithm**
Describe the proposed algorithm/model and explain what problem it solves.
- how does convnext solve this problem?

### **1.3 How It Works**
Explain the main steps of the algorithm: input -> preprocessing -> model inference -> final output.

### **1.4 High-level Visualisation**
Figure of pipeline of the proposed approach


## **2. Feasibility Review**
### **2.1 User Need, Scope & Acceptance Criteria**
#### **Intended User**
Describe the intended user of the system

#### **Prototype Purpose**  
The context in which the prototype would be used.

#### **Acceptance Criteria**
1. a
2. a
3. a
4. a

3–5 testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource
constraint (e.g., peak VRAM, runtime).

### **2.2 Model Choice & Course Concepts**
#### **Proposed Dataset Configuration**  
- which dataset used (dataset name) and where can find
- number of samples and classes
- training, validation and testing splits
- splitting strategy (random?)
- state if data augmentation applied to training and test set

#### **Model Architecture**
Describe the proposed model architecture and its main components.

#### **Course Concepts & Literature**
Explain how relevant course concepts and academic literature support the preprocessing methods, model architecture, and experimental approach.
- Equations
- Find sources

### **2.3 Preliminary Feasibility Evidence**
#### **Initial Data Audit**
- Class distribution.
- example images from dataset

#### **Baseline Smoke Test & Resource Estimate**
Describe the initial experiment used to determine whether the proposed approach is technically feasible.
- Training time.
- GPU/CPU requirements.
- Memory requirements.
- Expected inference cost or latency.

### **2.4 Risks, Budget & Fallback**
#### **Technical Risks**
- Overfitting.
- Data leakage.
- Insufficient training data.
- Poor generalisation.
- Computational limitations.
- High inference latency.

#### **Computing Budget**
- GPU
- CPU
- Maximum practical training time

#### **Next Planned Experiment**
Describe the next experiment you plan to conduct and explain why it is the most useful next step.
Describe a fallback approach


## **3. Dependencies and Reproducibility**
### **3.1 Software Requirements**
| Dependency | Version |
|---|---|
| Python | [Version] |
| PyTorch | [Version] |
| NumPy | [Version] |
| Pandas | [Version] |
| scikit-learn | [Version] |
| Matplotlib | [Version] |

### **3.2 Reproducing Results**
address reproducibility of results (random seeds)
submitting slurm script
- running python train.py and python predict.py


## **4. Data Preprocessing**
### **4.1 Data Augmentation**
Describe all preprocessing performed on the data.
- data augmentation and description of what it does

Include references where preprocessing methods are based on existing literature.

### **4.2 Training, Validation & Testing Splits**
Describe the final dataset split in detail.

| Split | Number of Samples | Percentage |
|---|---:|---:|
| Training | [Value] | [X]% |
| Validation | [Value] | [Y]% |
| Testing | [Value] | [Z]% |

- How the split was performed.
- Whether the split was random, stratified, subject-wise, time-based, etc.
- How class proportions were maintained.
- How data leakage was prevented.

Where multiple samples originate from the same subject, subject-level splitting will be used to prevent information leakage between the training and testing sets.


## **5. Experimental Runs**
### **5.1 Failed Runs**
- figures
- test accuracy
- What was attempted
- Why it failed
- What was learned
- What was changed as a result

### **5.2 Final Experimental Configuration**
| Parameter | Value |
|---|---|
| Batch size | [Value] |
| Learning rate | [Value] |
| Number of epochs | [Value] |
| Optimiser | [Value] |
| Loss function | [Value] |
| Learning-rate scheduler | [Value] |
| Random seed | [Value] |


## **6. ConvNeXt Outputs & Visualisations**
### **6.1 ConvNeXt Prediction Results**
- correct predictions examples

### **6.2 ConvNeXt Performance Visualisations**
- Training/validation curves (loss and accuracy)
- Confusion matrix (base and convnext)
- roc curve


## **7. Open Research Dilemma Investigation**
### **7.1 Quantitative Benchmarking**
| Metrics | Baseline | ConvNeXt |
|---|---:|---:|
| Accuracy | [Value] | [Value] |
| F1-Score | [Value] | [Value] |
| AUROC | [Value] | [Value] |
| AD Precision | [Value] | [Value] |
| NC Precision | [Value] | [Value] |
| AD Recall | [Value] | [Value] |
| NC Recall | [Value] | [Value] |

Compare the proposed model against the implemented baseline model using identical evaluation splits and evaluation conditions.
Discuss the observed differences between the baseline and proposed model.
- Whether the proposed approach achieved the intended performance target.
- Possible reasons for the observed results.

### **7.2 Resource Profiling**
| Metrics | Baseline | ConvNeXt |
|---|---:|---:|
| Peak GPU VRAM | [Value] | [Value] |
| Parameter Count | [Value] | [Value] |
| Inference Latency | [Value] | [Value] |

- Training time.

Discuss the practical implications of the resource measurements.

### **7.3 Qualitative Error Autopsies**
Analyse 3–5 representative failure cases from the dataset.  
Explain why the model likely produced the incorrect prediction and what this reveals about the model's limitations.
- Failure prediction examples
- What went wrong
- Root trigger (The likely characteristic of the image/data that caused the error)

The MRI contained structural characteristics that may have overlapped with patterns learned from AD training examples. The differences between normal ageing and early pathological changes can be subtle, making this sample difficult to distinguish from AD cases.
- Failure mode: NC --> AD class confusion (false positive)

### **7.4 Engineering Recommendation & Trade-off Synthesis**
Summarise the practical implications of the experimental findings.  
Discuss the trade-offs between:
- Model performance and computational cost.
- Accuracy and inference latency.
- Model complexity and maintainability.
- Resource requirements and deployment constraints.
- Performance and robustness.

Provide an actionable recommendation for the project manager based on the experimental evidence.
- risk coverage plot

### **7.5 Limitations & Future Development**
As dot points is fine


## **8. Artificial Intelligence Usage Disclosure**


## **9. References**