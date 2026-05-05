**Vision Language Models for Tumor Analysis**

**Spatial Guidance and Multi-Task Learning for Classification, Segmentation, and Morphological Regression**

This repository contains the implementation of a hierarchical multi-task framework designed to improve the diagnostic reliability of brain tumor grading, specifically for Lower-Grade Gliomas (LGG). By anchoring classification in explicit spatial and physical evidence, we move beyond "black-box" models towards interpretable and accountable medical AI.

**Overview**

Accurate grading of LGG is vital for clinical decision-making. Standard deep learning classifiers often provide labels without spatial justification. Our research explores the transformation of Vision-Language Models (VLMs) into autonomous diagnostic agents that can explain their reasoning through internal spatial grounding.

**Core Objectives**

**Multi-Task Synergy:** Integrating classification, pixel-level segmentation, and morphological regression into a single Comprehensive Multi-Task Model (CMM).

**Internal Grounding:** Demonstrating that internalizing the diagnostic process is more effective for reducing hallucinations than relying on external expert guidance.

**Visual Interference:** Analyzing how external prompts can sometimes distract specialized medical models.

**Physical Regularization:** Leveraging morphological properties like Area, Mass, and Diameter as quantitative regularizers.

**Modeling Strategies**

We evaluated four distinct strategies to measure how task complexity affects diagnostic precision:

**Grade Classification Model (GCM):** A baseline VQA-based model for hierarchical tumor grading.

**Joint Classification and Segmentation Model (JCSM):** A configuration that forces the model to justify its diagnosis by generating a tumor mask.

**Joint Classification and Morphology-Aware Model (JCMM):** A framework focusing on the physical scale of the pathology through auxiliary regression tasks.

**Comprehensive Multi-Task Model (CMM):** Our most advanced architecture that unifies classification, segmentation, and regression.

**Evaluated Backbones**

LLaVA-1.5-7B (General Purpose)

Qwen2.5-VL-7B (General Purpose)

Lingshu-7B (Medical Domain)

LLaVA-Med-7B (Medical Domain)

**Research Findings**

Our analysis shows that the CMM strategy provides the most robust results by treating identification, localization, and quantification as interdependent variables.

**The "Visual Interference" Phenomenon**

One of the most significant findings was that for specialized models like LLaVA-Med, external visual cues (like tumor contours) actually hindered performance. This suggests that as models become more specialized, they develop superior internal attention mechanisms that are more precise than external expert delineations.

**Experimental Setup**

**Dataset:** LGG MRI Segmentation Dataset sourced from TCIA and TCGA.

**Sample Size:** 3,929 MRI slices partitioned at the patient level to prevent data leakage.

**Fine-Tuning:** Low-Rank Adaptation (LoRA) with rank 32 and alpha 64.

**Evaluation Metrics:**

**Classification** Accuracy, Precision, Recall, and F1-Score.

**Segmentation** Accuracy via Intersection over Union (IoU).

**Regression** Precision using the R-squared coefficient

**Conclusion**

This research confirms that diagnostic precision in neuro-oncology is an emergent property of spatial and physical awareness. By requiring models to solve for a pathology's identity, location, and scale simultaneously, we achieve a level of accountability and accuracy that far exceeds traditional classification methods.

_For More Information Please wait for the paper._
