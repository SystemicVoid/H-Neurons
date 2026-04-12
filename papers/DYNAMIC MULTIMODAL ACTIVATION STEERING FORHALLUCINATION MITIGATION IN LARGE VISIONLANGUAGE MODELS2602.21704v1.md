## DYNAMIC MULTIMODAL ACTIVATION STEERING FOR HALLUCINATION MITIGATION IN LARGE VISIONLANGUAGE MODELS

Jianghao Yin, Qin Chen ∗ , Kedi Chen, Jie Zhou ∗ , Xingjiao Wu, Liang He

East China Normal University jhyin@stu.ecnu.edu.cn, qchen@cs.ecnu.edu.cn, jzhou@cs.ecnu.edu.cn

## ABSTRACT

Large Vision-Language Models (LVLMs) exhibit outstanding performance on vision-language tasks but struggle with hallucination problems. Through in-depth analysis of LVLM activation patterns, we reveal two key findings: 1) truthfulness and visual perception capabilities predominantly engage different subsets of attention heads within the model architecture; and 2) truthfulness steering vectors vary significantly across different semantic contexts. Based on these observations, we propose Dynamic Multimodal Activation Steering, a training-free approach for hallucination mitigation. Our method constructs a semantic-based truthfulness steering vector database and computes visual perception steering vectors, enabling context-aware interventions during inference by dynamically selecting the most relevant steering vectors based on input semantic similarity and applying them to the most influential attention heads. We conduct comprehensive experiments across multiple models and datasets, demonstrating that our approach significantly enhances model performance, outperforming existing state-of-the-art methods.

## 1 INTRODUCTION

Large Vision-Language Models (LVLMs) have demonstrated remarkable performance on visual question answering (VQA), image captioning, and related tasks (Liu et al., 2023; 2024c; Bai et al., 2023; Chen et al., 2024; Dai et al., 2023). However, these models suffer from significant hallucination phenomena (Huang et al., 2025; Chen et al., 2025b), manifested as fabricating non-existent objects or incorrectly describing image content (Liu et al., 2024b; Bai et al., 2024). Such hallucinations limit the applicability of LVLMs in downstream applications including autonomous driving (Cui et al., 2024), robotics (Li et al., 2024b), and other safety-critical domains.

Due to the complex architecture of LVLMs, the causes of multimodal hallucinations are diverse. To address these multimodal hallucination issues, numerous approaches have been proposed (Leng et al., 2024; Huang et al., 2024; An et al., 2025; Yin et al., 2024; Liu et al., 2024a), which can be broadly categorized into two classes: training-based and decoding-based methods. Training-based methods primarily focus on constructing less biased datasets to fine-tune LVLMs, such as LRV (Liu et al., 2024a), or employing reinforcement learning to train LVLMs, as demonstrated by RLHF-V (Yu et al., 2024a). The limitations of these approaches lie in their requirements for carefully curated data and substantial computational resources, as well as the need to retrain models separately for different architectures. Decoding-based methods, on the other hand, modify the decoding strategies of LVLMs, such as VCD (Leng et al., 2024) and ICD (Wang et al., 2024c). While these methods avoid the need for training, they often compromise the quality of the generated content (Yin et al., 2025).

More recently, researchers have begun investigating activation engineering (Zou et al., 2023; Li et al., 2023b; Wang et al., 2024b) as an alternative approach to reduce hallucinations through targeted intervention in model representations. ICT (Chen et al., 2025a) is an image-object cross-level trusted intervention method that mitigates model hallucinations by applying noise to both images and objects, thereby enhancing the model's attention to visual information. However, this approach

* Corresponding authors.

primarily focuses on visual-level interventions, neglecting the multimodal characteristics of LVLMs. VTI (Liu et al., 2025b) intervenes in the hidden states of both the visual encoder and large language model during inference by pre-computing steering vectors for visual and textual modalities. Nevertheless, this method employs fixed steering vectors regardless of input variation, ignoring potential semantic differences across diverse contexts. The uniformly applied steering vectors fail to account for the nuanced semantic variations that exist across different inputs.

To address these challenges, we propose dynamic multimodal activation steering (DMAS), a training-free approach for hallucination mitigation in LVLMs. Our method focuses on two types of attention heads in LVLMs: truthfulness-related and visual perception-related. For truthfulness heads, we explicitly model how truthfulness steering vectors vary across semantic contexts. We cluster data semantically and create sample pairs with and without hallucinations within each cluster. By contrasting attention activations between factual and hallucination-prone samples, we extract truthfulness steering vectors. These vectors are stored alongside their cluster embeddings in a keyvalue database. For visual perception, we calculate activation differences between noise-free and noisy image inputs to derive perception steering vectors that enhance visual attention. During inference, we dynamically retrieve the most semantically relevant truthfulness steering vector for the input query and apply both truthfulness and visual perception vectors to the top-K attention heads with the largest activation differences. This dual intervention effectively reduces hallucinations. The main contributions of our paper are:

- We investigate activation differences in LVLMs, revealing that truthfulness and visual perception capabilities predominantly engage different subsets of attention heads, and demonstrate that truthfulness vectors vary significantly across different semantic contexts through visualization, indicating the necessity for dynamic rather than static intervention approaches.
- We propose dynamic multimodal activation steering, a training-free method for hallucination mitigation that constructs a semantic-based truthful steering vector database and visual perception steering vector, enabling context-aware interventions during inference by dynamically selecting appropriate steering vectors based on input semantic similarity.
- We conduct comprehensive experiments on multiple models across discriminative tasks and openended generation datasets. The experimental results demonstrate that our method achieves significant improvements: increasing total scores by 94.66 on MME and reducing 20.2% hallucinations on CHAIR, outperforming existing state-of-the-art methods. These results highlight the effectiveness of our approach in hallucination mitigation.

## 2 RELATED WORK

## 2.1 LARGE VISION-LANGUAGE MODELS

Large Vision-Language Models (LVLMs) have recently undergone rapid development, achieving excellent performance in image captioning and VQA tasks (Yin et al., 2023; Jin et al., 2024). They typically consist of a vision encoder, a connection layer, and an LLM. As for the vision encoder, the VIT from CLIP (Radford et al., 2021) is commonly used. For the connection layer, some models use simple MLP layers for alignment, such as LLaVA (Liu et al., 2023; 2024c), Shikra (Chen et al., 2023), PandaGPT (Su et al., 2023), etc.; some models use Q-former for alignment, like BLIP2 (Li et al., 2023a), InstructBLIP (Dai et al., 2023), etc.; while others design special architectures. However, these LVLMs suffer from serious hallucination problems, and effectively eliminating hallucinations remains a popular research topic.

## 2.2 HALLUCINATION MITIGATION FOR LVLMS

Recently, numerous approaches have been proposed to mitigate multimodal hallucinations (Liu et al., 2024b; Bai et al., 2024), addressing this issue across three key stages: training, inference, and post-processing. At the training stage, some research concentrates on constructing better data to train models. For example, LRV (Liu et al., 2024a) constructs a high-quality instruction fine-tuning dataset containing balanced positive and negative samples, while other studies introduce reinforcement learning to the multimodal domain to reduce hallucination, such as RLHF-V (Yu et al., 2024a) and RLAIF-V (Yu et al., 2024b). These methods typically require carefully constructed training data

<!-- image -->

(a) The difference of truthfulness attention head activation.

- (b) The difference of visual perception attention head activation.
- (c) Steering vectors visualization.

Figure 1: Activation differences in LLaVAv1.5.

and consume substantial computational resources during training. Research on mitigating hallucinations at the inference stage often requires no training. VCD (Leng et al., 2024) uses the distribution from noise-added images and the original output distribution to jointly determine the final distribution to mitigate hallucination. ICD (Wang et al., 2024c) reduces hallucinations by contrasting output distributions between standard and deliberately disturbed instructions. These contrastive decoding methods often compromise the quality of the generated content (Yin et al., 2025). Post-processing approaches correct the generated content from LVLMs to achieve hallucination reduction effects. For instance, LURE (Zhou et al., 2024) constructs a dataset to train a hallucination revisor. However, these methods require the construction of a complex pipeline and increase the time required to obtain final outputs. To overcome these limitations, we propose dynamic multimodal activation steering, a training-free approach to mitigate hallucination in LVLMs by dynamically intervening in attention heads during inference time.

## 3 PRELIMINARY STUDY

To understand the internal mechanisms underlying multimodal hallucinations, we conduct a systematic analysis of attention patterns in LLaV Av1.5 (Liu et al., 2024c) across 3,000 samples from the SEED (Li et al., 2024a) and AMBER (Wang et al., 2023) datasets. Our investigation focuses on identifying which attention heads are most sensitive to truthfulness versus visual perception.

We design two complementary experiments to isolate attention mechanisms responsible for different aspects of multimodal processing. In the first experiment, we examine truthfulness related attention head by contrasting model activations when processing identical visual inputs paired with text prompts either with ground truth or hallucinated answers. This approach enables us to identify attention heads most relevant to truthfulness, we measure how each head's activation changes between truthful and hallucinated content by computing the difference: truthful activation minus hallucinated activation. In the second experiment, we investigate visual perception related attention head by comparing activations between clean images and their noise-corrupted counterparts, calculating activation differences by subtracting the activation values of non-noisy inputs from those with noise. As shown in Figures 1a and 1b, the activation patterns differ significantly between these two experiments. For truthfulness (Figure 1a), the most active attention heads appear predominantly in layer 30. In contrast, for visual perception (Figure 1b), the highest activation differences concentrate in layer 31. These distinct activation patterns provide a foundation for our targeted intervention approach that addresses both aspects simultaneously.

Furthermore, we divide the SEED and AMBER datasets into four semantic clusters and compute the activation differences for each cluster. Using t-SNE to visualize these differences in a twodimensional space (Figure 1c), we observe a clear separation between clusters, with each occupying a distinct region in the projection space. This separation indicates that truthfulness direction vectors vary significantly across different semantic contexts. The heterogeneity in these patterns suggests that a static intervention approach would be insufficient, as it cannot account for the semanticdependent nature of hallucinations. This observation directly motivates our dynamic multimodal activation steering method, which can adaptively select appropriate steering vectors based on the semantic content of the input query.

## 4 METHOD

In this section, we introduce dynamic multimodal activation steering. As shown in Figure 2, the method has three steps: the first step is to establish a dynamic truthfulness steering vector database, the second step is to calculate the steering vector for the model's visual perception attention heads, and the third step is to apply dynamic interventions to different attention heads during inference.

## 4.1 TRUTHFULNESS STEERING VECTOR DATABASE

We select the AMBER (Wang et al., 2023) and SEED (Li et al., 2024a) datasets as our data sources and divide the datasets into 4 clusters based on semantics. The questions in these two datasets are in the form of multiple-choice and discriminative questions, making it easy for us to create hallucinated answers for each sample (for discriminative questions, we change the answer to the opposite; for multiple-choice questions, we randomly select an incorrect option). Each cluster C i comprises the question prompt T , visual input V , ground truth response Y pos , and incorrect response Y neg for every sample.

We input ( V, T + Y pos ) and ( V, T + Y neg ) separately into LVLMs and preserve the attention head activation values of the last token at each layer, denoted as A pos and A neg . We define the truthfulness steering vector as the activation difference between non-hallucinated outputs and hallucinated outputs within each cluster according to Equation 1:

$$D _ { i } = \frac { 1 } { | C _ { i } | } \sum _ { j \in C _ { i } } ( A _ { p o s , j } - A _ { n e g , j } ) & & ( 1 )$$

| C i | represents the number of samples in cluster C i , and j indexes the samples within the cluster. Subsequently, we apply principal component analysis (PCA) to D i to reduce insignificant noise, thereby extracting the principal components that influence truthfulness. The magnitude of D i effectively quantifies the significance of each attention head in governing this specific model behavior.

Next, we construct a truthful steering vector database where the average embedding representation of questions from each cluster serves as the key, with the corresponding steering vector D i as the value. During inference, our approach dynamically matches the semantic content of the input question to retrieve the most semantically similar steering vector, enabling context-appropriate interventions. We obtain key embeddings in the database and input text embeddings via sentence transformer.

Figure 2: Overview of the DMAS framework.

<!-- image -->

## 4.2 VISUAL PERCEPTION STEERING VECTOR

Given a visual input V and a distorted visual input V ′ (obtained by adding noise to the image following the forward diffusion process (Ho et al., 2020)), we first input V into an object detector YOLOv11 (Khanam &amp; Hussain, 2024) to obtain objects O present in the image, and insert them into a simple template 'The image depicts { objects } ' , denoted as Y O . Then we randomly select an equal number of objects O ′ from a predefined object library within the same object category that are not in O , inserting them into template, denoted as Y O ′ . The prompt T is fixed as 'Please describe this image.' Next, we obtain the final inputs ( V, T + Y O ) and ( V ′ , T + Y O ′ ) , and input these two samples separately into LVLMs, preserving the attention head activation values of the last token at each layer, denoted as A v and A v ′ respectively. We define visual perception steering vector as the activation difference between visual input and distorted visual input according to Equation 2:

$$D _ { v } = A _ { v } - A _ { v ^ { \prime } }$$

Similarly, we apply PCA to D v to reduce noise, thereby extracting the principal components most relevant to visual perception.

## 4.3 DYNAMIC INTERVENTION FOR INFERENCE

During the inference phase, for a given text input T and visual input V , we dynamically retrieve the most appropriate steering vector by computing semantic similarity between the input and each key in database as shown in Equation 3.

$$D _ { f } = D _ { \hat { i } } , \, \text {where} \, \hat { i } = \arg \max _ { i } \sin ( E ( T ) , K e y _ { i } )$$

i

where E ( T ) is the embedding representation of the input text, Key i represents the key embedding for cluster i , and sim ( · , · ) denotes the cosine similarity function. This process identifies the most relevant truthfulness steering vector for the current input.

To achieve more precise control over model behavior, rather than intervening on all attention heads, we selectively target the most influential heads in both D f and D v .We define binary mask matrices M f and M v as Equation 4:

$$M _ { \{ f , v \} } ^ { ( l , h ) } = \begin{cases} 1 , & \text {if } ( l , h ) \in \text {Top} K ( \mathbf D _ { \{ f , v \} } , K ) \\ 0 , & \text {otherwise} \end{cases}$$

where ( l, h ) denotes the h -th attention head in the l -th layer, D represents the sum of activation differences for each attention head in D and TopK ( D { f,v } , K ) returns the indices of the K largest attention heads in either D f or D v , representing the most influential attention heads for truthfulness and visual perception respectively.

Building upon the standard attention mechanism, we modify the computation for layers where intervention is applied. Our intervention-enhanced computation is formulated as Equation 5:

$$x ^ { ( l + 1 ) } & = x ^ { ( l ) } + \text {Concat} _ { ( 0 \sim H ) } \left [ \text {Attn} ^ { ( l , h ) } ( x ^ { ( l ) } ) \\ & \quad + \alpha \cdot M _ { f } ^ { ( l , h ) } \cdot D _ { f } ^ { ( l , h ) } \\ & \quad + \beta \cdot M _ { v } ^ { ( l , h ) } \cdot D _ { v } ^ { ( l , h ) } \right ] \cdot \text {W} _ { o } ^ { ( l ) }$$

where x ( l ) represents the hidden states at the l -th layer, H is the number of attention heads per layer, α and β are hyperparameters controlling the intervention strength for truthfulness and visual perception respectively. The binary masks ensure that interventions are only applied to the most influential attention heads, allowing for precise and targeted steering of the model's behavior.

## 5 EXPERIMENTAL SETUP

## 5.1 DATASETS AND EVALUATION METRICS

To comprehensively evaluate our proposed approach, we test our method on discriminative tasks, including MME (Fu et al., 2023) and POPE (Li et al., 2023c), as well as on open-ended generation tasks using CHAIR (Rohrbach et al., 2018).

Table 1: Results on MME. The best results are shown in bold.

| Model     | Method   |   Existence ↑ |   Count ↑ |   Position ↑ |   Color ↑ |   Total Scores ↑ |
|-----------|----------|---------------|-----------|--------------|-----------|------------------|
| LLaVAv1.5 | Regular  |        175.67 |    124.67 |       114    |    151    |           565.33 |
| LLaVAv1.5 | VCD      |        184.66 |    138.33 |       128.67 |    153    |           604.66 |
| LLaVAv1.5 | OPERA    |        180.67 |    133.33 |       123.33 |    155    |           592.33 |
| LLaVAv1.5 | VAF      |        195    |    158.33 |       128.33 |    155    |           636.67 |
| LLaVAv1.5 | DECO     |        185    |    153.33 |       118.33 |    155    |           611.66 |
| LLaVAv1.5 | DAMO     |        195    |    150    |       143.33 |    165    |           653.33 |
| LLaVAv1.5 | AGLA     |        195    |    153.89 |       129.44 |    161.67 |           640    |
| LLaVAv1.5 | ICT      |        190    |    160.43 |       128.67 |    170    |           649.1  |
| LLaVAv1.5 | Ours     |        195    |    158.33 |       133.33 |    173.33 |           659.99 |
| QwenVL    | Regular  |        155    |    127.67 |       131.67 |    173    |           587.33 |
| QwenVL    | VCD      |        156    |    131    |       128    |    181.67 |           596.67 |
| QwenVL    | VAF      |        165    |    155    |       133.33 |    175    |           628.33 |
| QwenVL    | ICT      |        180    |    145    |       108.33 |    173.33 |           606.66 |
| QwenVL    | Ours     |        170    |    145    |       133.33 |    185    |           633.33 |

MME (Fu et al., 2023) is a comprehensive evaluation benchmark for LVLMs, comprising 14 subtasks. For questions in this dataset, models are required to respond with either 'yes' or 'no'. Following Yin et al. (2024) and Leng et al. (2024), we select 'existence', 'count', 'position', and 'attribute' as the hallucination test sets. Consistent with Fu et al. (2023), we adopt the sum of accuracy and accuracy+ as the evaluation metrics.

POPE (Li et al., 2023c) is a benchmark designed specifically to evaluate object hallucination. The benchmark features three sampling strategies of varying difficulty levels: random (randomly sampling nonexistent objects), popular (selecting frequently appearing objects), and adversarial (selecting objects that frequently co-occur with objects present in the image). We report Accuracy, Precision, Recall, F1 Score as the evaluation metrics.

CHAIR (Rohrbach et al., 2018) is an open-ended generation task. This benchmark comprises 500 images sourced from MSCOCO (Lin et al., 2014), where LVLMs are required to generate captions for the images, followed by evaluation of hallucinations present in these captions at sentence level CHAIR S and image level CHAIR I .

## 5.2 BASELINES AND IMPLEMENTATION DETAILS

We validate the effectiveness on mainstream LVLMs: LLaVAv1.5 7B (Liu et al., 2024c) and QwenVL 7B (Bai et al., 2023), and compare DMAS with state-of-the-art methods: ICT (Chen et al., 2025a), AGLA (An et al., 2025), VAF (Yin et al., 2025), VTI (Liu et al., 2025b), DECO (Wang et al., 2024a), DAMO (Wang et al., 2025), VCD (Leng et al., 2024), and OPERA (Huang et al., 2024).

Our method has three key parameters: α , β , and K . α and β respectively regulate the intensity of interventions for truthfulness and visual perception, while K refers to the intervention on the top K most active attention heads. We set the range of α and β to { 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } , and the range of K to { 32, 64, 128, 256, 512, 1024 } , and employ grid search to determine the parameters. In our experiments, we set the temperature to 0 and top p to 1. All experiments are conducted on NVIDIA RTX 4090(48GB) GPUs.

## 6 EXPERIMENT

## 6.1 RESULTS ON MME

The results on the MME (Fu et al., 2023) dataset are presented in Table 1. Our method demonstrates significant improvements of 94.66 and 46 points compared to the baseline models LLaVAv1.5 and QwenVL, respectively. On the LLaVAv1.5 model, our approach outperforms the existing state-

of-the-art method ICT (Chen et al., 2025a) by 10.89 points, while on QwenVL, it surpasses the current state-of-the-art method V AF (Yin et al., 2025) by 5 points. Across all subtasks, we observe notable improvements over regular baselines, which can be attributed to our dynamic intervention mechanism that retrieves the most semantically similar steering vector for each query.

## 6.2 RESULTS ON POPE

Table 2: Results on POPE. Best results are in bold, and second-best values are underlined.

| Dataset   | Setting   | Method   |   Accuracy ↑ | Precision   | Recall   |   F1 Score ↑ |
|-----------|-----------|----------|--------------|-------------|----------|--------------|
| MSCOCO    | LLaVAv1.5 | Regular  |        81.38 | 88.04       | 72.78    |        79.65 |
|           |           | VCD      |        84.33 | 85.93       | 83.28    |        84.52 |
|           |           | OPERA    |        84.21 | 88.23       | 79.79    |        83.72 |
|           |           | VAF      |        86.9  | 89.43       | 83.77    |        86.47 |
|           |           | AGLA     |        85.82 | 93.78       | 76.83    |        84.44 |
|           |           | VTI      |        86.48 | 90.11       | 82.09    |        85.9  |
|           |           | ICT      |        87.35 | -           | -        |        87.12 |
|           |           | Ours     |        86.81 | 87.23       | 86.57    |        86.79 |
|           | QwenVL    | Regular  |        83.71 | 93.30       | 72.69    |        81.7  |
|           | QwenVL    | VCD      |        86.67 | 90.66       | 81.94    |        83.04 |
|           | QwenVL    | OPERA    |        84.26 | 94.40       | 73.52    |        82.65 |
|           |           | AGLA     |        83.9  | 96.20       | 70.62    |        81.44 |
|           |           | VTI      |        85.18 | 91.31       | 78.18    |        84.08 |
|           |           | ICT      |        87.53 | -           | -        |        86.98 |
|           |           | Ours     |        87.63 | 87.92       | 87.3     |        87.65 |
| GQA       | LLaVAv1.5 | Regular  |        78.33 | 79.33       | 79.13    |        79.13 |
| GQA       | LLaVAv1.5 | VCD      |        81.16 | 77.31       | 89.08    |        82.67 |
| GQA       | LLaVAv1.5 | OPERA    |        80.8  | -           | -        |        83.24 |
| GQA       | LLaVAv1.5 | VAF      |        83.67 | 81.50       | 88.00    |        84.5  |
| GQA       | LLaVAv1.5 | AGLA     |        84.41 | 84.63       | 84.67    |        84.55 |
| GQA       | LLaVAv1.5 | ICT      |        85.27 | -           | -        |        85.5  |
| GQA       | LLaVAv1.5 | Ours     |        85.27 | 83.86       | 87.51    |        85.63 |
| GQA       | QwenVL    | Regular  |        77.47 | 81.54       | 71.37    |        76.06 |
| GQA       | QwenVL    | VCD      |        82.48 | 81.73       | 83.93    |        82.77 |
| GQA       |           | OPERA    |        82.74 | -           | -        |        82.68 |
| GQA       |           | AGLA     |        81.14 | 86.87       | 73.53    |        79.63 |
| GQA       |           | ICT      |        83.28 | -           | -        |        83.26 |
| GQA       |           | Ours     |        84.4  | 85.19       | 83.53    |        84.32 |

The experimental results of POPE (Li et al., 2023c) are shown in Table 2. We conduct experiments on MSCOCO (Lin et al., 2014) and GQA (Hudson &amp; Manning, 2019) under random, popular, and adversarial settings. Table 2 presents the average results across these three settings, with detailed experimental results provided in the Appendix. Our method improves LLaVAv1.5's performance on MSCOCO by 5.43% in accuracy and 7.14% in F1 score, while for QwenVL, it achieved improvements of 3.92% in accuracy and 5.95% in F1 score. On GQA, our method enhances LLaVAv1.5 by 6.94% in accuracy and 6.5% in F1 score, and improves QwenVL by 6.93% in accuracy and 8.26% in F1 score. Compared to existing methods, our approach achieves best results in most cases, demonstrating its significant effectiveness in mitigating object hallucination. Notably, while the ICT (Chen et al., 2025a) method applies noise to objects in images to increase the LVLMs' attention to these objects, our method achieves superior performance in most cases without such specialized design elements.

## 6.3 RESULTS ON CHAIR

We evaluate our method on open-ended generation tasks, with experimental results on CHAIR (Rohrbach et al., 2018) presented in Table 3. Our method reduces hallucinations by 20.2 at the

Figure 3: Impact of key hyperparameters.

<!-- image -->

sentence level (CHAIR S ) and by 3.8 at the image level (CHAIR I ). Compared to existing methods, our approach reduces sentence-level hallucinations by 5 points over the state-of-the-art method VTI (Liu et al., 2025b), and matching VTI's performance on image-level hallucinations. In summary, our method achieves significant improvements in hallucination mitigation on both discriminative tasks and open-ended generation tasks.

Table 3: Results on CHAIR.

| Method   |   CHAIR S ↓ |   CHAIR I ↓ | Method                  | CHAIR   | CHAIR   | POPE   | POPE   |
|----------|-------------|-------------|-------------------------|---------|---------|--------|--------|
| Regular  |        51   |        15.2 |                         |         | C ↓     | Acc ↑  | F1 ↑   |
| VCD      |        51   |        14.9 |                         | C S ↓   | I       |        |        |
| OPERA    |        47   |        14.6 | Ours                    | 30.8    | 11.4    | 81.70  | 82.47  |
| DECO     |        37.8 |        11.1 | w/o visual vector       | 34.2    | 11.7    | 81.67  | 82.42  |
| AGLA     |        43   |        14.1 | w/o truthfulness vector | 42.4    | 13.2    | 81.40  | 82.01  |
| VTI      |        35.8 |        11.1 | w/o both                | 51.0    | 15.2    | 75.08  | 76.06  |
| Ours     |        30.8 |        11.4 |                         |         |         |        |        |

## 6.4 FURTHER ANALYSIS

## 6.4.1 ABLATION STUDIES

To demonstrate the effectiveness of using both truthfulness steering vector and visual perception steering vector, we compare the results when utilizing only one intervention at a time. We conduct experiments on LLaVAv1.5 using the CHAIR and POPE. As shown in Table 4, 'w/o visual vector' indicates intervention with only the truthfulness steering vector, while 'w/o truthfulness vector' indicates intervention with only the visual perception steering vector. We observe that even when using only one intervention method, there is a notable improvement compared to the Regular baseline (w/o both). Furthermore, each intervention method exhibits hallucination mitigation effects on both discriminative and generation tasks. The optimal results are achieved when the two interventions are combined.

## 6.4.2 EFFECT OF DYNAMIC INTERVENTION

To validate our designed strategy of dynamically invoking the truthfulness steering vector based on semantics, we compare our method with combining all truthfulness steering vectors into a single fixed steering vector for intervention. We conduct experiments on QwenVL and LLaVAv1.5 using the MME, with results shown in Figure 4. We observe that dynamically invoking steering vectors based on semantics achieves optimal performance across all subtasks. When using a fixed steering vector, the improvement is smaller than with our method, and it even underperforms the original model on the Position subtask of QwenVL, which demonstrates the necessity of our designed dynamic invocation strategy.

## 6.4.3 IMPACT OF HYPERPARAMETERS

Table 4: Ablation studies on CHAIR and POPE. C S represents CHAIR S , C I represents CHAIR I .

In this section, we investigate the impact of key parameters α , β , and K on the experimental results. Here, α and β control the intervention strength, while K denotes the number of attention heads receiving intervention. Experiments conducted on QwenVL using the POPE GQA random subset are shown in Figure 3. Figure 3a illustrates the relationship between F1 score and parameters α and β . When α and β are negative, we observe a decrease in F1 score, which effectively represents intervention in the opposite direction, pushing activations toward hallucination. As α and β increase, F1 score exhibits an upward trend; however, when α and β become excessively large, F1 score shows a

Figure 4: Effect of dynamic intervention.

<!-- image -->

precipitous decline, indicating that the model's fundamental capabilities become impaired. Figure 3b shows how F1 varies with the number of intervened attention heads, revealing similar patterns for both truthfulness and visual perception attention heads. Few intervened heads produce minimal impact with no significant F1 improvement. As intervention extends to more heads, F1 score increases, but excessive intervention causes a dramatic decline in F1, indicating degradation of model performance.

## 6.4.4 IMPACT OF CLUSTERS

In this section, we investigate the impact of cluster quantity on the performance of our proposed method. We vary the number of clusters across { 2, 3, 4, 5 } and conduct experiments on both QwenVL and LLaVAv1.5 using the MME benchmark. The experimental results are presented in Figure 3c. We observe that both LLaVAv1.5 and QwenVL achieve optimal performance when the number of clusters is set to 4. When the cluster count is insufficient, the semantic granularity becomes too coarse for effective representation.

## 6.4.5 ANALYSIS OF GENERALITY

To verify the generalizability of our method, we tested our approach on ScienceQA (Lu et al., 2022) which is subject-based VQA dataset and ViQuAE (Lerner et al., 2022) which is a knowledge-based VQA dataset. The accuracy are shown in Table 5. Our method also achieved significant improvements on these datasets. These datasets are completely different from the dataset types we used to construct the steering vector, which demonstrates the generalizability of our method.

Table 5: Generality on ScienceQA and ViQuAE.

| Method   | ScienceQA   | ScienceQA   | ViQuAE    | ViQuAE   |
|----------|-------------|-------------|-----------|----------|
| Method   | LLaVAv1.5   | QwenVL      | LLaVAv1.5 | QwenVL   |
| Regular  | 52.75       | 46.41       | 43.38     | 50.09    |
| VTI      | 51.46       | -           | 42.29     | -        |
| ICT      | 52.95       | -           | 42.47     | -        |
| Ours     | 62.27       | 48.04       | 56.00     | 54.08    |

## 7 CONCLUSION

This paper proposes dynamic multimodal activation steering, a training-free approach to mitigate hallucination in LVLMs by dynamically intervening in attention head activations. The experiments on multiple benchmarks demonstrate the effectiveness of our method, with LLaVAv1.5 achieving a remarkable 94.66-point improvement on MME and reducing 20.2% hallucinations on CHAIR, outperforming existing SOTA methods. We compare the experimental performance of our proposed semantic-dynamic strategy for steering vector selection against fixed steering vector approaches, demonstrating the effectiveness and necessity of semantic-dynamic steering vector selection.

## ACKNOWLEDGMENTS

This research is funded by the National Nature Science Foundation of China (No. 62477010, No.62577022 and No.62307028), the Natural Science Foundation of Shanghai (No. 23ZR1441800), Shanghai Science and Technology Innovation Action Plan (No. 24YF2710100 and No.23YF1426100),Shanghai Qiji Zhifeng Co., Ltd. (2025-GZL-RGZN-01001) and the opening funding of the State Key Laboratory of DisasterReduction in Civil Engineering (Grant No. SLDRCE24-03).

## REFERENCES

- Wenbin An, Feng Tian, Sicong Leng, Jiahao Nie, Haonan Lin, QianYing Wang, Ping Chen, Xiaoqin Zhang, and Shijian Lu. Mitigating object hallucinations in large vision-language models with assembly of global and local attention. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 29915-29926, 2025.
- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966 .
- Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 , 2025.
- Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930 , 2024.
- Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Linfeng Zhang, Lijie Wen, and Xuming Hu. Ict: Image-object cross-level trusted intervention for mitigating object hallucination in large vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 4209-4221, 2025a.
- Kedi Chen, Qin Chen, Jie Zhou, Xinqi Tao, Bowen Ding, Jingwen Xie, Mingchen Xie, Peilong Li, and Zheng Feng. Enhancing uncertainty modeling with semantic graph for hallucination detection. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pp. 23586-23594, 2025b.
- Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm's referential dialogue magic. arXiv preprint arXiv:2306.15195 , 2023.
- Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 , 2024.
- Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, et al. A survey on multimodal large language models for autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pp. 958-979, 2024.
- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. URL https://arxiv.org/abs/2305.06500 .
- Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR , abs/2306.13394, 2023. doi: 10. 48550/ARXIV.2306.13394. URL https://doi.org/10.48550/arXiv.2306.13394 .

- Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 1437514385, 2024.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACMTransactions on Information Systems , 43(2):1-55, 2025.
- Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 13418-13427, 2024.
- Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 6700-6709, 2019.
- Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. Transactions on Machine Learning Research .
- Yizhang Jin, Jian Li, Yexin Liu, Tianjun Gu, Kai Wu, Zhengkai Jiang, Muyang He, Bo Zhao, Xin Tan, Zhenye Gan, et al. Efficient multimodal large language models: A survey. arXiv preprint arXiv:2405.10739 , 2024.
- Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725 , 2024.
- Hugo Laurenc ¸on, Andr´ es Marafioti, Victor Sanh, and L´ eo Tronchon. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637 , 2024.
- Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 13872-13882, 2024.
- Paul Lerner, Olivier Ferret, Camille Guinaudeau, Herv´ e Le Borgne, Romaric Besanc ¸on, Jos´ e G Moreno, and Jes´ us Lov´ on Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval , pp. 3108-3120, 2022.
- Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 13299-13308, 2024a.
- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning , pp. 19730-19742. PMLR, 2023a.
- Kenneth Li, Oam Patel, Fernanda Vi´ egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems , 36:41451-41530, 2023b.
- Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, and Hao Dong. Manipllm: Embodied multimodal large language model for objectcentric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 18061-18070, 2024b.

- Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pp. 292-305, 2023c.
- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision-ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13 , pp. 740-755. Springer, 2014.
- Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. In The Twelfth International Conference on Learning Representations , 2024a.
- Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253 , 2024b.
- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems , 36:34892-34916, 2023.
- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 26296-26306, 2024c.
- Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. Phd: A chatgpt-prompted visual hallucination evaluation dataset. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 19857-19866, 2025a.
- Sheng Liu, Haotian Ye, and James Zou. Reducing hallucinations in large vision-language models via latent space steering. In The Thirteenth International Conference on Learning Representations , 2025b.
- Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems , 35:2507-2521, 2022.
- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pp. 8748-8763. PMLR, 2021.
- Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing , pp. 4035-4045, 2018.
- Haz Sameen Shahgir, Khondker Salman Sayeed, Abhik Bhattacharjee, Wasi Uddin Ahmad, Yue Dong, and Rifat Shahriyar. Illusionvqa: A challenging optical illusion dataset for vision language models. arXiv preprint arXiv:2403.15952 , 2024.
- Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. In Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the era of Interactive Assistants! , pp. 11-23, 2023.
- Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 9568-9578, 2024.
- Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, and Huajun Chen. Mllm can see? dynamic correction decoding for hallucination mitigation. arXiv preprint arXiv:2410.11779 , 2024a.

- Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, et al. Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397 , 2023.
- Kaishen Wang, Hengrui Gu, Meijun Gao, and Kaixiong Zhou. Damo: Decoding by accumulating activations momentum for mitigating hallucinations in vision-language models. In The Thirteenth International Conference on Learning Representations , 2025.
- Weixuan Wang, Jingyuan Yang, and Wei Peng. Semantics-adaptive activation intervention for llms via dynamic steering vectors. arXiv preprint arXiv:2410.12299 , 2024b.
- Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics ACL 2024 , pp. 15840-15853, 2024c.
- Hao Yin, Guangzong Si, and Zilei Wang. Clearsight: Visual signal enhancement for object hallucination mitigation in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 14625-14634, 2025.
- Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549 , 2023.
- Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences , 67(12):220105, 2024.
- Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 13807-13816, 2024a.
- Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, Xiaocheng Feng, Jun Song, Bo Zheng, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness, 2024b. URL https://arxiv.org/abs/2405.17220 .
- Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning , pp. 57730-57754, 2024c.
- Ce Zhang, Zifu Wan, Zhehan Kan, Martin Q Ma, Simon Stepputtis, Deva Ramanan, Russ Salakhutdinov, Louis-Philippe Morency, Katia P Sycara, and Yaqi Xie. Self-correcting decoding with generative feedback for mitigating hallucinations in large vision-language models. In The Thirteenth International Conference on Learning Representations , 2025.
- Jianfei Zhao, Feng Zhang, Xin Sun, and Chong Feng. Mitigate language priors in large visionlanguage models by cross-images contrastive decoding. arXiv e-prints , pp. arXiv-2505, 2025.
- Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. In The Twelfth International Conference on Learning Representations , 2024.
- Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 , 2025a.
- Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. Ibd: Alleviating hallucinations in large vision-language models via image-biased decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 1624-1633, 2025b.
- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405 , 2023.

## A APPENDIX

## A.1 AI WRITING ASSISTANCE STATEMENT

Large language models were utilized solely for minor linguistic improvements, including enhanced phrasing and clarity. These tools played no role in content generation, experimental design, data analysis, or interpretation. The authors are entirely responsible for all ideas, results, and conclusions presented in this paper.

## A.2 MORE DETAILS ON CHAIR

In this paper, we report CHAIR S and CHAIR I as evaluation metrics. The calculation of CHAIR S and CHAIR I is shown in Equation 6, where we set the maximum number of new tokens to 512 in our experiments.

$$CHAIR _ { S } = \frac { | \{ \text {sentences with h allucinated objects} \} | } { | \{ \text {all sentences} \} | } \\ \text {CHAIR} _ { I } = \frac { | \{ \text {all lucinated objects} \} | } { | \{ \text {all objects mentioned} \} | }$$

## A.3 RESULTS ON POPE

The complete experimental results on POPE are presented in Table 6. Our method achieves significant improvements across all three experimental settings: random, popular, and adversarial.

Table 6: Results on LLaVAv1.5. The best results are shown in bold.

| Dataset   | Setting     | Method   |   Accuracy ↑ |   Precision |   Recall |   F1 Score ↑ |
|-----------|-------------|----------|--------------|-------------|----------|--------------|
| MSCOCO    |             | Regular  |        83.29 |       92.13 |    72.8  |        81.33 |
|           | Random      | VCD      |        87.73 |       91.42 |    83.28 |        87.16 |
|           |             | Ours     |        90.03 |       90.51 |    90.03 |        90.02 |
|           | Popular     | Regular  |        81.88 |       88.93 |    72.8  |        80.06 |
|           |             | VCD      |        85.38 |       86.92 |    83.28 |        85.06 |
|           |             | Ours     |        87.33 |       89.16 |    85    |        87.03 |
|           | Adversarial | Regular  |        78.96 |       83.06 |    72.75 |        77.57 |
|           |             | VCD      |        80.88 |       79.45 |    83.29 |        81.33 |
|           |             | Ours     |        83.07 |       82.04 |    84.67 |        83.33 |
|           |             | Regular  |        83.73 |       87.16 |    79.12 |        82.95 |
|           | Random      | VCD      |        86.65 |       84.85 |    89.24 |        86.99 |
|           |             | Ours     |        89.57 |       88.92 |    90.4  |        89.6  |
| GQA       |             | Regular  |        78.17 |       77.64 |    79.12 |        78.37 |
| GQA       | Popular     | VCD      |        80.73 |       76.26 |    89.24 |        82.24 |
| GQA       |             | Ours     |        84.53 |       83.51 |    86.07 |        84.77 |
| GQA       |             | Regular  |        75.08 |       73.19 |    79.16 |        76.06 |
| GQA       | Adversarial | VCD      |        76.09 |       70.83 |    88.75 |        78.78 |
| GQA       |             | Ours     |        81.7  |       79.15 |    86.07 |        82.47 |

## A.4 RESULTS ON AMBER

We conduct an evaluation of LLaVA v1.5 on the AMBER (Wang et al., 2023). AMBER contains both discriminative tasks and generative tasks. The experimental results are shown in Table 7. Our method outperforms existing methods on both discriminative and generative tasks, achieving significant effects in hallucination mitigation.

Table 7: Results on AMBER. The best results are shown in bold.

|                           | Discriminative   | Discriminative   | Generative   | Generative   |               |
|---------------------------|------------------|------------------|--------------|--------------|---------------|
| Method                    | Acc ↑            | F1 ↑             | CHAIR ↓      | Hal ↓        | AMBER SCORE ↑ |
| Regular                   | 67.4             | 71.2             | 11.6         | 47.7         | 79.80         |
| VCD (Leng et al., 2024)   | 68.1             | 71.1             | 9.8          | 43.8         | 80.65         |
| ICD (Wang et al., 2024c)  | 70.3             | 73.4             | 8.8          | 38.7         | 82.3          |
| IBD (Zhu et al., 2025b)   | 69.2             | 72.2             | 9.8          | 42.2         | 81.2          |
| DeFG (Zhang et al., 2025) | 70.2             | 73.0             | 9.1          | 39.9         | 81.95         |
| CICD (Zhao et al., 2025)  | 71.1             | 73.1             | 6.6          | 34.8         | 83.25         |
| Ours                      | 81.9             | 87.2             | 4.9          | 20.9         | 90.01         |

Figure 5: Effect of dynamic intervention.

<!-- image -->

## A.5 SCALABILITY ANALYSIS ACROSS MODEL SIZES

To verify that our method has hallucination mitigation effects for models of different sizes, we select the discriminative task dataset MME and the generative task dataset CHAIR on LLaVAv1.5 7B and 13B models. The experimental results show in Table 8 that our model achieves significant effects for models of different sizes in both discriminative and generative tasks.

|              |              | MME           | MME           | MME        | MME     | MME            | CHAIR     | CHAIR     |
|--------------|--------------|---------------|---------------|------------|---------|----------------|-----------|-----------|
| Model        | Method       | Existence ↑   | Count ↑       | Position ↑ | Color ↑ | Total Scores ↑ | CHAIR S ↓ | CHAIR I ↓ |
| LLaVAv1.5 7B | Regular Ours | 175.67 195.00 | 124.67 158.33 | 114.00     | 151.00  | 565.33 659.99  | 51.0 30.8 | 15.2 11.4 |
|              |              |               |               | 133.33     | 173.33  |                |           |           |
| LLaVA1.5 13B | Regular      | 185.00        | 131.67        | 95.00      | 175.00  | 586.67         | 45.0      | 11.8      |
|              | Ours         | 185.00        | 158.33        | 103.33     | 180.00  | 626.66         | 38.0      | 10.8      |

Table 8: Scalability Analysis Across Model Sizes on MME and CHAIR. The best results are shown in bold.

## A.6 ANALYSIS OF INFERENCE SPEED

In this section, we investigate DMAS's inference speed. We use LLaVAv1.5 7B and set the generation content lengths to { 64, 128, 256 } respectively, then compare the inference speed of our method with the original model and VCD. The experimental results show in Figure 5 that our method has faster inference speed compared to the decoding method VCD. VCD's inference latency is almost twice that of the original model, but our model achieves better hallucination mitigation effects while adding only a small amount of inference time.

## A.7 CASE STUDY

To intuitively demonstrate the hallucination mitigation effectiveness of our method, we conduct case studies on LLaVAv1.5. We utilize cases from MME and CHAIR datasets, with results shown

in Figure 6. Our method effectively mitigates multimodal hallucination issues across both VQA tasks and image captioning tasks. For VQA tasks, we present various question types, demonstrating our method's effectiveness in reducing hallucinations at different levels including object, attribute, relation, and count. For image captioning tasks, our method not only generates fewer hallucinations but also maintains the quality of the output content.

<!-- image -->

<!-- image -->

<!-- image -->

Question ： Are there a white dish in the image? Please answer yes or no.

Original model ： No.

Ours ： Yes.

Question ： Is there only one dog in the image? Please answer yes or no.

Original model No.

## ：

Ours ： Yes.

## Original model ：

The image features a large room with a unique design, resembling a barn or a store. The room is filled with various cow statues, including a large cow statue in the center and several smaller cow statues scattered throughout the space. The cow statues are positioned in different areas of the room, creating a visually interesting scene.

## Ours ：

The image features a large, open room with a fake ceiling that resembles a green field. In this room, there are two fake cows hanging from the ceiling, creating a unique and eye-catching display. The room also has a few potted plants placed around the space, adding a touch of nature to the environment.

Figure 6: Case study on LLaVAv1.5.

## A.8 EXPERIMENTS ON MORE BENCHMARKS

To further substantiate the effectiveness of our approach, we conduct extensive experiments across multiple challenging benchmarks, including MMVP (Tong et al., 2024), IllusionVQA (Shahgir et al., 2024), HallusionBench (multimodal subset) (Guan et al., 2024), and MM-Vet (Yu et al., 2024c), evaluating on both LLaVAv1.5 and QwenVL. The results summarized in the Table 9 confirm that our method delivers strong and robust performance across these diverse and complex evaluations.

| Model     | Method       | MMVP      | MM-Vet    | IllusionVQA   | HallusionBench   |
|-----------|--------------|-----------|-----------|---------------|------------------|
| LLaVAv1.5 | Vanilla      | 30.3 31.0 | 38.9 33.4 | 32.6 33.1     | 50.2 52.7        |
| LLaVAv1.5 | ICT          |           |           |               |                  |
| LLaVAv1.5 | Ours         | 54.3      | 39.3      | 35.2          | 53.7             |
| QwenVL    | Vanilla Ours | 41.0      | 32.9      | 31.7          | 42.4             |
| QwenVL    |              | 43.3      | 34.4      | 33.6          | 42.9             |

Table 9: Experiments on more benchmarks.

<!-- image -->

<!-- image -->

Question ： Is the cricket bat under the batter's body Please answer yes or no.

Original model ： Yes.

Ours ： No.

Question ： Is there a backpack in this image? Please answer yes or no.

Original model ： No.

Ours ： Yes.

## A.9 COMPREHENSIVE ANALYSIS OF THE CONSTRUCTED DATASET

In this subsection, we will explore how the dataset scale and the dataset composition for constructing steering vectors respectively affect the effectiveness of DMAS.

## A.9.1 DATASET SCALE

To investigate the impact of dataset size on steering vector construction, we systematically reduce the dataset to 75%, 50%, and 25% of its original size and evaluated performance on MME (Fu et al., 2023), AMBER (Wang et al., 2023), and MMVP (Tong et al., 2024). The results are presented in Table 10. The results show that dataset size has minimal impact on performance, with only minor fluctuations across benchmarks. This demonstrates that our method is robust to variations in dataset scale and does not require large amounts of labeled data.

Table 10: Analysis of dataset scale for steering vectors construction.

|       | MME         | MME     | MME        | MME     | MMVP       | AMBER      | AMBER   |
|-------|-------------|---------|------------|---------|------------|------------|---------|
| Scale | Existence ↑ | Count ↑ | Position ↑ | Color ↑ | Accuracy ↑ | Relation ↑ | F1 ↑    |
| 100%  | 195.00      | 158.33  | 133.33     | 173.33  | 54.3       | 69.0       | 87.2    |
| 75%   | 195.00      | 158.33  | 133.33     | 168.33  | 52.0       | 69.0       | 87.2    |
| 50%   | 195.00      | 158.33  | 133.33     | 168.33  | 52.0       | 69.0       | 87.2    |
| 25%   | 195.00      | 158.33  | 133.33     | 168.33  | 53.0       | 69.0       | 87.1    |

## A.9.2 DATASET COMPOSITION

To explore how different datasets affect steering vector construction, we build databases using two alternative datasets: POPE (Li et al., 2023c) and PHD (Liu et al., 2025a). POPE contains only object existence questions (homogeneous), while PHD includes diverse question types similar to our original datasets. Both databases are constructed using the same scale as our main experiments, and we evaluate performance on MME (Fu et al., 2023), AMBER (Wang et al., 2023), and MMVP (Tong et al., 2024). The results are presented in Table 11. Results show that the POPE-based database underperforms on several MME subtasks and AMBER's relation subtask, which is expected given POPE's limited diversity cannot fully exploit our dynamic approach. In contrast, the PHD-based database achieves comparable results to our original method across all benchmarks. These findings demonstrate two important insights: (1) when using datasets with good question diversity, our method achieves consistent performance regardless of the specific dataset chosen, and (2) this validates the necessity of our dynamic retrieval mechanism for selecting appropriate steering vectors based on semantic context.

Table 11: Analysis of dataset composition for steering vectors construction.

|       | MME         | MME     | MME        | MME     | MMVP       | AMBER      | AMBER   |
|-------|-------------|---------|------------|---------|------------|------------|---------|
| Scale | Existence ↑ | Count ↑ | Position ↑ | Color ↑ | Accuracy ↑ | Relation ↑ | F1 ↑    |
| Ours  | 195.00      | 158.33  | 133.33     | 173.33  | 54.3       | 69.0       | 87.2    |
| POPE  | 195.00      | 158.33  | 123.33     | 153.33  | 52.3       | 65.1       | 86.0    |
| PHD   | 195.00      | 158.33  | 133.33     | 173.33  | 53.7       | 68.3       | 86.9    |

## A.10 EXPERIMENTS ON DIFFERENT DECODING STRATEGIES

To explore the effectiveness of our method under different decoding strategies, including beam search (beam=2), nucleus sampling (temperature=0.5, topp=0.7) and greedy search, the results on MME and MMVP benchmarks are presented in the Table 12. Our method consistently demonstrates effective hallucination mitigation across all decoding settings, confirming the robustness of our approach.

Table 12: Experiments on different decoding strategies.

|                     |              | MME           | MME     | MME           | MME           | MME            | MMVP       |
|---------------------|--------------|---------------|---------|---------------|---------------|----------------|------------|
| Decoding Strategies | Method       | Existence ↑   | Count ↑ | Position ↑    | Color ↑       | Total Scores ↑ | Accuracy ↑ |
| Beam                | Vanilla Ours | 190.00 191.67 | 153.33  | 113.33 135.00 | 148.33 171.66 | 604.99 654.99  | 29.7 55.1  |
|                     |              |               | 156.66  |               |               |                |            |
| Nucleus             | Vanilla      | 190.00        | 147.78  | 123.89        | 150.55        | 612.22         | 32.3       |
|                     | Ours         | 190.00        | 150.00  | 130.00        | 159.44        | 629.44         | 53.1       |
| Greedy              | Vanilla      | 190.00        | 153.33  | 113.33        | 148.33        | 604.99         | 30.3       |
| Greedy              | Ours         | 195.00        | 158.33  | 133.33        | 173.33        | 659.99         | 54.3       |

## A.11 EXPERIMENTS ON MORE MODELS

In this subsection, we test the effectiveness of our method on several advanced LVLMs. We select Qwen2.5-VL (Bai et al., 2025), InternVL3 (Zhu et al., 2025a), Mantis (Jiang et al.), and Idefics3 (Laurenc ¸on et al., 2024), and conduct experiments on MME and CHAIR using greedy search. The experimental results are shown in Table 13. Our method remains effective for these advanced LVLMs with different architectures, demonstrating hallucination mitigation effects on both discriminative tasks and open-ended generation tasks, which confirms the scalability of our approach.

Table 13: Experiments on more models.

|            |              | MME            | CHAIR     | CHAIR     |
|------------|--------------|----------------|-----------|-----------|
| Model      | Method       | Total Scores ↑ | CHAIR S ↓ | CHAIR I ↓ |
| Qwen2.5-VL | Vanilla Ours | 698.33 718.33  | 47.4 43.2 | 10 9.5    |
| InternVL3  | Vanilla Ours | 698.33 713.33  | 28.4 26.4 | 7.6 6.9   |
| Mantis     | Vanilla Ours | 653.33 668.33  | 44.6 34.8 | 12.7 9.7  |
| Idefics3   | Vanilla Ours | 665.00 678.33  | 49.9 45.2 | 8.8 8.2   |

## A.12 ABLATION STUDIES ON HALLUCINATION OBJECTS SELECTION STRATEGIES

For visual perception attention heads, we compare two hallucination objects selection strategies: (1) random selection across categories (e.g., people → panda), and (2) hard negatives within the same category (e.g., runner → swimmer). As shown in the Table 14, experiments on MME and CHAIR benchmarks demonstrate that both strategies achieve comparable performance with no significant difference. This indicates that our visual steering vector approach is robust to the choice of negative selection strategy.

Table 14: Ablation studies on hallucination objects selection strategies.

|               | MME         | MME     | MME        | MME     | MME            | CHAIR     | CHAIR     |
|---------------|-------------|---------|------------|---------|----------------|-----------|-----------|
| Strategy      | Existence ↑ | Count ↑ | Position ↑ | Color ↑ | Total Scores ↑ | CHAIR S ↓ | CHAIR I ↓ |
| Random        | 195.00      | 158.33  | 133.33     | 173.33  | 659.99         | 31.2      | 10.9      |
| Hard Negative | 195.00      | 158.33  | 133.33     | 173.33  | 659.99         | 30.8      | 11.4      |