## PROBING CLASSIFIERS ARE UNRELIABLE FOR CONCEPT REMOVAL AND DETECTION

## Abhinav Kumar

Microsoft Research t-abkumar@microsoft.com

## Chenhao Tan

University of Chicago chenhao@uchicago.edu

## ABSTRACT

Neural network models trained on text data have been found to encode undesirable linguistic or sensitive concepts in their representation. Removing such concepts is non-trivial because of a complex relationship between the concept, text input, and the learnt representation. Recent work has proposed post-hoc and adversarial methods to remove such unwanted concepts from a model's representation. Through an extensive theoretical and empirical analysis, we show that these methods can be counterproductive: they are unable to remove the concepts entirely, and in some cases may fail severely by destroying all task-relevant features. The reason is the methods' reliance on a probing classifier as a proxy for the concept. Even under the most favorable conditions for learning a probing classifier when a concept's relevant features in representation space alone can provide 100% accuracy, we prove that a probing classifier is likely to use non-concept features and thus post-hoc or adversarial methods will fail to remove the concept correctly. These theoretical implications are confirmed by experiments on models trained on synthetic, Multi-NLI, and Twitter datasets. For sensitive applications of concept removal such as fairness, we recommend caution against using these methods and propose a spuriousness metric to gauge the quality of the final classifier.

## 1 Introduction

Neural models in text classification have been shown to learn spuriously correlated features [16, 27] or embed sensitive attributes like gender or race [8, 7, 6] in their representation layer. Classifiers that use such sensitive or spurious concepts (henceforth concepts ) raise concerns of model unfairness and out-of-distribution generalization failure [39, 2, 15]. Removing the influence of these concepts is non-trivial because the classifiers are based on hard-to-interpret deep neural networks. Moreover, since many concepts cannot be modified at the input tokens level, removal methods that work at the representation layer have been proposed: 1) post-hoc removal [7, 49, 14] on a pre-trained model (e.g., null space projection [32]), and 2) adversarial removal [15, 48, 12] by jointly training the main task classifier with an (adversarial) classifier for the concept.

In this paper, we theoretically show that both these classes of methods can be counter-productive in real-world settings where the main task label is often correlated with the concept. Examples include natural language inference ( main task) where the presence of negation (spurious concept ) may be correlated with the 'contradicts' label; or tweet sentiment classification ( main task) where the author's gender (sensitive concept ) maybe correlated with the sentiment label. Our key result is based on the observation that both these methods internally use an auxiliary (or probing) classifier [1, 42] that aims to predict the spurious concept based on the representation learnt by the main classifier.

We show that an auxiliary classifier cannot be a reliable signal on whether the representation includes features that are causally derived from the concept. As previous work has argued [4, 44, 40, 2], if the representation features causally derived from the concept are not predictive enough, the probing classifier for the concept can be expected to rely on correlated features to obtain a higher accuracy. However, we show a stronger result: this behavior holds even when there is no potential accuracy gain and the concept's features are easily learnable. Specifically, even when the concept's causally-related features alone can provide 100% accuracy and are linearly separable with respect to a binary probing task label, the probing classifier may still learn non-zero weights for the correlated main-task relevant features. Based on this result, under some simplifying assumptions, we prove that both post-hoc and adversarial training methods can fail to remove the undesired concept, remove useful task-relevant features in addition to the undesired concept, or do

## Amit Sharma

Microsoft Research amshar@microsoft.com

both. As a severe failure mode, we show that post-hoc removal methods can lead to a random-guess main-task classifier by removing all task-relevant information from the representation.

Empirical results on four datasets-natural language inference, sentiment analysis, tweet-mention detection, and a synthetic task-confirm our claims. Across all datasets, as the correlation between the main task and the concept increases, post-hoc removal using null space projection removes a higher amount of the main-task features, eventually leading to a random-guess classifier. In particular, for a pre-trained classifier that does not use the concept at all, the method modifies the representation to yield a classifier that either uses the concept or has lower main-task accuracy, irrespective of the correlation between the main task and the concept. Similarly, for the adversarial removal method, we find that it does not remove all concept-related features. For most datasets, the concept features left within an classifier's representation are comparable to that for a standard main-task classifier.

Our theoretical analysis complements past empirical critiques of adversarial methods for concept removal [12]. More generally, we extend the literature on probing classifiers and their unreliability [4]. Adding to known limitations of explainability methods [19, 35] based on the accuracy of a probing classifier, our results show that recent causallyinspired methods like amnesic probing [13] are also flawed because they depend on access to a good quality concept classifier. Our contributions include:

- Theoretical analysis of null space and adversarial removal methods showing that they fail to remove an undesirable concept from a model's representation, even under favorable conditions.
- Empirical results on four datasets showing that the methods are unable to remove a spurious concept's features fully and end up unnecessarily removing task-relevant features.
- A practical spuriousness score for evaluating the output of concept removal methods.

## 2 Concept removal: Background and problem statement

For a classification task, let ( x i , y i m ) n i =1 be set of examples in the dataset D m , where x i ∈ X are the input features and y i m ∈ Y m the label. We call this the main task and label y i m the main task label. The main task classifier can be written as c m ( h ( x )) where h : X → Z is an encoder mapping the input x to a latent representation z := h ( x ) and c m : Z → Y m is the classifier on top of the representation Z . Additionally, we are given labels for a spurious or sensitive concept, y p ∈ Y p , i.e., ( x i , y i p ) n ′ i =1 in a dataset D p , and our goal is to ensure that the representation h ( x ) learnt by the main classifier does not include features causally derived from the concept. Below we define what it means to be 'causally derived': the representation should not change under an intervention on concept.

Definition 2.1. ( Concept-causal feature ) A feature Z j ∈ Z (jth dimension of h ( x ) ) at the representation layer is defined to be causally derived from a concept (concept-causal for short) if upon changing the value of the concept, the corresponding change in the input's value x will lead to a change in the feature's ( Z j ) value.

For simplicity, we assume that the non-concept-causal features are the main task features. Often, the main task and the concept label are correlated; hence the learnt representation h ( x ) for the main task may include concept-causal features too. A concept removal algorithm is said to be successful if it produces a clean representation h ′ ( x ) to be used by the main-classifier that has no concept-causal features and it does not corrupt or removes the main-task features. If the representation does not contain such features, the main classifier cannot use them [12]. In practice, it is okay if the concept-causal features are not completely removed, but our key criterion is that the removal process should not remove the correlated main task features.

Existing concept removal methods. When the text input can be changed based on changing the value of concept label, methods like data augmentation [23, 50, 41] have been proposed for concept removal. However, for most sensitive or spurious concepts, it is not possible to know the correct change to apply at the input level corresponding to a change in the concept's value.

Instead, methods based on the representation layer have been proposed. To determine whether features in a representation are causally derived by the concept, these methods train an auxiliary, probing classifier c p : Z → Y p where y p ∈ Y p is the label of the concept we want to remove from the latent space z ∈ Z . The accuracy of the classifier indicates the predictive information about the concept embedded in the representation. This probing classifier is then used to remove the sensitive concept from the latent representation which will ensure that the main-task classifier cannot use them. Two kinds of feature removal methods have been proposed: 1) post-hoc methods such as null space removal [32, 13, 24, 20], with removal after the main-task classifier is trained; 2) adversarial methods that jointly train the main task with the probing classifier as the adversary [15, 48, 33, 34].

For adversarial removal, recent empirical results cast doubt on the method's capability to fully remove the sensitive concept from the model's representation [12]. We extend those results with a rigorous theoretical analysis and provide experiments for both adversarial and post-hoc removal methods.

## 3 Attribute removal using probing classifier can be counter-productive

As mentioned above, both removal methods internally use a probing classifier as a proxy for the concept's features. In §3.1, we start off by showing that for any classification task be it probing or main-task classification, it is difficult to learn a clean classifier which doesn't use any spuriously correlated feature (Lemma 3.1 and Lemma B.1). Hence the key assumption driving the use of predictive classifiers within both removal methods is incorrect. Next in §3.2 and 3.3, we will show how these individual components' failure leads to the failure of both removal methods. Finally, in §3.4, we propose a practical spuriousness score to assess the output classifier from any of the removal methods. Throughout this section, we assume that both the main task label y m and probing task label y p are binary ( ∈ {-1 , 1 } ) and there is a basic, fixed encoder h converting the text input to features in the representation space (e.g., a pre-trained model like BERT [10]).

## 3.1 Fundamental limits to learning a clean classifier: Probing and Main Classifier

Given z = h ( x ) and the concept label y p , the goal of the probing task is to learn a classifier c p ( z ) such that it only uses the concept-causal features and the accuracy for y p is maximized. We assume that the main task and concept labels are correlated, so it can be beneficial to use main-task features to maximize accuracy for y p . As argued in the probing literature [19, 4], if there are features in z outside concept-causal that help improve the accuracy of the classifier, a classifier trained on standard losses such as cross-entropy or max-margin is expected to use those features too. Below we show a stronger result: even when there is no accuracy benefit of using non concept-causal features, we find that a probing classifier may still use those features.

Creating a favorable setup for the probing classifier. Specifically, we create a setting that is the most favorable for a probing classifier to use only concept-causal features: 1) no accuracy gain on using features outside of concept-causal because concept-causal features are linearly separable for concept labels , and 2) disentangled representation so that no further representation learning is required. Yet we find that a trained probing classifier would use non-concept-causal features.

Assumption 3.1 (Disentangled Latent Representation) . The latent representation z is disentangled and is of form [ z m , z p ] , where z p ∈ R d p are the concept-causal features and z m ∈ R d m are the main task features. Here d m and d p are the dimensions of z m and z p respectively.

Assumption 3.2 (Concept-causal Feature Linear Separability) . The concept-causal features ( z p ) of the latent representation ( z ) are linearly separable/fully predictive for the concept labels y p , i.e., y i p · (ˆ ϵ p · z i p + b p ) &gt; 0 , ∀ ( x i , y i p ) in training dataset D p for some ˆ ϵ p ∈ R d p and b p ∈ R .

The effect of spurious correlation between concept and label. Now we are ready to state the key lemma which will show that if there is a spurious correlation between the main task and concept labels such that the main-task features z m are predictive of the concept label for only a few special points, then the probing classifier c p ( z ) will use those features. We operationalize spurious correlation as,

Assumption 3.3 (Spurious Correlation) . For a subset of training points S ⊂ D p in the training dataset for a probing classifier, z m is linearly-separable with respect to concept label y p , i.e., y i p · (ˆ ϵ m · z i m + b m ) &gt; 0 ∀ i ∈ S , where ˆ ϵ m ∈ R d m and b m ∈ R .

For simplicity, we assume that the encoder h ( · ) which maps the input X to latent representation Z is frozen or non-trainable. Following [29], we assume max-margin as training loss; under some mild conditions on separable data, a classifier trained using logistic/exponential loss converges to max-margin classifier given infinite training time [43, 21].

̸

Lemma 3.1. Let the latent representation be frozen and disentangled such that z = [ z m , z p ] (Assm 3.1), and conceptcausal features z p are fully predictive for the concept label y p (Assm 3.2). Let c ∗ p ( z ) = w p · z p where w p ∈ R d p be the desired clean linear classifier trained using the max-margin objective (§B.1) that only uses z p for its prediction. Let z m be the main task features, spuriously correlated s.t. z m are linearly-separable w.r.t. probing task label y p for the margin points of c ∗ p ( z ) (Assm 3.3). Then, assuming a zero-centered latent space ( b p = 0 ), a concept-probing classifier c p trained using the max-margin objective will use spurious features, i.e., c p ( z ) = w p · z p + w m · z m where w m = 0 and w m ∈ R d m .

<!-- image -->

- (a) Clean pretrained main task ( Profession ) classifier.

(b) Probing (

Gender

) classifier with a slanted projection direction.

- (c) Main classifier becomes unfair after null space projection.

Figure 1: Failure mode of null space removal. Consider a main task ( Profession ) classifier where Gender is the spurious concept to be removed. Assume a 2-dimensional latent representation z , where one dimension corresponds to profession and the other to the gender feature. (a) A 'clean' (fair) main task classifier that only uses the Profession feature, shown by its vertical projection direction, that is input to INLP for concept removal. Its decision boundary is orthogonal to the projection direction. (b) From Lemma 3.1, INLP trains a probing classifier for gender with a slanted projection direction (ideal gender projection direction would be horizontal). (c) For two points having the same profession but different gender features (marked '1' ), projection to the null-space ( '2' ) has their profession feature reversed ( '3' ), thus making the fair pretrained classifier become unfair (also see §3.2).

Proof Sketch. Starting from c ∗ p ( z ) , we show that there always exists a perturbed classifier which uses the main task features and has a bigger margin than c ∗ p ( z ) . Within some range of perturbation, for all margin points of c ∗ p , using the main task features increases the margin by Assm 3.3, and does not reduce the margin for non-margin points s.t. it becomes the same as the margin of c ∗ p . Proof in §B.3.

Our result shows that not just accuracy, even geometric skews in the dataset can yield an incorrect probing classifier. In §B.4 we prove that the assumptions for Lemma 3.1 are both sufficient and necessary for a classifier to use nonconcept-features z m when z p is 1-dimensional. Lemma 3.1 generalizes a result from [29] by using fewer assumptions (we do not restrict z m to be binary, do not assume that z m and z p are conditionally independent given y , and do not assume monotonicity of classifier norm with dataset size). We present a similar result for the main task classifier: under spurious correlation of concept and main task labels, the main task classifier would use concept-causal features even when 100% accuracy can be achieved using only main task features (Lemma B.1, §B.2).

## 3.2 Failure mode of post-hoc removal methods: Null-space removal (INLP)

The null space method [32, 13], henceforth referred as INLP , removes a concept from latent space by projecting the latent space to a subspace that is not discriminative of that concept. First, it estimates the subspace in the latent space discriminative of the concept we want to remove by training a probing classifier c p : Z → Y p , where Y p is the concept label. Then the projection is done onto the null-space of this probing classifier which is expected to be non-discriminative of the concept. For instance, [32] use a linear probing classifier c p ( z ) to ensure that the any linear classifier cannot recover the removed concept from modified latent representation z ′ and hence the main task classifier ( c m ( z ′ ) ) becomes invariant to removed concept. Also, they recommend running this removal step for multiple iterations to ensure the unwanted concept is removed completely (details are in §C.1). Below we state the failure of the null-space method using z i ( k ) to denote the representation z i after k steps of INLP.

̸

Theorem 3.2. Let c m ( z ) be a pre-trained main-task classifier where the latent representation z = [ z m , z p ] satisfies Assm 3.1 and 3.2. Let c p ( z ) be the probing classifier used by INLP to remove the unwanted features z p from the latent representation. Under Assm 3.3, Lemma 3.1 is satisfied for the probing classifier c p ( z ) such that c p ( z ) = w p · z p + w m · z m and w m = 0 . Then,

1. Damage in the first step of INLP. The first step of linear-INLP will corrupt the main-task features and this corruption is non-invertible with subsequent projection steps of INLP.

̸

- (a) Mixing: If w p = 0 , the main task z m and concept-causal features z p will get mixed such that z i (1) = [ g ( z i m , z i p ) , f ( z i p , z i m )] = [ g ( z i m ) , f ( z i p )] for some function ' f ' and ' g '. Thus, the latent representation is no longer disentangled and removal of concept-causal features will also lead to removal of main task features.

̸

Figure 2: Failure mode of adversarial removal. As in Fig. 1, the main task label is Profession and Gender is the spurious concept, each corresponding to one of the dimensions of the 2-dimensional feature representation z . Assume that the shared representation is a scalar value obtained by projecting the two features in some direction. The adversarial goal is to find a projection direction such that the concept (gender) classifier obtains a random-guess accuracy of 50% but has good accuracy on the main task label (profession). (a) Two projection directions, shown by vertical and slanted lines, that yield random-guess 50% accuracy on gender prediction, and (b) have the same 100% accuracy for profession prediction. (c) However, the slanted projection direction has a bigger margin for the main task and will be preferred, thus leading to a final classifier that uses the gender concept (see §3.3).

<!-- image -->

- (b) Removal: If w p = 0 , then the first projection step of INLP will do opposite of what is intended, i.e., damage the main task features z m (in case z m ∈ R , it will completely remove z m ) but have no effect on the concept-causal features z p .
2. Removal in the long term: The L2-norm of the latent representation z decreases with every projection step as long as the parameters of probing classifier ( w k ) at a step ' k ' does not lie completely in the space spanned by parameters of previous probing classifiers, i.e., span( w 1 , . . . , w k -1 ), z i ( k -1) , z i (0) and z i (0) in direction of w k is not trivially zero. Thus, after sufficiently many steps, INLP can destroy all information in the representation s.t. z i ( ∞ ) = [ 0 , 0 ] .

Proof Sketch. From Lemma 3.1, in the first step, probing classifier for z p will use z m in addition to z p . Consequently, the projection matrix for INLP based on the probing classifier will be incorrect, hence corrupting the main task features z m with z p (1a) or damage z m without any effect on z p (1b) . Next, we show that each step of the projection operation reduces the norm of latent representation z ; thus the latent representation can go to 0 as the number of steps increases (2) . Proof in §C.

Failure Mode: Fig. 1a-1c demonstrate the mixing problem stated in Theorem 3.2, where a fair classifier becomes unfair after the first step of projection. Note that after first step the main task classifier's accuracy will drop because of this mixing of features, affecting INLP-based probing methods like Amnesic Probing [13] that interpret a drop in the main classifier's accuracy after INLP projection as evidence that the main classifier was using the sensitive concept.

## 3.3 Failure mode of adversarial removal methods

To remove the unwanted features z p from the latent representation, adversarial removal methods jointly train the main classifier c m : Z → Y m and the probing classifier c p : Z → Y p by specifying c p 's loss as an adversarial loss. For details refer to §D.1.

As in Lemma 3.1, we assume that the encoder h : X → Z mapping the input to the latent representation Z is frozen. To allow for the removal of the unwanted features z p , we introduce additional representation layers after it. For simplicity in the proof, we assume a linear transformation to the latent representation h 2 : Z → ζ . This layer is followed by the linear main-task classifier c m : ζ → Y m , as before. The probing classifier c p : ζ → Y p is trained adversarially to remove z p from the latent representation ζ . Thus, the goal of the adversarial method can be stated as removing the information of z p from ζ . Let the main-task classifier satisfy assumptions of the generalized version of Lemma 3.1 (Lemma B.1, §B.2). We also need an additional assumptions on the hard-to-classify margin points to ensure that main-task labels and concept labels are correlated on the margin points of a clean main-task classifier. Proof of the Theorem 3.3 stated below is in §D.

Assumption 3.4 (Label Correlation on Margin Points) . For the margin points of a clean classifier for the main task, the adversarial-probing labels y p and the main task labels y m are correlated, i.e., w.l.o.g., y i m = y i p for all margins points of the clean main task classifier.

Theorem 3.3. Let the latent representation z satisfy Assm 3.1 and be frozen, h 2 ( z ) be a linear transformation over Z s.t. h 2 : Z → ζ , the main-task classifier be c m ( ζ ) = w c m · ζ , and the adversarial probing classifier be c p ( ζ ) = w c p · ζ . Let all the assumptions of Lemma B.1 be satisfied for main-classifier c m ( · ) when using z directly as input and Assm 3.2 be satisfied on z w.r.t. the adversarial task. Let h ∗ 2 ( z ) be the desired encoder which is successful in removing z p from ζ . Then there exists an undesired/incorrect encoder h α 2 ( z ) s.t. h α 2 ( z ) is dependent on z p and the main-task classifier c m ( h α 2 ( z )) has bigger margin than c m ( h ∗ 2 ( z )) and has,

1. Accuracy ( c p ( h α 2 ( z )) , y p ) = Accuracy ( c p ( h ∗ 2 ( z )) , y p ) ; when adversarial probing classifier c p ( · ) is trained using any learning objective like max-margin or cross-entropy loss. Thus, the undesired encoder h α 2 ( z ) is indistinguishable from desired encoder h ∗ 2 ( z ) in terms of adversarial task prediction accuracy but better for main-task in terms of max-margin objective.
2. L h 2 ( c m ( h α 2 ( z )) , c p ( h α 2 ( z )) ) &lt; L h 2 ( c m ( h ∗ 2 ( z )) , c p ( h ∗ 2 ( z )) ) ; when Assm 3.4 is satisfied and concept-causal features z p M of any margin point z M of c m ( h ∗ 2 ( z )) are more predictive of the main task label than z P p of any margin point z P of c p ( h ∗ 2 ( z )) is predictive for the probing label (Assm D.1). Thus, undesired encoder h α 2 ( z ) is preferable over desired encoder h ∗ 2 ( z ) for both main and combined adversarial objective. Here L h 2 = L ( c m ( · )) -L ( c p ( · )) is the combined adversarial loss w.r.t. to h 2 and L ( c ( · )) is the max-margin loss for a classifier 'c' (see §D.1).

Proof Sketch. (1) The proof is by construction. Using Lemma B.1, we show that there exists h α 2 s.t. L ( c m ( h α 2 )) &lt; L ( c m ( h ∗ 2 )) , and that accuracy of the probing classifier remains the same when using either encoder. (2) Compared to h ∗ 2 , we show that the improvement in main task loss when using z p features is larger than the improvement in the probing loss for h α 2 , thus preferred by overall objective.

## 3.4 Implications for real-world data: A metric for quantifying degree of spuriousness

Our theoretical analysis shows that probing-based removal methods fail to make the main task classifier invariant to unwanted concepts. However, to verify whether the final classifier is using the concept or not, the theorem statements require knowledge of the concept's features z p . For practical usage, we propose a metric that quantifies the degree of failure or spuriousness for both the main and probing classifier. For simplicity, we define it assuming that both main and concept labels are binary.

Let D m,p be the dataset where for every input x i we have both the main task label y m and the concept label y p . We define 2 × 2 groups, one for each combination of ( y m , y p ) . Without loss of generality, assume that the main-task label y m = 1 is spuriously correlated with concept label y p = 1 and similarly y m = 0 is correlated with y p = 0 . Thus, ( y m = 1 , y p = 1) and ( y m = 0 , y p = 0) are the majority group S maj while groups ( y m = 1 , y p = 0) and ( y m = 0 , y p = 1) make up the minority group S min . We expect the main classifier to exploit this correlation and hence perform badly on S min where the correlation breaks. Following [39], we posit that minority group accuracy i.e Acc ( S min ) can be a good metric to evaluate the degree of spuriousness . We bound the metric by comparing it with the accuracy on S min of a 'clean' classifier that does not use the concept features.

Definition 3.1 (Spuriousness Score) . Given a dataset, D m,p = S min ∪ S maj with binary task label and binary concept, let Acc f ( S min ) be the minority group accuracy of a given main task classifier ( f ) and Acc ∗ ( S min ) be the minority group accuracy of a clean main task classifier that does not use the spurious concept. Then spuriousness score of f is: ψ ( f ) = | 1 -Acc f ( S min ) /Acc ∗ ( S min ) | .

To estimate Acc ∗ ( S min ) , we subsample the dataset such that y p takes a single value in the sample and train the main classifier on it, as in [35]. Here the probing label y p no longer is correlated with the main task label y m . The spuriousness score of a probing classifier can be defined analogously to Def 3.1, by swapping the task and concept label (see Def E.1). For creating a clean probing classifier, we subsample the dataset such that y m takes a single value and train the probing classifier.

## 4 Experimental Results

Theorems 3.2 and 3.3 show the failure of concept removal methods under a simplified setup and max-margin loss. But current deep-learning models are not trained using max-margin objective and might not satisfy the required assumptions (Assm 3.1,3.2,3.3,3.4). Thus, we now verify the failure modes on three real-world datasets and one synthetic dataset, without making any restrictive assumptions. We use RoBERTa [25] as default encoder and fine-tune it over each real-world dataset. For Synthetic-Text dataset we use the sum of pre-trained GloVe embeddings [30] of words in a sentence as the default encoder. For details on the experimental setup, refer §E.

## 4.1 Datasets: Main task and spurious/sensitive concept

Real-world data. We use three datasets: MultiNLI [46], Twitter-PAN16 [31] and Twitter-AAE [6]. In MultiNLI, given two sentences-premise and hypothesis-the main task is to predict whether hypothesis entails , contradicts or is neutral with respect to premise. We simplify to a binary task of predicting whether a hypothesis contradicts the premise or not. Since negation words like nobody,no,never and nothing have been reported to be spuriously correlated with the contradiction label [16], we create a 'negation' concept denoting the presence of these words. The goal is to remove the negation concept from an NLI model's representation space. In Twitter-PAN16, the main task is to detect whether a tweet mentions another user or not, as in [12]. The dataset contains gender label for each tweet, which we consider as the sensitive concept to be removed from the main model's representation. In Twitter-AAE, again following [12], the main-task is to predict binary sentiment labels from a tweet's text. The tweets are associated with race of the author, the sensitive concept to be removed from the main model's representation.

Synthetic-Text. To understand the reasons for failure, we introduce a Synthetic-Text dataset where it is possible to change the input text based on a change in concept (thus implementing Def. 2.1). Here we can directly evaluate whether the concept is being used by the main-task classifier by intervening on the concept (adding or removing) and observing the change in model's prediction. The main-task is to predict whether a sentence contains a numbered word (e.g., one, fifteen, etc. ). We introduce a spurious concept (length) by increasing the length of sentences that contain numbered words.

Predictive correlation. To assess robustness of removal methods, we create multiple datasets with different predictive correlation between the two labels y m and y p . The predictive correlation ( κ ) is a practical measure for the spurious correlation defined in Assm 3.3, that does not require access to z m features. It measures how informative one label is for predicting the other, κ = Pr ( y m · y p &gt; 0) = ∑ N i =1 1 [ y m · y p &gt; 0] N , where N is the size of dataset and 1 [ · ] is the indicator function that is 1 if the argument is true otherwise 0 . Predictive-correlation lies in κ ∈ [0 . 5 , 1] where κ = 0 . 5 indicates no correlation and κ = 1 indicates that the attributes are fully correlated. For more details on the datasets and how we vary the predictive-correlation, refer to §E; and for additional results see §F.

Measuring spuriousness of a classifier. We use the Spuriousness Score (Def 3.1) to measure the degree of reliance of the main task classifier on the spurious concept, and vice-versa for the probing classifier. In addition, for the Synthetic-Text dataset, we use a metric ∆ Prob that exactly implements Def 2.1 for estimating a model's reliance on the spurious concept. Since we can modify the concept directly in input space for Synthetic-Text, ∆ Prob changes the parts of a input sentence corresponding to spurious concept and measures the change in the main task classifier's prediction probability. As a sanity check, on the Synthetic-Text dataset, ∆ Prob and Spuriousness Score are highly correlated (Pearson correlation=0.83, §G). For all real-world datasets, we use the Spuriousness Score.

## 4.2 Results: Null space removal

In general, for any model given as input to INLP, it may be difficult to verify whether INLP removed the correct features. Hence, we construct a benchmark where the input classifier is clean , i.e., it does not use the concept at all. We do so by training on a subset of data with one particular value of spurious concept label, as in [35]. Since the input classifier does not use the concept-causal features, we expect that INLP should not have any effect on the main task classifier. Note that we keep the main task classifier frozen in all the experiments described below. For the setting where the main task classifier is retrained after every projection step of INLP, refer §F.2 and Fig. 9.

Eventually all task-relevant features are destroyed. We start with the Synthetic-Text dataset by training a clean classifier on the main-task and inputting it to INLP for removing the spurious concept. To keep the conditions favorable for INLP, both the main task and concept-probing task can achieve 100% accuracy using their causally derived features respectively. In Fig. 3a, colored lines show datasets with different levels of predictive correlation κ that are provided to INLP and iterations 21-40 show individual steps of null-space removal. Since, the given pre-trained classifier was clean , i.e., not using the concept features, null-space removal shouldn't have any effect on it. We observe that for all values of κ , the main-task classifier's accuracy eventually goes to 50% random guess accuracy implying that the main-task related attribute has been removed by INLP, as predicted by Theorem 3.2. Higher the value of correlation κ , faster the removal of main-task attribute happens. We obtain a similar pattern over real-world datasets. Fig. 3d,3e and 3f show a decrease in the main-task accuracy even when the input classifier for each dataset is ensured to be clean : except for κ = 0 . 5 (no correlation), all values of κ yield a random-guess classifier after applying INLP on MultiNLI while they yield classifiers with less than 60% accuracy for Twitter-PAN16 and Twitter-AAE.

Early stopping increases the reliance on spurious features. To avoid full collapse of the main-task features, a stopping criterion in INLP is to stop when the main-task classifier's performance drops [32]. In Fig. 3b we measure spuriousness, sensitivity of the Synthetic-Text main task classifier w.r.t. to the spurious concept, using ∆ Prob (see §4.1 and E.8). At lower iterations of INLP, ∆ Prob is higher than that of the input classifier. For example, for κ = 0 . 8 , when the main-task classifier's performance drops at iteration 27, the classifier has a high ∆ Prob ≈ 25% , higher than the

Figure 3: Null space removal failure. Top row corresponds to the Synthetic-Text dataset and bottom row shows the failure on three real-world datasets. In each figure, the x-axis shows the INLP iteration and y-axis shows different evaluation metrics. Colored lines correspond to the different levels of predictive correlation ( κ ) in the datasets used by INLP. (a), (d), (e), (f) show that as INLP removal progresses, main-task classifier is getting corrupted which leads to drop in its accuracy (see §4.2).

<!-- image -->

input classifier (Fig. 3b). Hence it is possible that stopping prematurely will lead us to a classifier that is more reliant on the spurious concept than it was before, consistent with the statement 1(a) in Theorem 3.2. The reason is the mixing of the main task and concept-causal features in each iteration, as shown in Fig. 3c using the spuriousness score of the probing classifier (Def 3.1). At lower iterations, the spuriousness score of probing classifier increases to a very high value (close to max value 1), for all values of κ .

Failure of causally-inspired probing. Amnesic Probing [13] declares that a sensitive concept is being used by the model if, after removal of the concept from from the latent representation using INLP, there is a drop in the main-task performance. But Fig. 3a, 3d, 3e and 3f show that even when the input classifier for its corresponding main task is clean, i.e., does not use the sensitive concept, INLP leads to drop in performance of the main-classifier. Hence, removal-based methods like Amnesic probing will falsely conclude that the sensitive concept is being used.

## 4.3 Results: Adversarial removal

We now demonstrate failure of the adversarial-removal method (AR) in removing the spurious concept from the main classifier. We train a separate main-task classifier without any adversarial objective with standard cross-entropy loss objective (referred as ERM ). Then we compare standard ERM training of the main classifier with the AR method over the same number of epochs (20). We follow the training procedure of [12] and conduct a hyper-parameter sweep on the adversarial training strength to select the value which is most effective in removing the concept. For details, refer to §E. Cannot remove the spurious concept fully. For MultiNLI, Fig. 4a shows the spuriousness score (Def 3.1) of ERM and AR classifiers as we vary the predictive correlation ( κ ) between the main-task label and sensitive concept label in the training dataset. While the spuriousness score for classifier trained using AR (blue curve) is lower than that of ERM for all values of κ , it is substantially away from zero. Thus, the AR method fails to completely remove the spurious concept completely from the latent representation. By inspecting the concept probing classifier accuracy for ERM and AR in Fig. 4b, we obtain a possible explanation. The probe accuracy after adversarial training doesn't decrease to 50% but stops at accuracy proportional to the predictive correlation κ . This is expected since even if the AR would have been successful in removing the concept-causal features, the main-task features would still be predictive of the concept label by κ due to the spurious correlation between them. However, the converse is not true: an accuracy of κ does not imply that the concept is fully removed. The results substantiate the first statement of Theorem 3.3: given two representations where one ( desired ) does not have concept features while the other ( undesired ) contains the concept features, the undesired one may be better for the main task accuracy even as both may have the same probing accuracy. Fig. 4d, 4e and 4f show the spuriousness score of AR in comparison to classifier trained with ERM on Twitter-PAN16, Twitter-AAE and Synthetic-Text datasets respectively. The failure of AR is worse here: there is no significant reduction in spuriousness score for AR in comparison to ERM. For the Synthetic-Text dataset, ERM has zero spuriousness score

Figure 4: Adversarial Removal Failure. Top row explains failure of the AR method on MultiNLI. Bottom row shows the failure on Twitter-PAN16, Twitter-AAE, and Synthetic-Text datasets. In each figure, the x-axis shows different levels of predictive-correlation ( κ ) between the main task and concept labels in the dataset used by AR and the y-axis shows different evaluation metrics. Orange lines denote the ERM model and the blue lines denote the model trained using AR. (b), (d), (e), (f) show that AR is unable to completely remove the spurious concept from the main task classifier. (c) shows a stronger failure where the AR method introduces spuriousness into a clean input classifier.

<!-- image -->

but AR has non-zero score. We expand more on this observation and include additional results on adversarial removal in §F.3.

AR makes a clean classifier use the spurious concept. In Fig. 4c we provide a clean main task classifier (see §F.3 for training details) as input to AR method. For all values of κ , the input classifier's spuriousness score is low (iteration 1-6). From iteration 7 onwards, the AR method corrupts the clean classifier as shown by increasing spuriousness scores. For more results, see Fig. 12b in §F.3.

Comparison with previous work. If post-removal the latent representation used by the main-task classifier is still predictive of the removed concept, [12] claimed it as a failure of the adversarial removal method. However, this claim may not be correct since a feature could be present in the latent space and yet not used by model [35]. Our proposed spuriousness score metric avoids this limitation.

Ablations. In Appendix, we report results on using BERT instead of RoBERTa as the input encoder (§F.2, F.3), the effect of using different modeling choices like loss-function, regularization, e.t.c. (§F.4), and the behavior of probing classifiers when concept is not present in latent space (§F.1).

## 5 Related work

Concept removal methods. When the removal of a concept can be simulated in input space (e.g., in tabular data or simpler concepts), removing a concept directly using data augmentation [23] or gradient regularization [38, 22] can work. However, concept removal is non-trivial when change in a concept cannot be propagated via change in input tokens. Combining the ideas of null space and adversarial removal [32, 24, 48, 12], methods like [33, 34] restrict the adversarial function to be a projection operation and derive a closed form solution. Other approaches use explanations of the classifier's prediction for concept removal [17]. Our work highlights the difficulty of building an estimator for the features causally derived from a concept, as a general limitation for concept removal.

Limitations of a probing classifier for model interpretability. We also contribute to the growing literature on the limitations of a probing classifier's accuracy in capturing whether the main classifier is using a concept [4]. It is known that probing classifiers capture not just the concept but any other features that may be correlated with it [39, 2, 23, 45]. As a result, many improvements have been proposed to better estimate whether a concept is being used, including the use of control labels or datasets [19, 35]. Parallelly, new causality-inspired probing methods [13] compare the main task accuracy on a representation without the concept constructed using the null space removal method. The hope is that such improvements can make probing more robust. Our results question this direction. To demonstrate the fundamental

unreliability of probing classifier, we construct a setup that is most favorable for learning only the concept's features and still find that learned probing classifier includes non-zero weight for other features, limiting effectiveness of any interpretation method based on it.

## 6 Conclusion

Our theoretical and empirical evaluations show that it is difficult to create a probing-based explainability and removal method due to the fundamental limitation of learning a 'clean' probing classifier. We recommend two tests for validating removal methods. First, we provide a sanity-check: any reasonable removal method should not modify a 'clean' classifier that does not use any spurious features to produce a final classifier that uses those features. Second, we propose a spuriousness score that can be used to evaluate the dependence of any classifier on spurious features. As a future step, we encourage the community to develop more such sanity-check tests to evaluate proposed methods.

Alternatively, we point attention to other approaches that may provide better guarantees for concept removal. An example is extending data augmentation techniques like counterfactual data augmentation ([23, 9]) to non-trivial concepts. For a given training point that may include a spurious correlation, a new data point is generated that breaks the correlation but keeps the semantics of the rest of the text identical (hence the name, 'counterfactual'). This can be done by human labeling or handcrafted rules for modifying text (e.g., Checklists [37]). Then the main classifier is regularized to have the same representation for such pairs of inputs ([3, 26]). By construction, with good quality pairs, such a method will not remove task-relevant features and will satisfy the sanity checks listed above. That said, a limitation is that the removal quality will depend on the diversity of the counterfactual examples generated and whether they capture all aspects of the spurious concept. Another direction is to take inspiration from the algorithmic fairness literature [18, 28] and focus on the predictions of the classifier rather than the representation. Compared to removal in latent space, enforcing certain fairness properties on model predictions is a more well-formed task, more interpretable, and definitely more relevant if the final goal is fair decision making.

Limitations. A limitation of our theoretical work is assuming frozen or non-trainable latent representation which makes the analysis of task-classifier trained on top of them relatively easier. We address this limitation in our empirical work where we do not make such assumptions. Also, our work addresses failure modes of two popular methods, null space removal and adversarial removal. We conjecture that any other method based on probing classifiers will lead to similar failure modes.

## References

- [1] Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, and Yoav Goldberg. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings . OpenReview.net, 2017.
- [2] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization, 2020.
- [3] Ananth Balashankar, Xuezhi Wang, Ben Packer, Nithum Thain, Ed Chi, and Alex Beutel. Can we improve model robustness through secondary attribute counterfactuals? In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 4701-4712, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.
- [4] Yonatan Belinkov. Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics , 48(1):207-219, 04 2022.
- [5] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics) . SpringerVerlag, Berlin, Heidelberg, 2006.
- [6] Su Lin Blodgett, Lisa Green, and Brendan O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of African-American English. In Proceedings of EMNLP , 2016.
- [7] Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Proceedings of the 30th International Conference on Neural Information Processing Systems , NIPS'16, page 4356-4364, Red Hook, NY, USA, 2016. Curran Associates Inc.
- [8] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. What you can cram into a single $&amp;!#* vector: Probing sentence embeddings for linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 2126-2136, Melbourne, Australia, July 2018. Association for Computational Linguistics.

- [9] Saloni Dash, Vineeth N Balasubramanian, and Amit Sharma. Evaluating and mitigating bias in image classifiers: A causal perspective using counterfactuals. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 915-924, 2022.
- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- [11] P. Diehl, H. Kellerhals, and E. Lustig. Diagonalization of symmetric matrices. In Computer Assistance in the Analysis of High-Resolution NMR Spectra , pages 73-77, Berlin, Heidelberg, 1972. Springer Berlin Heidelberg.
- [12] Yanai Elazar and Yoav Goldberg. Adversarial removal of demographic attributes from text data. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing , pages 11-21, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.
- [13] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Trans. Assoc. Comput. Linguistics , 9:160-175, 2021.
- [14] Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Understanding undesirable word embedding associations. In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers , pages 1696-1705. Association for Computational Linguistics, 2019.
- [15] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37 , ICML'15, page 1180-1189. JMLR.org, 2015.
- [16] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers) , pages 107-112, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.
- [17] Xiaochuang Han and Yulia Tsvetkov. Influence tuning: Demoting spurious correlations via instance attribution and instance-driven updates. In Findings of the Association for Computational Linguistics: EMNLP 2021 , pages 4398-4409, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.
- [18] Moritz Hardt, Eric Price, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 29. Curran Associates, Inc., 2016.
- [19] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 2733-2743, Hong Kong, China, November 2019. Association for Computational Linguistics.
- [20] Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, and Yoav Goldberg. Contrastive explanations for model interpretability. In EMNLP (1) , pages 1597-1611, 2021.
- [21] Ziwei Ji and Matus Telgarsky. The implicit bias of gradient descent on nonseparable data. In Alina Beygelzimer and Daniel Hsu, editors, Proceedings of the Thirty-Second Conference on Learning Theory , volume 99 of Proceedings of Machine Learning Research , pages 1772-1798. PMLR, 25-28 Jun 2019.
- [22] Sai Srinivas Kancheti, Abbavaram Gowtham Reddy, Vineeth N Balasubramanian, and Amit Sharma. Matching learned causal effects of neural networks with domain priors. In International Conference on Machine Learning , pages 10676-10696. PMLR, 2022.
- [23] Divyansh Kaushik, Amrith Setlur, Eduard Hovy, and Zachary C Lipton. Explaining the efficacy of counterfactually augmented data. International Conference on Learning Representations (ICLR) , 2021.
- [24] Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. Towards understanding and mitigating social biases in language models. In ICML , pages 6565-6576, 2021.
- [25] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. CoRR , abs/1907.11692, 2019.
- [26] Divyat Mahajan, Shruti Tople, and Amit Sharma. Domain generalization using causal matching. In International Conference on Machine Learning , pages 7313-7324. PMLR, 2021.

- [27] Tom McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 3428-3448, Florence, Italy, July 2019. Association for Computational Linguistics.
- [28] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM Comput. Surv. , 54(6), jul 2021.
- [29] Vaishnavh Nagarajan, Anders Andreassen, and Behnam Neyshabur. Understanding the failure modes of out-ofdistribution generalization. In International Conference on Learning Representations , 2021.
- [30] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP) , pages 1532-1543, 2014.
- [31] Francisco Rangel, Paolo Rosso, Ben Verhoeven, Walter Daelemans, Martin Potthast, and Benno Stein. Pan16 author profiling, https://doi.org/10.5281/zenodo.3745963. In CLEF 2016 Labs and Workshops, Notebook Papers. Zenodo, September 2016.
- [32] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 7237-7256, Online, July 2020. Association for Computational Linguistics.
- [33] Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan D Cotterell. Linear adversarial concept erasure. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 18400-18421. PMLR, 17-23 Jul 2022.
- [34] Shauli Ravfogel, Francisco Vargas, Yoav Goldberg, and Ryan Cotterell. Adversarial concept erasure in kernel space. CoRR , abs/2201.12191, 2022.
- [35] Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. Probing the probing paradigm: Does probing accuracy entail task relevance? In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume , pages 3363-3377, Online, April 2021. Association for Computational Linguistics.
- [36] Radim Rehurek and Petr Sojka. Gensim-python framework for vector space modelling. NLP Centre, Faculty of Informatics, Masaryk University, Brno, Czech Republic , 3(2), 2011.
- [37] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 4902-4912, Online, July 2020. Association for Computational Linguistics.
- [38] Andrew Slavin Ross, Michael C. Hughes, and Finale Doshi-Velez. Right for the right reasons: Training differentiable models by constraining their explanations. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17 , pages 2662-2670, 2017.
- [39] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations (ICLR) , 2020.
- [40] Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh, and Percy Liang. An investigation of why overparameterization exacerbates spurious correlations. In Proceedings of the 37th International Conference on Machine Learning , ICML'20. JMLR.org, 2020.
- [41] Indira Sen, Mattia Samory, Claudia Wagner, and Isabelle Augenstein. Counterfactually augmented data and unintended bias: The case of sexism and hate speech detection. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 4716-4726, Seattle, United States, July 2022. Association for Computational Linguistics.
- [42] Xing Shi, Inkit Padhi, and Kevin Knight. Does string-based neural MT learn source syntax? In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing , pages 1526-1534, Austin, Texas, November 2016. Association for Computational Linguistics.
- [43] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. J. Mach. Learn. Res. , 19(1):2822-2878, jan 2018.
- [44] Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, and Aleksander Madry. Robustness may be at odds with accuracy. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019 . OpenReview.net, 2019.

- [45] Victor Veitch, Alexander D'Amour, Steve Yadlowsky, and Jacob Eisenstein. Counterfactual invariance to spurious correlations in text classification. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 16196-16208, 2021.
- [46] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) , pages 1112-1122. Association for Computational Linguistics, 2018.
- [47] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface's transformers: State-of-the-art natural language processing, 2019.
- [48] Qizhe Xie, Zihang Dai, Yulun Du, Eduard Hovy, and Graham Neubig. Controllable invariance through adversarial feature learning. In Proceedings of the 31st International Conference on Neural Information Processing Systems , NIPS'17, page 585-596, Red Hook, NY, USA, 2017. Curran Associates Inc.
- [49] Ke Xu, Tongyi Cao, Swair Shah, Crystal Maung, and Haim Schweitzer. Cleaning the null space: A privacy mechanism for predictors. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence , AAAI'17, page 2789-2795. AAAI Press, 2017.
- [50] Ran Zmigrod, Sabrina J. Mielke, Hanna Wallach, and Ryan Cotterell. Counterfactual data augmentation for mitigating gender stereotypes in languages with rich morphology. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 1651-1661, Florence, Italy, July 2019. Association for Computational Linguistics.

## A Broader Impact and Ethical Consideration

Removal of spurious or sensitive concepts is an important problem to ensure that machine learning classifiers generalize better to new data and are fair towards all groups. We found multiple limitations with current removal methods and recommend caution against the use of these methods in practice.

## B Probing and Main Classifier Failure Proofs

## B.1 Notation and Setup: Max-margin Classifier

We assume that encoder h : X → Z , mapping the input to latent representation is frozen/non-trainable. Thus for every input x i in the dataset D , we have a corresponding latent representation z i which is fixed. Also, the latent representation Z is disentangled i.e z = [ z m , z p ] where z m are the main task features, i.e., causally derived from the main task label and z p are the concept-causal features, causally derived from the concept label. Let c p ( z ) = w p · z p + w m · z m be the linear probing classifier which we train using max-margin objective. The hyperplane c p ( z ) = 0 is the decision boundary of this linear classifier. The points which fall on one side of the decision boundary ( c p ( z ) &gt; 0 ) are assigned one label (say positive label 1) and the rest are assigned another label (say negative label -1). The margin M c p of this probing classifier ( c p ( z )) is the distance of the closest latent representation ( z ) from the decision boundary. The points which are closest to the decision boundary are called the margin points . The distance of a given latent representation z i having class label y i , where y i ∈ {-1 , 1 } , from the decision boundary is given by

$$\mathcal { M } _ { c _ { p } } ( z ^ { i } ) \colon = \frac { m _ { c _ { p } } ( z ^ { i } ) } { \| w \| } = \frac { y _ { p } ^ { i } \cdot c _ { p } ( z ^ { i } ) } { \| w \| } = \frac { y _ { p } ^ { i } \cdot ( w \cdot z ^ { i } + b ) } { \| w \| }$$

where ∥ w ∥ is the L2 norm of parameters w = [ w p , w m ] of the probing classifier c p ( z ) .

Max-Margin (MM): Then the max-margin classifier is trained by optimizing the following objective:

$$\arg \max _ { w , b } \left \{ \min _ { i } \mathcal { M } _ { c _ { p } } ( z ^ { i } ) \right \} & & ( 2 )$$

For ease of exposition we convert this objective into multiple equivalent forms. To do this we observe that scaling the parameters of c p ( z ) by a positive scalar γ i.e w → γ w and b → γb does not change the distance of the point ( M c p ( z i ) ) from the decision boundary.

MM-Denominator Version: We can use this freedom of scaling the parameters to set m c p ( z i ) = 1 for the closest point of any given probing classifier c p ( z ) , thus all the data points will satisfy the constraint,

$$m _ { c _ { p } } ( z ^ { i } ) = y ^ { i } \cdot c _ { p } ( z ^ { i } ) \geq 1$$

$$\arg \max _ { w } \left \{ \frac { 1 } { \| w \| } \right \}$$

under the constraint m c p ( z i ) ≥ 1 corresponding to all the points in the dataset.

MM-Numerator Version: Alternatively, one can choose γ such that ∥ w ∥ = c where c ∈ R is some constant value. The the modified objective becomes:

$$\arg \max _ { w , b } \left \{ m _ { c _ { p } } ( z ^ { i } ) \right \}$$

under constraint ∥ w ∥ = c which is usually set to 1 .

We will use one of these formulations in our proofs based on the ease of exposition and give a clear indication when we do so. One can refer to Chapter 7 , Section 7 . 1 of [5] for further details about max-margin classifiers and different formulations of the max-margin objective.

## B.2 Problem with learning a clean main-task classifier

In this section, we will restate the assumptions and results of Lemma 3.1 for the main-task classifier (instead of the probing classifier) and show that the same results will hold.

giving us the final max-margin objective:

Assm 3.1 remains the same since it is made on the latent-representation being disentangled and frozen/non-trainable. Next, parallel to Assm 3.2, we show that even when main-task feature is 100% predictive of main-task and is linearly separable, the trained main-task classifier will also use the concept-causal features. Formally,

Assumption B.1 (main-task feature Linear Separability) . The main-task features ( z m ) of the latent representation ( z ) for every point are linearly separable/fully predictive for the main-task labels y m , i.e y i m · (ˆ ϵ m · z i m + b m ) &gt; 0 for all datapoints ( x i , y i m ) for some ˆ ϵ m ∈ R d m and b m ∈ R . For the case of zero-centered latent space, we have b m = 0 .

Next similar to Assm 3.3, we define the spurious correlation between main-task and concept label: a function using only z p may also be able to classify correctly on some non-empty subset of points w.r.t. main-task label ( y m ).

Assumption B.2 (Main-Task Spurious Correlation) . For a subset of training points S ⊂ D m , main-task label y m is linearly separable using z p i.e y i m · (ˆ ϵ p · z i p + b p ) &gt; 0 for some ˆ ϵ p ∈ R d p and b p ∈ R . For the case of zero-centered latent space we have b p = 0 .

Next we rephrase Lemma 3.1 which shows that for only a few special points if the concept-causal features z p are linearly-separable w.r.t. to main task classifier y m (Assm B.2), then the main-task classifier c m ( z ) will use those features.

̸

Lemma B.1 (Sufficient Condition for Main-task Classifier) . Let the latent representation be frozen and disentangled such that z = [ z m , z p ] (Assm 3.1), where main-task-features z m be fully predictive (Assm B.1). Let c ∗ m ( z ) = w m · z m be the desired/clean linear main-task classifier trained using max-margin objective (§B.1) which only uses z m for its prediction. Let z p be the spurious feature s.t. for the margin points of c ∗ m ( z ) , z p be linearly-separable w.r.t. task label y m (Assm B.2). Then, assuming the latent space is centered around 0 (i.e. b m = 0 and b p = 0 ), the main-task classifier trained using max-margin objective will be of form c m ( z ) = w m · z m + w p · z p where w p = 0 .

The proof of Lemma B.1 is identical to Lemma 3.1 and is provided in §B.3.

## B.3 Proof of Sufficient Condition: Lemma 3.1 and Lemma B.1

Lemma 3.1. Let the latent representation be frozen and disentangled such that z = [ z m , z p ] (Assm 3.1), and conceptcausal features z p are fully predictive for the concept label y p (Assm 3.2). Let c ∗ p ( z ) = w p · z p where w p ∈ R d p be the desired clean linear classifier trained using the max-margin objective (§B.1) that only uses z p for its prediction. Let z m be the main task features, spuriously correlated s.t. z m are linearly-separable w.r.t. probing task label y p for the margin points of c ∗ p ( z ) (Assm 3.3). Then, assuming a zero-centered latent space ( b p = 0 ), a concept-probing classifier c p trained using the max-margin objective will use spurious features, i.e., c p ( z ) = w p · z p + w m · z m where w m = 0 and w m ∈ R d m .

̸

In this section we prove that, given the assumption in Lemma 3.1 is satisfied, they are sufficient for a probing classifier c p ( z ) to use the spuriously correlated main-task feature z m . See §B.1 for detailed setup and max-margin training objective. Also, we could use the same line of reasoning to prove a similar result for the main-task classifier i.e. when conditions in Lemma B.1 are satisfied, the main-task classifier will use the spuriously correlated concept-causal feature z p . To keep the proof general for both the lemmas, we prove the result for a general classifier c ( z ) trained to predict a task label y . Here the latent representation z be of form z = [ z inv , z sp ] where z inv are the features which are causally-derived from the task concept ('invariant' features) and z sp be the features spuriously correlated to the task label y . With respect to probing classifier c p ( z ) in Lemma 3.1 z inv := z p and z sp := z m . Similarly, for the main-task classifier in Lemma B.1, z inv := z m and z sp := z p . For ease of exposition, we define two categories of classifiers based on which features they use:

Definition B.1 (Purely-Invariant Classifier) . A linear classifier of form c ( z ) = w inv · z inv + w sp · z sp + b is called "purely-invariant" if it does not use the spurious features z sp i.e., w sp = 0 .

̸

Definition B.2 (Spurious-Using Classifier) . A linear classifier of form c ( z ) = w inv · z inv + w sp · z sp + b is called "spurious-using" if it uses the spurious features z sp i.e. w sp = 0 .

̸

Proof of Lemma 3.1 and B.1. Let c inv ( z ) = w inv · z inv be the clean /purely-invariant classifier trained using the max-margin objective using the MM-Denominator formulation given in Eq. 4 such that w inv = 0 . The classifier c inv ( z ) is 100% predictive of the task labels y (from Assm 3.2 for the probing task or Assm B.1 for the main-task). Here the bias term b = 0 since we assume the latent representation z is zero-centered. The norm of this classifier ( c inv ( z ) ) is ∥ w inv ∥ and the distance of each input latent representation ( z i ) with class label y i ( y i ∈ {-1 , 1 } ) from the decision boundary ( c inv ( z ) = 0 ) is given by Eq. 1 i.e.:

$$\mathcal { M } _ { i n v } ( z ^ { i } ) = \frac { m _ { i n v } ( z ^ { i } ) } { \| w _ { i n v } \| } = \frac { y ^ { i } \cdot c _ { i n v } ( z ^ { i } ) } { \| w _ { i n v } \| } = \frac { y ^ { i } \cdot ( w _ { i n v } \cdot z _ { i n v } ^ { i } ) } { \| w _ { i n v } \| }$$

Since we have used the MM-Denominator version of max-margin to train c inv ( z ) , from Eq. 3 we have m inv ( z i ) = 1 for the margin-points and greater than 1 for rest of the points. Next we will construct a new classifier parameterized by α ∈ [0 , 1] by perturbing the clean/purely-invariant classifier c inv ( z ) such that:

$$c _ { \alpha } ( z ) = \alpha ( w _ { i n v } \cdot z _ { i n v } ) + \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } ( \hat { \epsilon } _ { s p } \cdot z _ { s p } )$$

where ˆ ϵ sp ∈ R d sp is a unit vector in spurious subspace of features, d sp is the dimension of the spurious feature subspace ( z sp ). We observe that the norm of this perturbed classifier c α ( z ) is also ∥ w inv ∥ , which is same as the clean/purely-invariant classifier c inv ( z ) . Thus from Eq. 1, the distance of any input z i with class label y i from the decision boundary of this perturbed classifier c α ( z ) is given by:

$$\mathcal { M } _ { \alpha } ( z ^ { i } ) = \frac { m _ { \alpha } ( z _ { i } ) } { \| w _ { i n v } \| } = \frac { y ^ { i } \cdot c _ { \alpha } ( z ^ { i } ) } { \| w _ { i n v } \| }$$

The perturbed classifier will be spurious-using i.e use the spurious feature z sp when α ∈ [0 , 1) since w sp = ( ∥ w inv ∥ √ 1 -α 2 ) = 0 for these setting of α . Thus to show that there exist a spurious-using classifier which has a margin greater than the margin of the purely-invariant classifier, we need to prove that there exist an α ∈ [0 , 1) such that c α ( z ) has bigger margin than c inv ( z ) i.e. min z M α ( z ) &gt; min z M inv ( z ) . Since norm of parameters of both the classifier is same, substituting the expression of M α and M inv from Eq. 6 and 8, we need to show m α ( z i ) &gt; 1 for all z i . We have:

$$m _ { \alpha } ( z ^ { i } ) = y ^ { i } \cdot \left ( \alpha ( w _ { i n v } \cdot z _ { i n v } ^ { i } ) + \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { i } ) \right )$$

$$= \alpha \cdot m _ { i n v } ( z ^ { i } ) + y ^ { i } \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { i } )$$

Let S y m denote the set of margin-points of purely-invariant classifier c inv ( z ) with class label y having m inv ( z ) = 1 and S r y contain rest of points (non-margin points) having m inv ( z ) &gt; 1 with the class label y . Here 'm' stands for margin-point in superscript of S and 'r' stands for rest of point with label y . In rest of the proof, first we will show that for margin-points z m ∈ ( S y m =1 ∪S y m = -1 ) , we need the assumption that spurious feature ( z m sp ) be linearly separable with respect to class label y (Assm 3.3 for probing task or B.2 for main-task) for having m α ( z i ) &gt; 1 . But for all non-margin points z r ∈ ( S r y =1 ∪ S r y = -1 ) , we can always choose α ∈ [0 , 1) such that m α ( z i ) &gt; 1 . Below we handle margin and non-margin of points separately.

Case 1 : Margin Points ( S y m =1 ∪ S y m = -1 ) : For the margin-points in latent space, z m ∈ S y m we have m inv ( z m ) = 1 and we need to show that there exists α ∈ [0 , 1) such that m α ( z m ) &gt; 1 for all z m ∈ S y m . From Eq. 10 we have:

$$m _ { \alpha } ( z ^ { m } ) = \alpha \cdot 1 + y \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } \left ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } \right ) > 1$$

$$( \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } ) y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } ) > 1 - \alpha$$

From Assm 3.3 for probing task or B.2 for the main-task, we know that spurious-feature z m sp of margin-points z m are linearly-separable w.r.t to task label y . Since ˆ ϵ sp ∈ R d sp used in the perturbed classifier c α ( z ) is arbitrary, let's set it to be an unit vector such that y ( ˆ ϵ sp · z m sp ) &gt; 0 for all z m ∈ S y m (guaranteed by Assm 3.3 or B.2). Also since α ∈ [0 , 1) and ∥ w inv ∥ &gt; 0 , we have ( ∥ w inv ∥ √ 1 -α 2 ) &gt; 0 . Hence the left hand side of Eq. 12 is &gt; 0 for such ˆ ϵ sp . If Assm 3.3 or B.2 (corresponding to the task) wouldn't have been satisfied then the above equation might have been inconsistent since right hand side is always &gt; 0 ; since we need to find a solution to Eq. 12 when α ∈ [0 , 1) thus (1 -α ) &gt; 0 , but left hand side wouldn't have been always greater than 0. This shows the motivation why we need Assm 3.3 or B.2 for proving this lemma. Continuing, let β := ( y ( ˆ ϵ sp · z m sp ) ) , then squaring both sides and cancelling (1 -α ) since we need to find a solution to Eq. 12 when α ∈ [0 , 1) = ⇒ (1 -α ) &gt; 0 , we get:

$$\| w _ { i n v } \| ^ { 2 } ( \wr \widehat { \alpha } ) ( 1 + \alpha ) \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } ) \right ) ^ { 2 } > \wr ( \wr \wr \wr ) ( 1 - \alpha )$$

$$\| w _ { i n v } \| ^ { 2 } ( 1 + \alpha ) \beta ^ { 2 } > ( 1 - \alpha )$$

$$\| w _ { i n v } \| ^ { 2 } \beta ^ { 2 } + \alpha \| w _ { i n v } \| ^ { 2 } \beta ^ { 2 } > 1 - \alpha$$

$$\alpha \left ( 1 + \| w _ { i n v } \| ^ { 2 } \beta ^ { 2 } \right ) > \left ( 1 - \| w _ { i n v } \| ^ { 2 } \beta ^ { 2 } \right )$$

̸

After substituting back the value of β and rearranging we get:

$$& & 1 - \| w _ { i n v } \| ^ { 2 } \cdot \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } ) \right ) ^ { 2 } \\ & \alpha > \frac { 1 - \| w _ { i n v } \| ^ { 2 } \cdot \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } ) \right ) ^ { 2 } } { 1 + \| w _ { i n v } \| ^ { 2 } \cdot \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m } ) \right ) ^ { 2 } } \colon = \alpha _ { y } ^ { l b _ { 1 } } ( z ^ { m } )$$

Lets define α lb 1 y := max z m ∈S y m ( α lb 1 y ( z m )) . Since ∥ w inv ∥ 2 · ( y ( ˆ ϵ sp · z m sp ) ) 2 &gt; 0 , the right hand side of above equation α lb 1 y ( z m ) &lt; 1 for all z m ∈ S y m = ⇒ α lb 1 y &lt; 1 , which sets a new lower bound on allowed value of α for which Eq. 12 is satisfied. Thus when α ∈ ( α lb 1 y , 1) , m p ( z m ) &gt; 1 for all z m ∈ S y m . That is, the perturbed probing classifier c α ( z ) has larger margin than purely-invariant/clean classifier c inv ( z ) for the margin points z m ∈ S y m .

̸

Case 2: Non-Margin Points ( S r y =1 ∪ S r y = -1 ) : For the non-margin points z r ∈ S r y in the latent space we have m inv ( z r ) &gt; 1 . Let γ := min z r ∈S r y ( m inv ( z r ) ) thus we also have γ &gt; 1 . Let α = 0 and we choose α such that:

$$\frac { 1 } { \alpha } < \gamma \\$$

$$\alpha > \frac { 1 } { \gamma }$$

$$\alpha > \frac { 1 } { \min _ { z ^ { r } \in \mathcal { S } _ { y } ^ { r } } \left ( m _ { i n v } ( z ^ { r } ) \right ) } = \alpha _ { y } ^ { l b _ { 2 } }$$

Since γ &gt; 1 , thus right hand side in above equation α lb 2 y &lt; 1 , which sets a new lower bound on allowed values of α . Since m inv ( z r ) ≥ γ &gt; 1 α for all z r ∈ S r y for α ∈ ( α lb 2 y , 1) (Eq. 20), we can write m inv ( z r ) = 1 α + η ( z r ) where η ( z r ) := ( m inv ( z r ) -1 α ) &gt; 0 for all z r ∈ S r y . Now we need to show that there exist an α ∈ ( α lb 2 y , 1) such that m α ( z r ) &gt; 1 for all z r ∈ S r y . Thus from Eq. 10 we need:

$$m _ { \alpha } ( z ^ { r } ) = \alpha \cdot m _ { i n v } ( z ^ { r } ) + \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { r } ) \right ) > 1$$

$$\alpha \cdot ( \frac { 1 } { \alpha } + \eta ( z ^ { r } ) ) + \| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { r } ) \right ) > 1$$

$$\| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } \left ( y ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { r } ) \right ) > - ( \alpha \cdot \eta ( z ^ { r } ) )$$

Since α ∈ ( α 2 lb , 1) , we have ( α · η ( z r )) &gt; 0 and ∥ w inv ∥ √ 1 -α 2 &gt; 0 . Let's define δ ( z r ) := y ( ˆ ϵ sp · z r sp ) . Thus for the latent-points z r ∈ S r y which have δ ( z r ) ≥ 0 , Eq. 23 is always satisfied since left side of inequality is greater than or equal to zero and right side is always less than zero. For the points for which δ ( z r ) &lt; 0 we have:

$$\| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } \cdot ( - 1 ) \cdot | \delta ( z ^ { r } ) | > - ( \alpha \cdot \eta ( z ^ { r } ) )$$

$$\| w _ { i n v } \| \sqrt { 1 - \alpha ^ { 2 } } | \delta ( z ^ { r } ) | < \left ( \alpha \cdot \eta ( z ^ { r } ) \right )$$

$$\| w _ { i n v } \| ^ { 2 } ( 1 - \alpha ^ { 2 } ) \delta ( z ^ { r } ) ^ { 2 } < \left ( \alpha \cdot \eta ( z ^ { r } ) \right ) ^ { 2 }$$

$$\| w _ { i n v } \| ^ { 2 } \delta ( z ^ { r } ) ^ { 2 } < \alpha ^ { 2 } \cdot \left ( \eta ( z ^ { r } ) ^ { 2 } + \| w _ { i n v } \| ^ { 2 } \delta ( z ^ { r } ) ^ { 2 } \right )$$

$$\alpha > \sqrt { \frac { \| w _ { i n v } \| ^ { 2 } \delta ( z ^ { r } ) ^ { 2 } } { \eta ( z ^ { r } ) ^ { 2 } + \| w _ { i n v } \| ^ { 2 } \delta ( z ^ { r } ) ^ { 2 } } } = \alpha _ { y } ^ { l b _ { 3 } } ( z ^ { r } )$$

Now different z r will have different η ( z r ) which will give different lower bound of α . Since the m α ( z r ) &gt; 1 has to be satisfied for every point in z r ∈ S r y we will choose the maximum value of α lb 3 y ( z r ) which will give tightest lower bound on value of α . Lets define α lb 3 y := max z r ∈S r y ( z r ) , then for m α ( z r ) &gt; 1 , we need α &gt; α lb 3 y . Also, since for all z r ∈ S r y , η ( z r ) &gt; 0 we have α lb 3 y ( z r ) &lt; 1 = ⇒ α lb 3 y &lt; 1 .

Substituting the value of γ we get:

Finally, combining all the lower bound of α from Eq. 17, Eq. 20 and Eq. 28 let the overall lower bound of α be α lb given by:

$$\alpha _ { l b } = \max \{ \alpha _ { y = 1 } ^ { l b _ { 1 } } , \alpha _ { y = - 1 } ^ { l b _ { 1 } } , \alpha _ { y = 1 } ^ { l b _ { 2 } } , \alpha _ { y = - 1 } ^ { l b _ { 2 } } , \alpha _ { y = 1 } ^ { l b _ { 3 } } , \alpha _ { y = - 1 } ^ { l b _ { 3 } } , \}$$

This provides a way to construct a spurious-using classifier: given any purely-invariant classifier, we can always choose α ∈ ( α lb , 1) and construct a perturbed spurious-using classifier from Eq. 7 which has a bigger margin than purely-invariant . Thus, given all the assumptions, there always exists a spurious-using classifier which has greater margin than the purely-invariant classifier.

## B.4 Proof of necessary condition

In this section we will show that Assm 3.3 is also a necessary condition for the probing classifier to use the spuriously correlated main task features ( z m ) when the dimension of concept-causal feature d p = 1 . That is, the probing classifier will use the spurious features if and only if spurious features satisfy Assm 3.3 for the margin points of the clean/purelyinvariant (Def B.1) probing classifier when the concept-causal feature is 1-dimensional. Also, same line of reasoning will hold for the main-task classifier where we will show that main-task classifier will use the spurious feature ( z p ) iff spurious feature satisfies Assm B.2 for the margin point of clean main-task classifier. Formally:

̸

Lemma B.2 (Necessary Condition for concept-Probing Classifier) . Let the latent representation be frozen and disentangled (Assm 3.1) such that z = [ z m , z p ] where z p is the concept-causal feature which is 1-dimensional scalar and fully predictive (Assm 3.2) and z m ∈ R d m . Let c ∗ p ( z ) = w p · z p be the desired clean/purely-invariant probing classifier trained using max-margin objective which only uses z p for prediction. Then the probing classifier trained using max-margin objective will be spurious-using i.e. c p ( z ) = w p · z p + w m · z m where w m = 0 iff the spurious feature z m is linearly separable w.r.t to probing task label y p for the margin point of c ∗ p ( z ) (Assm 3.3).

̸

Lemma B.3 (Necessary Condition for Main-task Classifier) . Let the latent representation be frozen and disentangled (Def 3.1) such that z = [ z m , z p ] where z m is the main-task feature which is 1-dimensional scalar and fully predictive (Assm B.1) and z p ∈ R d p . Let c ∗ m ( z ) = w m · z m be the desired clean/purely-invariant main-task classifier trained using max-margin objective which only uses z m for prediction. Then the main-task classifier trained using max-margin objective will be spurious-using i.e. c m ( z ) = w m · z m + w p · z p where w p = 0 iff the spurious feature z p is linearly separable w.r.t to main task label y m for the margin point of c ∗ m ( z ) (Assm B.2) .

Since proof of both Lemma B.2 and B.3 follows same line of reasoning, hence for brevity, following §B.3, we will prove the lemma for a general classifier c ( z ) trained using max-margin objective to predict the task-label y . Let the latent representation be of form z = [ z inv , z sp ] where z inv ∈ R is the feature causally derived from the task concept and z sp ∈ R d sp is the feature spuriously correlated to task label y . With respect to probing classifier c p ( z ) in Lemma B.2 z inv := z p and z sp := z m . Similarly, for the main-task classifier in Lemma B.3, z inv := z m and z sp := z p .

Proof of Lemma B.2 and B.3. Our goal is to show that Assm 3.3 for probing classifier or Assm B.2 for the main-task classifier is necessary for obtaining a spurious-using classifier for the case when z inv is one-dimensional. We show this by assuming that optimal classifier is spurious-using even when Assm 3.3 or B.2 breaks and then show that this will lead to contradiction.

Contradiction Assumption: Formally, let's assume that Assm 3.3 or B.2 is not satisfied for probing or main task respectively, and the optimal classifier for the given classification task is spurious-using c ∗ ( z ) , where:

$$c _ { * } ( z ) = w _ { i n v } ^ { * } \cdot z _ { i n v } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } )$$

where ∥ w ∗ sp ∥ ̸ = 0 and ˆ w ∗ sp ∈ R d sp is a unit vector in spurious-feature subspace with dimension d sp .

Let c inv ( z ) = w inv · z inv be the optimal purely-invariant classifier. Let both c ∗ ( z ) and c inv ( z ) be trained using the max-margin objective using MM-Denominator formulation in Eq. 4. Thus from the constraints of this formulation (Eq. 3), for all z we have:

$$m _ { * } ( z ) = y \cdot c _ { * } ( z ) = y \cdot ( w _ { i n v } ^ { * } \cdot z _ { i n v } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } ) ) \geq 1 \ , \&$$

$$m _ { i n v } ( z ) = y \cdot c _ { i n v } ( z ) = y \cdot ( w _ { i n v } \cdot z _ { i n v } ) \geq 1$$

From Assm 3.2 or B.1, the invariant feature z inv is 100% predictive and linearly separable w.r.t task label y . Then without loss of generality let's assume that:

$$z _ { i n v } > 0 , \text { when } y = + 1$$

$$z _ { i n v } < 0 , \text { when } y = - 1$$

From Eq. 33 and 34 we have y · z inv &gt; 0 thus from Eq. 32 we get:

$$w _ { i n v } \geq 0$$

Also, from our contradiction-assumption the max-margin trained classifier is spurious-using , thus the norm of parameters of c ∗ ( z ) is less or equal to c inv ( z ) (Eq. 4). Thus we have:

$$\sqrt { ( w _ { i n v } ^ { * } ) ^ { 2 } + ( \| w _ { s p } ^ { * } \| ) ^ { 2 } } \leq | w _ { i n v } |$$

$$\implies | w _ { i n v } ^ { * } | < | w _ { i n v } | \quad ( \| w _ { s p } ^ { * } \| \neq 0 )$$

$$\Longrightarrow | w _ { i n v } ^ { * } | < w _ { i n v } \quad ( w _ { i n v } \geq 0 , E q . 3 5 )$$

$$\Longrightarrow w _ { i n v } ^ { * } < w _ { i n v }$$

Form our contradiction-assumption , Assm 3.3 for concept-probing task or Assm B.2 for the main-task breaks by one of the following two ways:

1. Opposite Side Failure: This occurs when the spurious part of margin points (of c inv ( z ) ) on opposite sides of decision-boundary of the optimal task classifier ( c ∗ ( z ) = 0 ) are not linearly-separable with respect to task label y . Formally, there exist two datapoints, P m + := [ z m + inv , z m + sp ] and P m -:= [ z m -inv , z m -sp ] such that they are margin points of purely-invariant classifier c inv ( z ) where P m + has class label y = +1 and P m -has class label y = -1 and ∀ ˆ ϵ sp ∈ R d sp , the spurious feature z sp of both the points lies on same side of ˆ ϵ sp i.e:

$$\left ( ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m + } ) \cdot ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m - } ) \right ) \geq 0$$

2. Same Side Failure: This occurs when the spurious part of margin points (of c inv ( z ) ) on same side of decisionboundary ( c ∗ ( z ) = 0 ) are linearly-separable. Formally, there exist two datapoints, P m 1 y := [ z m 1 inv , z m 1 sp ] and P m 2 y := [ z m 2 inv , z m 2 sp ] such that they are margin points of purely-invariant classifier c inv ( z ) and both points have same class label y and ∀ ˆ ϵ sp ∈ R d sp , w.l.o.g we have:

$$\left ( ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m 1 } ) \cdot ( \hat { \epsilon } _ { s p } \cdot z _ { s p } ^ { m 2 } ) \right ) \leq 0 .$$

We will use the following two lemma to proceed with our proof:

Lemma B.4. If Assm 3.3 or B.2 breaks by opposite-side failure mode, it leads to contradiction.

Lemma B.5. If Assm 3.3 or B.2 breaks by same-side failure mode, it leads to contradiction.

This implies that our contradiction-assumption which said that the max-margin trained optimal classifier is spurioususing even when Assm 3.3 or B.2 breaks, is wrong. Thus Assm 3.3 for concept-probing task or Assm B.2 for main-task is necessary for the optimal max-margin classifier to be spurious-using . This completes our proof.

Proof of Lemma B.4. We have two points, P m + := [ z m + inv , z m + sp ] and P m -:= [ z m -inv , z m -sp ] , which break the Assm 3.3 or B.2. From Eq. 33, z inv &gt; 0 for all the points with label y = 1 , thus we have z m + inv &gt; 0 and using Eq. 39 ( w ∗ inv &lt; w inv ) we get:

$$w _ { i n v } ^ { * } < w _ { i n v }$$

$$w _ { i n v } ^ { * } \cdot z _ { i n v } ^ { m + } < w _ { i n v } \cdot z _ { i n v } ^ { m + }$$

$$w _ { i n v } ^ { * } \cdot z _ { i n v } ^ { m + } < 1$$

where the right hand side w inv · z m + inv = 1 since P m + is the margin-point of c inv ( z ) (Eq. 32). Similarly from Eq. 34, z inv &lt; 0 for all the points with label y = -1 , thus we have z m -inv &lt; 0 and using Eq. 39 ( w ∗ inv &lt; w inv ) we get:

$$w _ { i n v } ^ { * } < w _ { i n v }$$

$$( - 1 ) \cdot w _ { i n v } ^ { * } \cdot z _ { i n v } ^ { m - } < ( - 1 ) \cdot w _ { i n v } \cdot z _ { i n v } ^ { m - }$$

$$( - 1 ) \cdot w _ { i n v } ^ { * } \cdot z _ { i n v } ^ { m - } < 1$$

where the right hand side ( -1) · ( w p inv · z m -inv ) = 1 since P m -is the margin-point of c inv ( z ) (Eq. 32).

Next from Eq. 31 we have m ∗ ( z ) ≥ 1 for all z hence it is also true for P m + with y = 1 and P m -with y = -1 . Then:

$$m _ { * } ( P ^ { m + } ) = y \cdot c _ { * } ( P ^ { m + } ) = 1 \cdot \left \{ w _ { i n v } ^ { * } z _ { i n v } ^ { m + } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } ^ { m + } ) \right \} \geq 1$$

$$\implies w _ { i n v } ^ { * } z _ { i n v } ^ { m + } + \| w _ { s p } ^ { * } \| \cdot \beta ^ { m + } \geq 1$$

$$\implies w _ { i n v } ^ { * } z _ { i n v } ^ { m + } \geq 1 - \| w _ { s p } ^ { * } \| \cdot \beta ^ { m + }$$

where β m + = ( ˆ w ∗ sp · z m + sp ) . Also we have:

$$m _ { * } ( P ^ { m - } ) = y \cdot c _ { * } ( P ^ { m - } ) = - 1 \cdot \left \{ w _ { i n v } ^ { * } z _ { i n v } ^ { m - } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } ^ { m - } ) \right \} \geq 1$$

$$\Longrightarrow \, - w _ { i n v } ^ { * } z _ { i n v } ^ { m - } - \| w _ { s p } ^ { * } \| \cdot \beta ^ { m - } \geq 1$$

$$\Longrightarrow - w _ { i n v } ^ { * } z _ { i n v } ^ { m - } \geq 1 + \| w _ { s p } ^ { * } \| \cdot \beta ^ { m - }$$

where β m -= ( ˆ w ∗ sp · z m -sp ) . From Eq. 40 we have ( ( ˆ ϵ sp · z m + sp ) · ( ˆ ϵ sp · z m -sp ) ) ≥ 0 for all ˆ ϵ sp ∈ R d sp which states the opposite-side failure of Assm 3.3 or B.2. Thus:

$$\beta ^ { m + } \cdot \beta ^ { m - } \geq 0$$

Now we will show that Eq. 44, 47, 50 and 53 cannot be satisfied simultaneously for any allowed value of β m + and β m -(given by Eq. 54) which are:

1. β m + &gt; 0 and β m -&gt; 0 : From Eq. 53 we have -w ∗ inv z m -inv &gt; 1 since ∥ w ∗ sp ∥ ̸ = 0 and β m -&gt; 0 . But from Eq. 47 we have -w ∗ inv z m -inv &lt; 1 which is a contradiction.
2. β m + &lt; 0 and β m -&lt; 0 : From Eq. 50 we have w ∗ inv z m + inv &gt; 1 since ∥ w ∗ sp ∥ ̸ = 0 and β m + &lt; 0 . But from Eq. 44 we have w ∗ inv z m + inv &lt; 1 which is a contradiction.
3. β m + = 0 and β m -∈ R : From Eq. 50 we have w ∗ inv z m + inv ≥ 1 but from Eq. 44 we have w ∗ inv z m + inv &lt; 1 which is a contradiction.
4. β m + ∈ R and β m -= 0 : From Eq. 53 we have -w ∗ inv z m -inv ≥ 1 but from Eq. 47 we have -w ∗ inv z m -inv &lt; 1 which is a contradiction.

Thus we have a contradiction for all the possible values β m + and β m -could take, completing the proof of this lemma.

Proof of Lemma B.5. We have two margin-points, P m 1 y := [ z m 1 inv , z m 1 sp ] and P m 2 y := [ z m 2 inv , z m 2 sp ] , which break Assm 3.3 or B.2. From Eq. 33 and Eq. 34 we have y · z m 1 inv &gt; 0 and y · z m 2 inv &gt; 0 . Using Eq. 39 ( w ∗ inv &lt; w inv ) we get:

$$w _ { i n v } ^ { * } < w _ { i n v }$$

$$w _ { i n v } ^ { * } \cdot ( y \cdot z _ { i n v } ^ { m j } ) < w _ { i n v } \cdot ( y \cdot z _ { i n v } ^ { m j } )$$

$$y \cdot ( w _ { i n v } ^ { * } \cdot z _ { i n v } ^ { m j } ) < 1$$

where j ∈ { 1 , 2 } and right hand side w inv · ( y · z mj inv ) = 1 since P mj y is the margin point of purely-invariant classifier c inv ( z ) (Eq. 32).

From Eq. 31 we have m ∗ ( z ) ≥ 1 for all z thus also true for P m 1 y and P m 2 y . Then:

$$m _ { * } ( P _ { y } ^ { m 1 } ) = y \cdot c _ { * } ( P _ { y } ^ { m 1 } ) = y \cdot \left \{ w _ { i n v } ^ { * } z _ { i n v } ^ { m 1 } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } ^ { m 1 } ) \right \} \geq 1$$

$$\Longrightarrow \, y \cdot ( w _ { i n v } ^ { * } z _ { i n v } ^ { m 1 } ) + y \cdot ( \| w _ { s p } ^ { * } \| \cdot \beta ^ { m 1 } ) \geq 1$$

$$\implies y \cdot ( w _ { i n v } ^ { * } z _ { i n v } ^ { m 1 } ) \geq 1 - y \cdot ( \| w _ { s p } ^ { * } \| \cdot \beta ^ { m 1 } )$$

where β m 1 = ( ˆ w ∗ sp · z m 1 sp ) . Also we have:

$$m _ { * } ( P _ { y } ^ { m 2 } ) = y \cdot c _ { * } ( P _ { y } ^ { m 2 } ) = y \cdot \left \{ w _ { i n v } ^ { * } z _ { i n v } ^ { m 2 } + \| w _ { s p } ^ { * } \| ( \hat { w } _ { s p } ^ { * } \cdot z _ { s p } ^ { m 2 } ) \right \} \geq 1$$

$$\Longrightarrow \, y \cdot ( w _ { i n v } ^ { * } z _ { i n v } ^ { m 2 } ) + y \cdot ( \| w _ { s p } ^ { * } \| \cdot \beta ^ { m 2 } ) \geq 1$$

$$\implies y \cdot ( w _ { i n v } ^ { * } z _ { i n v } ^ { m 2 } ) \geq 1 - y \cdot ( \| w _ { s p } ^ { * } \| \cdot \beta ^ { m 2 } )$$

where β m 2 = ( ˆ w ∗ sp · z m 2 sp ) . Now from Eq. 41 we have ( ( ˆ ϵ sp · z m 1 sp ) · ( ˆ ϵ sp · z m 2 sp ) ) ≤ 0 for all unit vectors ˆ ϵ sp ∈ R d sp which states the same-side failure mode of Assm 3.3 or B.2. Thus we have:

$$\beta ^ { m 1 } \cdot \beta ^ { m 2 } \leq 0$$

Now we will show that for all allowed values of β m 1 and β m 2 , Eq. 57, 60 and 63 will lead to a contradiction. Following are the cases for different allowed values of β m 1 and β m 2 :

1. β m 1 = 0 and β m 2 ∈ R : Substituting β m 1 = 0 in Eq. 60 we get y · ( w ∗ inv z m 1 inv ) ≥ 1 , but from Eq. 57 we have y · ( w ∗ inv z m 1 inv ) &lt; 1 . Thus we have a contradiction.
2. β m 1 ∈ R and β m 2 = 0 : Substituting β m 2 = 0 in Eq. 63 we get y · ( w ∗ inv z m 2 inv ) ≥ 1 , but from Eq. 57 we have y · ( w ∗ inv z m 2 inv ) &lt; 1 . Thus we have a contradiction.
3. The only case which is left now is when both β m 1 and β m 2 is non-zero but of opposite sign. Without loss of generality, let β m 1 &gt; 0 , β m 2 &lt; 0 and y = (+1) : Substituting β m 2 &lt; 0 and y = (+1) in Eq. 63 we get y · ( w ∗ inv z m 2 inv ) ≥ 1 , but from Eq. 57 we have y · ( w ∗ inv z m 2 inv ) &lt; 1 . Thus we have a contradiction.
4. β m 1 &gt; 0 , β m 2 &lt; 0 and y = ( -1) : Substituting β m 1 &gt; 0 and y = ( -1) in Eq. 60 we get y · ( w ∗ inv z m 1 inv ) ≥ 1 , but from Eq. 57 we have y · ( w ∗ inv z m 1 inv ) &lt; 1 . Thus we have a contradiction.

Thus we have a contradiction for all the possible values β m 1 , β m 2 and y could take, completing the proof of this lemma.

## C Null-Space Removal Failure: Setup and Proof of Theorem 3.2

## C.1 Null-Space Setup

As described in §3, the given main-task classifier have an encoder h : X → Z mapping the input X to latent representation Z . Post that, the main-task classifier c m : Z → Y m is used to predict the main-task label y i m from latent representation z i for every input x i . Given this (pre) trained main-task classifier the goal of a post-hoc removal method is to remove any undesired/sensitive/spurious concept from the latent representation Z without retraining the encoder h or main-task classifier c m ( z ) .

The null space method [32, 13], henceforth referred to as INLP , is one such post-hoc removal method that removes a concept from latent space by projecting the latent space to a subspace that is not discriminative of that attribute. First, it estimates the subspace in the latent space discriminative of the concept we want to remove by training a probing classifier c p ( z ) → y p , where y p is the concept label. [32] used a linear probing classifier ( c p ( z )) to ensure that the any linear classifier cannot recover the removed concept from modified latent representation z ′ and hence the main task classifier ( c m ( z ′ ) ), which is also a linear layer, become invariant to removed attribute. Let linear probing classifier c p ( z ) be parametrized by a matrix W , and null-space of matrix W is defined as space N ( W ) = { z | W z = 0 } . Give the basis vectors for the N ( W ) we can construct the projection matrix P N ( W ) such that W ( P N ( W ) z ) = 0 for all z . This projection matrix is defined as the guarding operator g := P N ( W ) (estimated by c p ( z ) ), when applied on the z will remove the features which are discriminative of undesired concept from z . For the setting when Y p is binary we have:

$$P _ { N ( W ) } = I - \hat { w } \hat { w } ^ { T }$$

where I is the identity matrix and ˆ w is the unit vector in the direction of parameters of classifier c p ( z ) ([32]). Also, the authors recommend running this removal step for multiple iterations to ensure that the unwanted concept is removed completely. Thus after S steps of removal, the final guarding function is:

$$g \coloneqq \prod _ { i = 1 } ^ { S } P _ { N ( W ) } ^ { i }$$

where P i N ( W ) is the projection matrix at i th removal step. Amnesic Probing ([13]) builds upon this idea for testing whether concept is being used by a given pre-trained classifier or not. The core idea is to remove the concept we want to test from the latent representation. If the prediction of the given classifier is influenced by this removal then the concept was being used by the given classifier otherwise not.

## C.2 Null-Space Removal Failure : Proof of Theorem 3.2

̸

Theorem 3.2. Let c m ( z ) be a pre-trained main-task classifier where the latent representation z = [ z m , z p ] satisfies Assm 3.1 and 3.2. Let c p ( z ) be the probing classifier used by INLP to remove the unwanted features z p from the latent representation. Under Assm 3.3, Lemma 3.1 is satisfied for the probing classifier c p ( z ) such that c p ( z ) = w p · z p + w m · z m and w m = 0 . Then,

1. Damage in the first step of INLP. The first step of linear-INLP will corrupt the main-task features and this corruption is non-invertible with subsequent projection steps of INLP.

̸

- (a) Mixing: If w p = 0 , the main task z m and concept-causal features z p will get mixed such that z i (1) = [ g ( z i m , z i p ) , f ( z i p , z i m )] = [ g ( z i m ) , f ( z i p )] for some function ' f ' and ' g '. Thus, the latent representation is no longer disentangled and removal of concept-causal features will also lead to removal of main task features.

̸

- (b) Removal: If w p = 0 , then the first projection step of INLP will do opposite of what is intended, i.e., damage the main task features z m (in case z m ∈ R , it will completely remove z m ) but have no effect on the concept-causal features z p .
2. Removal in the long term: The L2-norm of the latent representation z decreases with every projection step as long as the parameters of probing classifier ( w k ) at a step ' k ' does not lie completely in the space spanned by parameters of previous probing classifiers, i.e., span( w 1 , . . . , w k -1 ), z i ( k -1) , z i (0) and z i (0) in direction of w k is not trivially zero. Thus, after sufficiently many steps, INLP can destroy all information in the representation s.t. z i ( ∞ ) = [ 0 , 0 ] .

The proof of Theorem 3.2 proceeds in following steps:

1. First using Lemma 3.1, we show that even under very favourable conditions probing classifier will not be clean i.e will also use other features in addition to concept-causal feature for it's prediction. Then, for the more likely case when probing classifier uses both main-task and concept-causal feature, we show that after first step of null-space projection (INLP), both the main-task features and concept-causal features get mixed .
2. Next, for the extreme case when probing classifier uses only main-task feature, the first step of INLP will do opposite of what is intended. It will damage the main-task feature but will have no effect on the concept-causal feature which we wanted to remove from latent space representation.
3. In addition, we also show that the damage or mixing of latent space after first step of INLP projection cannot be corrected in subsequent step since the projection operation is non-invertible.
4. Next, we show that the projection operation is lossy, i.e removes the norm of latent representation under some conditions. Hence after sufficient steps, INLP could destroy all the information in latent representation.

̸

Proof of Theorem 3.2. First Claim (1a). Let c p ( z ) = w p z p + w m z m be the linear probing classifier trained to predict the concept label y p from the latent representation z . Since all the assumptions of Lemma 3.1 are satisfied for the probing classifier c p ( z ) , it is spurious using , i.e., w m = 0 and for the claim 1(a) we have w p = 0 . Since the concept label y p is binary, the projection matrix for the first step of INLP removal is defined as P 1 N ( W ) = I -ˆ w ˆ w T where ˆ w T = [ ˆ w m , ˆ w p ] , ˆ w m and ˆ w p are the unit norm parameters of c p ( z ) i.e w m and w p respectively. On applying this projection on the latent space representation z i we get new projected representation z i (1) s.t.:

̸

$$\text {this projection on the latent space representation} \, z ^ { i } \, \text {we get new projected representation} \, z ^ { i ( 1 ) } \, s . t . & \\ & \begin{bmatrix} z _ { m } ^ { i ( 1 ) } \\ z _ { p } ^ { i } \end{bmatrix} = \left ( I - \begin{bmatrix} \hat { w } _ { m } \\ \hat { w } _ { p } \end{bmatrix} \, \hat { w } _ { m } \right ) \, \hat { w } _ { p } \right ) \left ( \begin{bmatrix} z _ { m } ^ { i } \\ z _ { p } ^ { i } \end{bmatrix} \\ & = \begin{bmatrix} z _ { m } ^ { i } \\ z _ { p } ^ { i } \end{bmatrix} - \hat { c } _ { p } ( z ^ { i } ) \cdot \left [ \hat { w } _ { p } ^ { m } \right ] & \text {define } \hat { c } _ { p } ( z ^ { i } ) \equiv \hat { w } _ { m } \cdot z _ { m } ^ { i } + \hat { w } _ { p } \cdot z _ { p } ^ { i } \\ & = \left [ \hat { z } _ { p } ^ { i } - \hat { c } _ { p } ( z ^ { i } ) \hat { w } _ { m } \right ] \\ & = \left [ \hat { z } _ { p } ^ { i } - \hat { c } _ { p } ( z ^ { i } ) \hat { w } _ { p } \right ] \\ & = \left [ g ( z _ { m } ^ { i } , z _ { p } ^ { i } ) \right ] \\ & = f ( z _ { m } ^ { i } , z _ { p } ^ { i } ) \right ] \\ \text {Next, we will show that the main task features and probing features get mixed after projection. To do so, we first show}$$

̸

Next, we will show that the main task features and probing features get mixed after projection. To do so, we first show that g ( z i m , z i p ) = g ( z i m ) for some function g i.e projected main task features z i (1) m = g ( z i m , z i p ) are not independent of

probing features z i p . From Eq. 69, we have:

$$z _ { m } ^ { i ( 1 ) } = g ( z _ { m } ^ { i } , z _ { p } ^ { i } )$$

$$= z _ { m } ^ { i } - ( \hat { w } _ { m } \cdot z _ { m } ^ { i } + \hat { w } _ { p } \cdot z _ { p } ^ { i } ) \hat { w } _ { m }$$

$$= ( I - \hat { w } _ { m } \hat { w } _ { m } ^ { T } ) z _ { m } ^ { i } - ( \hat { w } _ { p } \cdot z _ { p } ^ { i } ) \hat { w } _ { m }$$

̸

In this case we are given w p = 0 and w m = 0 . Since z i p can take any value (subject to Assm 3.2), ˆ w p · z i p is not trivially zero for all z i p = ⇒ ( ˆ w p · z i p ) ˆ w m = 0 . Thus g ( z i m , z i p ) is not independent of z i p .

̸

̸

Next, we will show that f ( z i m , z i p ) = f ( z i p ) for some function f i.e projected probing feature z i (1) p = f ( z i m , z i p ) is not independent of the main task feature z i m . From Eq. 69, we have:

$$z _ { p } ^ { i ( 1 ) } = f ( z _ { m } ^ { i } , z _ { p } ^ { i } )$$

$$= z _ { p } ^ { i } - ( \hat { w } _ { m } \cdot z _ { m } ^ { i } + \hat { w } _ { P } \cdot z _ { p } ^ { i } ) \hat { w } _ { p }$$

$$= ( I - \hat { w } _ { p } \hat { w } _ { p } ^ { T } ) z _ { p } ^ { i } - ( \hat { w } _ { m } \cdot z _ { m } ^ { i } ) \hat { w } _ { p }$$

̸

Again, in this case we are given w p = 0 and w m = 0 . Since z i m can take any value (subject to Assm 3.3), ˆ w m · z i m is not trivially zero for all z i m = ⇒ ( ˆ w m · z i m ) ˆ w p = 0 . Thus f ( z i m , z i p ) is not independent of z i m . Hence both concept-feature z p and the main-task feature z m got mixed after the first step of projection.

̸

Next, we will show that this mixing of the main task and concept-causal feature cannot be corrected in subsequent steps of null-space projection. Formally, the following Lemma C.1 proves that the above projection matrix ( P 1 N ( W ) ) which resulted in mixing of features is non-invertible. The subsequent steps of INLP applies projection transformation which can be combined into one single matrix P &gt; 1 N ( W ) = ∏ j&gt; 1 P j N ( W ) . In order for mixing to be reversed, we need P &gt; 1 N ( W ) × P 1 N ( W ) = I , thus we need P &gt; 1 N ( W ) = ( P 1 N ( W ) ) -1 which is not possible from Lemma C.1. Hence the mixing of the main-task feature and the concept-causal feature which happened after the first step of projection cannot be corrected in the subsequent steps of INLP thus completing the first claim of our proof.

Lemma C.1. The projection matrix P j N ( W ) at any projection step of INLP is non invertible.

Proof of Lemma C.1. The projection matrix for binary target case is defined as P := P j N ( W ) = I -A where A = ˆ w ˆ w T be a n × n matrix and w is the parameter vector of the probing classifier c p ( z ) trained at j th -step of INLP. We can see that A is a symmetric matrix. Every symmetric matrix is diagonalizable (Equation W.9 in [11]), hence we can write A = Q Λ Q T , where Q is a some orthonormal matrix such that QQ T = I and Λ = diag ( λ 1 , . . . , λ n ) be a n × n diagonal matrix where the diagonal entries ( λ 1 . . . λ n ) are the eigen-values of A . Since QQ T = I we can write P = I -A = QQ T -Q Λ Q T = Q ( I -Λ) Q T . Next, for the projection matrix P to be invertible P -1 should exist. We have:

$$P ^ { - 1 } = \left ( Q ( I - \Lambda ) Q ^ { T } \right ) ^ { - 1 }$$

$$= ( Q ^ { T } ) ^ { - 1 } ( I - \Lambda ) ^ { - 1 } Q ^ { - 1 }$$

$$= Q ( I - \Lambda ) ^ { - 1 } Q ^ { T }$$

So projection matrix is only invertible when ( I -Λ) is invertible. We will show next that ( I -Λ) is not invertible thus completing our proof. We have I -Λ = diag (1 -λ 1 , . . . , 1 -λ n ) , hence:

$$( I - \Lambda ) ^ { - 1 } = d i a g ( \frac { 1 } { 1 - \lambda _ { 1 } } , \dots , \frac { 1 } { 1 - \lambda _ { 2 } } )$$

Now, if one of the eigenvalues of A is 1 , then the diagonal matrix ( I -Λ) is not invertible. If one of the eigenvalues of A is 1 , then there exist an eigenvector x such that A x = ˆ w ˆ w T × x = 1 × x . The vector x = ˆ w is the eigenvector of A with eigenvalue 1 : A ˆ w = ˆ w ˆ w T × ˆ w = 1 × ˆ w since ˆ w T × ˆ w = 1 as it is a unit vector. Hence the projection matrix is not invertible.

̸

̸

̸

̸

First Claim (1b). For a probing classifier of form c (1) p ( z ) = w p · z p + w m · z m for the first step of INLP projection -denoted by superscript (1)- trained to predict concept label y p and Assm 3.1,3.2 and 3.3 of Lemma 3.1 is satisfied then we have w m = 0 i.e main task feature z m will be used by probing classifier along with the concept feature z p . For this second case, we are given that w p = 0 i.e probing classifier will not use concept feature at all. This is only possible when the main-task feature is fully predictive of the concept label i.e Assm 3.3 is satisfied for all the points in the dataset, otherwise optimal probing classifier will use the concept-causal feature to achieve better margin and accuracy. Moreover, even if we assume Assm 3.3 is satisfied for all the points in the dataset, to have w p = 0 , the margin achieved by probing classifier using only main task feature ( z m ) should be bigger than any other classifier i.e one using both the main-task feature and probing feature or using probing feature alone. Thus, it is very unlikely that the optimal probing classifier will have w p = 0 .

Having said this, even in the case when we have w p = 0 , we show that the first projection step of INLP will do something unintended, i.e., damage the main-task features while having no effect on concept-causal features which we intended to remove. First, we will show that main-task features will get damaged. From Eq. 73 we have:

$$z _ { m } ^ { i ( 1 ) } = ( I - \hat { w } _ { m } \hat { w } _ { m } ^ { T } ) z _ { m } ^ { i } - ( \hat { w } _ { p } \cdot z _ { p } ^ { i } ) \hat { w } _ { m }$$

$$= z _ { m } ^ { i } - ( \hat { w } _ { m } \cdot z _ { m } ^ { i } ) \hat { w } _ { m } - 0 \quad ( \text {since } w _ { p } = 0 )$$

̸

Since w m = 0 and z i m can take any value (subject to Assm 3.3), ˆ w m · z i m is not trivially zero for all the z i m = ⇒ ( ˆ w m · z i m ) ˆ w m = 0 . Thus, projected main-task feature z i (1) m = z i m . In case z m ∈ R , we have ˆ w m = ˆ z i m , thus ( ˆ w m · z i m ) ˆ w m = z i m . Consequently, z i (1) m = z i m -z i m = 0 . Thus, first projection step of INLP leads to complete removal of main-task feature z m when z m ∈ R . Also, from Lemma C.1, this projection step is non-invertible and hence the main-task feature cannot be recovered back in the subsequent projection step.

̸

Next, we will show the first projection step has no effect on the concept-causal features which we wanted to remove in the first place. From Eq. 76, we have:

$$z _ { p } ^ { i ( 1 ) } = ( I - \hat { w } _ { p } \hat { w } _ { p } ^ { T } ) z _ { p } ^ { i } - ( \hat { w } _ { m } \cdot z _ { m } ^ { i } ) \hat { w } _ { p }$$

$$= z _ { p } ^ { i } - 0 - 0 & & ( \text {since } w _ { p } = 0 )$$

Thus the first step of projection has no effect on the concept-causal feature we wanted to remove. In the next step of projection, if we again have w p = 0 , then this same case will repeat. Otherwise if Assm 3.3 still holds for main-task feature for the margin points of optimal probing classifier c ∗ (2) p ( z ) for this second step of projection, then we will have both w m = 0 and w p = 0 and the first case of this theorem will apply.

̸

̸

Second Claim. Now for proving the second statement, we will make use of the following lemma. The proof of the lemma is given below the proof of this theorem.

̸

Lemma C.2. After every projection step of INLP, the norm of every latent representation z i decreases, i.e., ∥ z i ( k ) ∥ &lt; ∥ z i ( k -1) ∥ for step k and k -1 , if (1) z i ( k -1) = 0 , (2) z i (0) ˆ w k = 0 and (3) the parameters of probing classifier in step ' k ' i.e ˆ w k don't lie in the space spanned by parameters of previous probing classifier, span( ˆ w 1 , . . . , ˆ w k -1 ).

̸

Next, we will show that starting from the first step, at every k th -step of projection either we will have z i ( k ) = 0 or the norm will decrease after projection. Once we reach a step when z i ( k ) = 0 , then after every subsequent projection we will have z i ( k +1) = 0 = ⇒ ∥ z i ( k +1) ∥ = 0 since:

$$z ^ { i ( k + 1 ) } = P _ { N ( w ^ { k } ) } z ^ { i ( k ) } = P _ { N ( w ^ { k } ) } 0 = 0$$

where P N ( w k ) is the projection matrix at step "k". Since ∥·∥ ≥ 0 and ∥ z i ( k ) ∥ is decreasing with every stey, thus with large number of z i ( ∞ ) → 0 .

̸

We will start with the first step of projection. In the second statement of this Theorem 3.2, we are given that z i (0) is not trivially zero in direction of w 0 i.e z i (0) w 0 = 0 (satisfying Assm(2) of above Lemma C.2). We are also given that z i (0) = 0 (satisfying Assm(1) of above lemma) and since this is the first step of projection Assm(3) of above Lemma C.2) is also satisfied. Thus from Lemma C.2, we have ∥ z i (1) ∥ &lt; ∥ z i (0) ∥ . Now, either z i (1) = 0 , which will imply that ∥ z i (1) ∥ = 0 and will remain 0 for all subsequent step (from Eq. 85). Otherwise if z i (1) = 0 , it satisfies the Assm(1) of Lemma C.2, for next step of projection. Since Assm (2) and (3) are already satisfied (from the assumption in the second claim of Theorem 3.2), again we will have ∥ z i (2) ∥ &lt; ∥ z i (1) ∥ and the same idea will repeat eventually making z i ( k ) = 0 at some step-k, thus completing our proof.

̸

̸

Proof of Lemma C.2. After ( k -1) -steps of INLP let the latent space representation z i be denoted as z i ( k -1) . Let ˆ w k be the parameters of classifier c p ( z k -1 ) trained to predict the concept label y p which we want to remove at step k . Then prior to the projection step in the k th iteration of the INLP, we can write z i ( k -1) as:

$$z _ { B } ^ { i ( k - 1 ) } = z _ { \hat { w } ^ { k } } ^ { i ( k - 1 ) } \hat { w } ^ { k } + z _ { N ( \hat { w } ^ { k } ) } ^ { i ( k - 1 ) } N ( \hat { w } ^ { k } )$$

where B = { ˆ w k , N ( ˆ w k ) } is the basis set and N ( ˆ w k ) is the null-space of ˆ w k . The parameter ˆ w k in this new basis is:

$$\hat { w } _ { B } ^ { k } = I _ { \hat { w } ^ { k } } \hat { w } ^ { k } + 0 N ( \hat { w } ^ { k } )$$

where I ˆ w k is identity matrix with dimension d ˆ w k × d ˆ w k . Now, in the new basis when we project the z k -1 to the null space of ˆ w i ( k ) we have:

$$z ^ { i ( k ) } = P _ { N ( \hat { w } ^ { k } ) } z ^ { i ( k - 1 ) }$$

$$z _ { B } ^ { i ( k ) } = \left ( I - \hat { w } _ { B } ^ { k } ( \hat { w } _ { B } ^ { k } ) ^ { T } \right ) z _ { B } ^ { i ( k - 1 ) }$$

$$= \left ( I - \begin{bmatrix} I _ { \hat { w } ^ { k } } \\ 0 \end{bmatrix} [ I _ { \hat { w } ^ { k } } & 0 ] \right ) \begin{bmatrix} z _ { \hat { w } ^ { k } } ^ { i ( k - 1 ) } \\ z _ { N ( \hat { w } ^ { k } ) } ^ { i ( k - 1 ) } \end{bmatrix} \\$$

$$= \begin{bmatrix} z _ { \hat { w } ^ { k } } ^ { i ( k - 1 ) } \\ z _ { N ( \hat { w } ^ { k } ) } ^ { i ( k - 1 ) } \end{bmatrix} - \begin{bmatrix} z _ { \hat { w } ^ { k } } ^ { i ( k - 1 ) } \\ 0 \end{bmatrix}$$

$$= \begin{bmatrix} 0 \\ z _ { N ( \hat { w } ^ { k } ) } ^ { i ( k - 1 ) } \end{bmatrix}$$

̸

Thus the norm of ∥ z i ( k ) ∥ = √ ∥ z i ( k -1) N ( ˆ w k ) ∥ +0 is less than ∥ z k -1 ∥ = √ ∥ z i ( k -1) ˆ w k ∥ 2 + ∥ z i ( k -1) N ( ˆ w k ) ∥ 2 if z i ( k -1) ˆ w k = 0 . Next we will show that z i ( k -1) ˆ w k cannot be zero. From assumption (2) in C.2 z i (0) w k = 0 i.e z i (0) w k is not trivially zero in the given latent representation z i (0) before any projection from INLP, thus z i ( k -1) ˆ w k is not trivially zero from beginning. Also, from Eq. 92, we observe that at any step ' k ' INLP removes the part of the representation from z i ( k -1) which is in the direction of ˆ w k i.e. z i ( k -1) ˆ w k . Consequently, a sequence of removal steps with parameters ˆ w 1 , . . . , ˆ w k -1 will remove the part of z which lies in the span( ˆ w 1 , . . . , ˆ w k -1 ). Thus z i ( k -1) ˆ w k = 0 if ˆ w k lies in the span of parameters of previous classifier i.e span( ˆ w 1 , . . . , ˆ w k -1 ) which violates the assumption (3) in Lemma C.2. Thus z i ( k -1) ˆ w k is neither trivially zero from the beginning nor it could have been removed in the previous steps of projection as long as the assumption in Lemma C.2 is satisfied, which completes our proof of the lemma.

̸

Remark. The following lemma from [32] tells us some of the sufficient conditions when the parameters of the probing classifier at the current iteration will not be same as the previous step.

Lemma C.3 (Lemma A.1 from [32]) . If the concept-probing classifier is being trained using SGD (stochastic gradient descent) and the loss function is convex, then parameters of the probing classifier at step k , w k , are orthogonal to parameters at step k -1 , w k -1 .

We conjecture that Lemma C.3 will be true for any loss since after k -1 steps of projection, the component of z in the direction of span( w 1 , . . . , w k -1 ) will be removed. Hence the concept-probing classifier at step k should be orthogonal to span( w 1 , . . . , w k -1 ) in order to have non-random guess accuracy on probing task.

## D Adversarial Removal: Setup and Proof

## D.1 Adversarial Setup

As described in §3.3, let h : X → Z be an encoder mapping the input x to latent representation z . The main task classifier c m : Z → Y m is applied on top of z to predict the main task label y m for every input x . As described in §3.3, the goal of an adversarial removal method, henceforth referred as AR, is to remove any undesired/sensitive/spurious concept from the latent representation z . Once the concept is removed from the latent representation, any (main-task) classifier using the latent representation Z will not be able to use it [15, 48, 12]. These methods jointly train the main-task classifier c m ( z ) and the probing classifier c p : Z → Y p . The probing classifier is adversarially trained to

predict the concept label y i p from latent representation z i . Hence, AR methods optimize the following two objectives simultaneously:

$$\arg \min _ { c _ { p } } L ( c _ { p } ( h ( x ) ) , y _ { p } )$$

$$a r g \min _ { h , c _ { m } } \left \{ L ( c _ { m } ( h ( x ) , y _ { m } ) - L ( c _ { p } ( h ( x ) , y _ { p } ) ) \Big \}$$

Here L ( · ) is a loss function which estimates the error given the ground truth y m /y p and corresponding prediction c m ( z ) /c p ( z ) . The above adversarial objective between the encoder and probing classifier is a min-max game. The encoder wants to learn a latent representation z s.t. it maximize the loss of probing classifier but at the same time probing classifier tries to minimize it's loss. The desired solution and simultaneously a valid equilibrium point of the above min-max objective is an encoder h such that it removes all the features from latent space that are useful for prediction of y p while keeping intact other features causally derived from the main task prediction. In practice, the optimization to above objective is performed using a gradient-reversal (GRL) layer ([15]). It introduces an additional layer g λ between the latent representation h ( z ) and the adversarial classifier c p ( z ) . The g λ layer acts as an identity layer (i.e., has no effect) during the forward pass but scales the gradient by ( -λ ) when back-propagating it during the backward pass. Thus resulting combined objective is:

$$a r g \min _ { h , c _ { m } , c _ { p } } \left \{ L ( c _ { m } ( h ( z ) ) , y _ { m } ) + L ( c _ { p } ( g _ { \lambda } ( h ( z ) ) ) , y _ { p } ) \right \}$$

Setup for theoretical result: As stated in Theorem 3.3, for our theoretical result showing the failure mode of adversarial removal, we assume that the encoder is divided into two sub-parts. The first encoder i.e h 1 : X → Z is frozen (non-trainable) and maps the input x i to intermediate latent representation z i which is frozen and disentangled (Assm 3.1). The second encoder h 2 : Z → ζ is a linear transformation mapping the intermediate latent representation z i to final latent representation ζ i and is trainable. On top of this final latent representation ζ i , we train the main task classifier c m ( ζ i ) and probing classifier c p ( ζ i ) . Thus the training objective from Eq. 93 and 94 becomes:

$$\arg \min _ { c _ { p } } L ( c _ { p } ( h _ { 2 } ( z ) ) , y _ { p } )$$

$$a r g \min _ { h _ { 2 } , c _ { m } } \left \{ L ( c _ { m } ( h _ { 2 } ( z ) , y _ { m } ) - L ( c _ { p } ( h _ { 2 } ( z ) , y _ { p } ) \Big \}$$

## D.2 Adversarial Proof

For a detailed discussion of adversarial training objective and specific setup for our theoretical result refer §D.1.

We formally state the new assumption made in the second statement of Theorem 3.3. This assumption imposes constraints on strength of correlation between main task label and concept-causal feature i.e it requires the conceptcausal feature to be more predictive of main task label than for probing task.

Assumption D.1 (Strength of Correlation) . Let ˆ w p ∈ R d p be the unit vector s.t. z p is linearly separable for conceptlabel y p (see Assm 3.2) and let h ∗ 2 ( z ) be the desired encoder which is successful in removing the concept-causal features z p from ζ . Then, concept-causal features z p M of any margin point z M of c m ( h ∗ 2 ( · )) is more predictive of the main task than concept-causal features z P p of any margin-point z P of c p ( h ∗ 2 ( · )) for probing task by a factor of | β | ∈ R where | β | is the norm of parameter of probing classifier c p ( h ∗ 2 ( · )) i.e y m ( ˆ w p · z p M ) &gt; | β | y p ( ˆ w p · z P p ) .

Theorem 3.3. Let the latent representation z satisfy Assm 3.1 and be frozen, h 2 ( z ) be a linear transformation over Z s.t. h 2 : Z → ζ , the main-task classifier be c m ( ζ ) = w c m · ζ , and the adversarial probing classifier be c p ( ζ ) = w c p · ζ . Let all the assumptions of Lemma B.1 be satisfied for main-classifier c m ( · ) when using z directly as input and Assm 3.2 be satisfied on z w.r.t. the adversarial task. Let h ∗ 2 ( z ) be the desired encoder which is successful in removing z p from ζ . Then there exists an undesired/incorrect encoder h α 2 ( z ) s.t. h α 2 ( z ) is dependent on z p and the main-task classifier c m ( h α 2 ( z )) has bigger margin than c m ( h ∗ 2 ( z )) and has,

1. Accuracy ( c p ( h α 2 ( z )) , y p ) = Accuracy ( c p ( h ∗ 2 ( z )) , y p ) ; when adversarial probing classifier c p ( · ) is trained using any learning objective like max-margin or cross-entropy loss. Thus, the undesired encoder h α 2 ( z ) is indistinguishable from desired encoder h ∗ 2 ( z ) in terms of adversarial task prediction accuracy but better for main-task in terms of max-margin objective.
2. L h 2 ( c m ( h α 2 ( z )) , c p ( h α 2 ( z )) ) &lt; L h 2 ( c m ( h ∗ 2 ( z )) , c p ( h ∗ 2 ( z )) ) ; when Assm 3.4 is satisfied and concept-causal features z p M of any margin point z M of c m ( h ∗ 2 ( z )) are more predictive of the main task label than z P p of any margin point z P of c p ( h ∗ 2 ( z )) is predictive for the probing label (Assm D.1). Thus, undesired encoder h α 2 ( z ) is preferable

over desired encoder h ∗ 2 ( z ) for both main and combined adversarial objective. Here L h 2 = L ( c m ( · )) -L ( c p ( · )) is the combined adversarial loss w.r.t. to h 2 and L ( c ( · )) is the max-margin loss for a classifier 'c' (see §D.1).

Proof of Theorem 3.3. Let the main classifier be of the form c m ( ζ ) = w c m · ζ where w c m and ζ are d ζ dimensional vectors. Since both parameters w c m and ζ are learnable, for ease of exposition we constrain w c m to be [1 , 0 , . . . , 0] . This constraint on w c m is w.l.o.g. since w c m makes the prediction for main-task by projecting ζ into one specific direction to get a scalar ( w c m · ζ ) . We constrain that direction to be the first dimension of ζ . Since the encoder h 2 ( z ) : Z → ζ is trainable it could learn to encode the scalar ( w c m · ζ ) in the first dimension of ζ . Thus, effectively a single dimension of the representation ζ encodes the main-task information. As a result, the main classifier is effectively of the form c m ( ζ ) = w (0) c m × ζ (0) = 1 × ζ (0) where w (0) c m and ζ (0) are the first elements of w c m and ζ respectively and w (0) c m = 1 . We can now write the goal of the adversarial method as removing the information of z p from ζ (0) because the other dimensions are not used by the main classifier. Also, the adversarial probing classifier can be written effectively as c p ( ζ ) = β × ζ (0) where β ∈ R is a trainable parameter. Since both the main and adversarial classifier are using only ζ (0) , the second encoder, with a slight abuse of notation, can be simplified as ζ := ζ (0) := h 2 ( z ) = w m · z m + w p · z p , where ζ ∈ R and w m and w p are the weights that determine the first dimension of ζ . Also, let the desired (correct) second encoder which is successful in removing the concept-causal feature z p from ζ be ζ ∗ := ζ ∗ (0) := h ∗ 2 ( z ) = w ∗ m · z m . Thus using Eq. 96 and 97, our overall objective for adversarial removal method becomes:

$$\arg \min _ { \beta } L ( c _ { p } ( h _ { 2 } ( z ) ) , y _ { p } )$$

$$\ a r g \min _ { h _ { 2 } } \left \{ L ( c _ { m } ( h _ { 2 } ( z ) , y _ { m } ) - L ( c _ { p } ( h _ { 2 } ( z ) , y _ { p } ) ) \right \}$$

1. First claim. The ideal main classifier with desired encoder can be written as, c m ( ζ ∗ ) = 1 × h ∗ 2 ( z ) = w ∗ m · z m . Therefore, it can be trained using the MM-Denominator formulation of the max-margin objective and would satisfy the constraint in Eq. 3:

$$m ( c _ { m } ( \zeta ^ { * i } ) ) = m ( h _ { 2 } ^ { * } ( z ^ { i } ) ) = y _ { m } ^ { i } \cdot h _ { 2 } ^ { * } ( z ^ { i } ) \geq 1$$

for all the points x i with latent representation z i and m ( · ) is the numerator of the distance of point from the decision boundary of classifier (Eq. 1).

However, the main task classifier which does not use the desired encoder is of the form, c m ( ζ ) = 1 × h α 2 ( z ) = w m · z m + w p · z p . Since this main task classifier is trained using max-margin objective by MM-Denominator formulation, it would satisfy the constraint in Eq. 3:

$$m ( c _ { m } ( \zeta ^ { i } ) ) = m ( h _ { 2 } ^ { \alpha } ( z ^ { i } ) ) = y _ { m } ^ { i } \cdot h _ { 2 } ^ { \alpha } ( z ^ { i } ) \geq 1$$

Since in our case, main task classifier is the same as the encoder i.e c m ( ζ ) = 1 × h α 2 ( z ) = w m · z m + w p · z p , and the latent representation z satisfies the Assm 3.1, B.1 and B.2, from Lemma B.1 the main-task classifier is spurious-using i.e z p = 0 . Hence there exists an undesired/incorrect encoder h α 2 ( z ) such that the main classifier c m ( ζ ) = h α 2 ( z ) has bigger margin than c m ( ζ ∗ ) = h ∗ 2 ( z ) .

̸

Next, we show that the accuracy of the adversarial classifier remains the same irrespective of whether the desired ( h ∗ 2 ( z ) ) or undesired encoder h α 2 ( z ) is used. The accuracy of the adversarial classifier c p ( ζ ) = β × ζ , using the desired/correct encoder ζ = h ∗ 2 ( z ) is given by:

$$\Omega ( y ) = \frac { \sum _ { i = 1 } ^ { N } 1 \left ( s i g n ( \beta \cdot h _ { 2 } ^ { * } ( z ^ { i } ) ) = & y _ { p } ^ { i } \right ) } { N } \\ \intertext { i s y i n t e f u r a c y ( c _ { p } ( \zeta ^ { * } ) , y _ { p } ) = \frac { \sum _ { i = 1 } ^ { N } 1 \left ( s i g n ( \beta \cdot h _ { 2 } ^ { * } ( z ^ { i } ) ) = & y _ { p } ^ { i } \right ) } { N } } { 1 }$$

where 1 ( · ) is an indicator function which takes the value 1 if the argument is true otherwise 0 , and sign ( γ ) = +1 if γ ≥ 0 and -1 otherwise. Combining Eq. 100 and Eq. 101, since y i m ∈ {-1 , 1 } , we see that whenever h α 2 ( z i ) &gt; 1 we also have h ∗ 2 ( z i ) &gt; 1 and similarly whenever h α 2 ( z i ) &lt; -1 , we have h ∗ 2 ( z i ) &lt; -1 . Thus,

$$h _ { 2 } ^ { \alpha } ( z ) \cdot h _ { 2 } ^ { * } ( z ) > 0$$

From Eq. 103, h ∗ 2 ( z i ) and h α 2 ( z i ) has the same sign for every input z i = ⇒ sign ( β · h ∗ 2 ( z i )) = sign ( β · h α 2 ( z i )) . Thus we can replace h ∗ 2 ( z i ) with h α 2 ( z i ) in the above equation and we have:

$$\ p r a c { y ( c _ { p } ( \zeta ^ { * } ) , y _ { p } ) } & = \frac { \sum _ { i = 1 } ^ { N } 1 \left ( s i g n \left ( \beta \cdot h _ { 2 } ^ { \alpha } ( z ^ { i } ) \right ) = y _ { p } ^ { i } \right ) } { N } \\ \ p r a c { u r a c { y ( c _ { p } ( h _ { 2 } ^ { * } ( z ^ { i } ) ) , y _ { p } ) } } & = A c u r a c { y ( c _ { p } ( h _ { 2 } ^ { \alpha } ( z ^ { i } ) ) , y _ { p } ) } \\ \ p r a c { i n t o r p r a c { y } }$$

thus completing the first part our proof.

2. Second claim. Since we are training both the main task and the probing classifier with a max-margin objective (see MM-Numerator version at Eq. 5), we can effectively write the adversarial objective (from 98 and 99) as:

$$\ a r g \max _ { \beta } ( P ( \beta ) ) \colon = \ a r g \max _ { \beta } \left \{ \min _ { z ^ { * } } m _ { c _ { p } } ( h _ { 2 } ( z ^ { i } ) ) \right \}$$

$$\arg \max _ { h _ { 2 } } ( E ( h _ { 2 } ) ) \coloneqq \arg \max _ { h _ { 2 } } \left \{ \min _ { z ^ { i } } m _ { c _ { m } } ( h _ { 2 } ( z ^ { i } ) ) - \min _ { z ^ { i } } m _ { c _ { p } } ( h _ { 2 } ( z ^ { i } ) ) \right \}$$

where m c m ( h 2 ( z )) and m c p ( h 2 ( z )) are the numerator of margin of a point (Eq. 1). Next, our goal is to show that the desired encoder h ∗ 2 is not an equilibrium point of the above adversarial objective. To do so, we will create an undesired/incorrect encoder h α 2 ( z ) by perturbing h ∗ 2 by small amount and showing that the combined encoder objective E ( h α 2 ) &gt; E ( h ∗ 2 ) (Eq. 105) irrespective of choice of β chosen by the probing objective P ( β ) (Eq. 104).

Construction of the undesired/incorrect encoder. We have h ∗ 2 ( z ) = ∥ w m ∥ ( ˆ w ∗ m · z m ) where ˆ w ∗ m ∈ R d m is a unit vector. We will perturb this desired encoder by parameterizing with α ∈ [0 , 1) s.t.:

$$h _ { 2 } ^ { \alpha } ( z ) = \alpha \| w _ { m } ^ { * } \| ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| ( \hat { \epsilon } _ { p } \cdot z _ { p } )$$

where ˆ ϵ p ∈ R d p is a unit vector. The clean main-task classifier is defined as c ∗ m ( h ∗ 2 ( z )) = h ∗ 2 ( z ) . The main-task classifier c m when using the incorrect encoder takes form c m ( h α 2 ( z )) = h α 2 ( z ) . As stated in the theorem statement, all the assumptions of Lemma B.1 are satisfied. Since Assm B.2 (one of the assumptions of Lemma B.1) are satisfied, there exists a unit-vector in R d p such that concept-causal features of margin points of the main task classifier using encoder h ∗ 2 are linearly separable w.r.t main-task label. Let ˆ ϵ p in our constructed undesired encoder h α 2 (Eq. 106) be set to that unit vector such that:

$$y _ { m } ^ { M } \cdot ( \hat { \epsilon } _ { p } \cdot z _ { p } ^ { M } ) > 0$$

where z p M is the concept-causal feature of margin point z M of the main-task classifier when using encoder h ∗ 2 . Now since all the assumption of Lemma B.1 is satisfied, the margin of main-task classifier when using undesired encoder h α 2 ( z ) is bigger than when desired encoder h ∗ 2 is used for some α ∈ ( α 1 lb , 1) . Consequently, we have:

$$m _ { c _ { m } } ( h _ { 2 } ^ { \alpha } ( z ^ { M } ) ) > m _ { c _ { m } } ( h _ { 2 } ^ { * } ( z ^ { M } ) )$$

where z M is the margin point of c m ( h ∗ 2 ) . Since Assm 3.2 is satisfied, we have a fully predictive concept-causal feature z p for prediction of adversarial label y p such that for some unit vector ˆ w p ∈ R d p we have:

$$y _ { p } ^ { i } ( \hat { w } _ { p } \cdot z _ { p } ^ { i } ) > 0 \ \forall ( z ^ { i } , y _ { p } ^ { i } )$$

Next, since Assm 3.4 is also satisfied for this second part of theorem, we have y i p = y i m for every margin point of the desired/correct main-task classifier using the desired/correct encoder h ∗ 2 ( z ) . Thus we can assign ˆ ϵ p := ˆ w p which satisfies the inequality in Eq. 107. Hence, our incorrect encoder h α 2 ( z ) take the following form:

$$h _ { 2 } ^ { \alpha } ( z ) = \alpha \| w _ { m } ^ { * } \| ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } )$$

Note that when α = 1 , we recover back the correct encoder h ∗ 2 . Thus to perturb the h ∗ 2 , we set α close to but less than 1.

Showing h ∗ 2 is not the equilibrium point. From Eq. 105, we want to show that for some α ∈ [0 , 1) s.t. α → 1 ( α close to but less than 1 ), the undesired encoder h α 2 has bigger combined objective than desired encoder h ∗ 2 . Since the combined adversarial objective for encoder h 2 ( z ) ( E ( h 2 ( z )) in Eq. 105) is evaluated on the margin points of main-task and probing task classifier. We use the following lemma to show that for small perturbation of the optimal encoder ( α → 1 ), the margin point of main-task classifier and probing classifier when using perturbed encoder h α 2 remains same or is a subset of margin points when using desired encoder h ∗ 2 . The proof of the lemma below is given after the proof of the current theorem.

Lemma D.1. There exist an α 2 lb ∈ [0 , 1) s.t. when α &gt; α 2 lb we have: (i) margin points of probing classifier when using perturbed encoder h α 2 is same or is a subset of margin points when using desired encoder h ∗ 2 . (ii) margin points of main-task classifier when using perturbed encoder h α 2 is same or is a subset of margin points when using desired encoder h ∗ 2 .

Let z M ∗ be one of the margin point of main-task classifier c m and z P ∗ be one of the margin point of probing classifier c p when using the correct encoder h ∗ 2 . Let z M α be one of the margin point of main-task classifier c m and z P α be one of the margin point of probing classifier c p when using the perturbed encoder h α 2 . Thus we want to show that for all ( z M ∗ , z P ∗ , z M α , z P α ) tuple, there exists some α close to but less than 1 s.t. we have:

$$E ( h _ { 2 } ^ { \alpha } ) > E ( h _ { 2 } ^ { * } )$$

$$m _ { c _ { m } } ( h _ { 2 } ^ { \alpha } ( z ^ { M _ { \alpha } } ) ) - m _ { c _ { p } } ( h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) ) > m _ { c _ { m } } ( h _ { 2 } ^ { * } ( z ^ { M _ { * } } ) ) - m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) )$$

$$m _ { c _ { m } } ( h _ { 2 } ^ { \alpha } ( z ^ { M _ { \alpha } } ) ) - m _ { c _ { m } } ( h _ { 2 } ^ { * } ( z ^ { M _ { * } } ) ) > m _ { c _ { p } } ( h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) ) - m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) )$$

For β &lt; 0 . From Lemma D.1, both z P α and z P ∗ are the margin point of probing classifier when using the desired encoder h ∗ 2 . Thus we have:

$$m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ) ^ { P _ { \alpha } } ) = m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) )$$

$$y _ { p } ^ { P _ { \alpha } } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) = y _ { p } ^ { P _ { * } } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { * } } )$$

Also, since α ∈ [0 , 1) , from above equation we have:

$$y _ { p } ^ { P _ { \alpha } } \beta \alpha ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) < y _ { p } ^ { P _ { * } } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { * } } )$$

From Eq. 109 we have y P α p ( ˆ w p · z P α p ) &gt; 0 . Since β &lt; 0 and α ∈ [0 , 1) , we have √ 1 -α 2 β ∥ w ∗ m ∥ y P α p ( ˆ w p · z P α p ) &lt; 0 . Adding this to LHS of the above equation we get:

$$y _ { p } ^ { P _ { \circ } } \beta \alpha ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \circ } } ) + \sqrt { 1 - \alpha ^ { 2 } } \beta \| w _ { m } ^ { * } \| y _ { p } ^ { P _ { \circ } } \left ( \hat { w } _ { p } \cdot z _ { p } ^ { P _ { \circ } } \right ) < y _ { p } ^ { * } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \circ } } )$$

$$y _ { p } ^ { P _ { \alpha } } \beta \left \{ \alpha ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } ^ { P _ { \alpha } } ) \right \} < y _ { p } ^ { P _ { * } } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { * } } )$$

$$y _ { p } ^ { P _ { \alpha } } \beta h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) < y _ { p } ^ { P _ { * } } \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { * } } )$$

$$m _ { c _ { p } } ( h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) ) < m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) )$$

From Eq. 120 the RHS of Eq. 113 is less than zero. Also, from Eq. 108, for α ∈ ( α 1 lb , 1) we have m c m ( h α 2 ( z M α )) -m c m ( h ∗ 2 ( z M ∗ )) &gt; 0 where value of α 1 lb is given by Lemma B.1. Thus the LHS of Eq. 113 is greater than 0 . Thus the inequality in 113 is always satisfied when β &lt; 0 and α ∈ ( max { α 1 lb , α 2 lb } , 1) . The constraint α &gt; α 1 lb is enforced by Lemma B.1 when constructing the perturbed encoder and α &gt; α 2 lb is enforced by Lemma D.1 which ensures z P α is also a margin point of probing classifier when using desired encoder h ∗ 2 . Hence, we have shown that when β &lt; 0 , h ∗ 2 is not the equilibrium point since there exist a perturbed undesired encoder h α 2 such that the combined encoder objective is greater in Eq. 105 and consequently the optimizer will try to move away from/change h ∗ 2 .

For β &gt; 0 . Next we have to show that there exist α ∈ [0 , 1) s.t. when α → 1 we have Eq. 113 satisfied. Thus we solve for allowed values of α :

$$\left \{ m _ { c _ { m } } ( h _ { 2 } ^ { \alpha } ( z ^ { M _ { \alpha } } ) ) - m _ { c _ { m } } ( h _ { 2 } ^ { * } ( z ^ { M _ { * } } ) ) \right \} > \left \{ m _ { c _ { p } } ( h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) ) - m _ { c _ { p } } ( h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) ) \right \}$$

$$\left \{ y _ { m } h _ { 2 } ^ { \alpha } ( z ^ { M _ { \alpha } } ) - y _ { m } h _ { 2 } ^ { * } ( z ^ { M _ { * } } ) \right \} > \left \{ y _ { p } ( \beta \cdot h _ { 2 } ^ { \alpha } ( z ^ { P _ { \alpha } } ) ) - y _ { p } ( \beta \cdot h _ { 2 } ^ { * } ( z ^ { P _ { * } } ) ) \right \}$$

From the second statement from Lemma D.1, z M α and z M ∗ both are margin point of main-task classifier using the desired encoder h ∗ 2 . Thus we have m c m ( h ∗ 2 ( z M α )) = m c m ( h ∗ 2 ( z M ∗ )) = ⇒ y M α m h ∗ 2 ( z M α ) = y M ∗ m h ∗ 2 ( z M ∗ ) = ⇒ y M α m ( w ∗ m · z M α m ) = y M ∗ m ( w ∗ m · z M ∗ m ) . Substituting this observation in LHS of Eq. 121 we get m c m ( h α 2 ( z M α )) -m c m ( h ∗ 2 ( z M ∗ )) =

$$= y _ { m } ^ { M _ { \alpha } } \left \{ \alpha ( w _ { m } ^ { * } \cdot z _ { m } ^ { M _ { \alpha } } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } ^ { M _ { \alpha } } ) \right \} - y _ { m } ^ { M _ { * } } \left \{ ( w _ { m } ^ { * } \cdot z _ { m } ^ { M _ { * } } ) \right \}$$

$$= y _ { m } ^ { M _ { \alpha } } \left \{ \alpha ( w _ { m } ^ { * } \cdot z _ { m } ^ { M _ { \alpha } } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } ^ { M _ { \alpha } } ) \right \} - y _ { m } ^ { M _ { \alpha } } \left \{ ( w _ { m } ^ { * } \cdot z _ { m } ^ { M _ { \alpha } } ) \right \}$$

$$= ( \alpha - 1 ) y _ { m } ^ { M _ { \alpha } } \left \{ \| w _ { m } ^ { * } \| ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { M _ { \alpha } } ) \right \} + \sqrt { 1 - \alpha ^ { \frac { M _ { \alpha } } { 2 } } y _ { m } ^ { M _ { \alpha } } } \left \{ \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } ^ { M _ { \alpha } } ) \right \}$$

$$= ( \alpha - 1 ) y _ { m } ^ { M } \left \{ \| w _ { m } ^ { * } \| ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { M } ) \right \} + \sqrt { 1 - \alpha ^ { 2 } } y _ { m } ^ { M } \left \{ \| w _ { m } ^ { * } \| ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) \right \}$$

where for ease of exposition we have defined M := M α . Now again for RHS of Eq. 121, from Lemma D.1, z P α and z P ∗ both are margin point of probing classifier using the desired encoder h ∗ 2 . Thus we have m c p ( h ∗ 2 ( z P α )) = m c p ( h ∗ 2 ( z P ∗ )) = ⇒ y P α p βh ∗ 2 ( z P α ) = y P ∗ p βh ∗ 2 ( z P ∗ ) = ⇒ y P α p ( w ∗ m · z P α m ) = y P ∗ p ( w ∗ m · z P ∗ m ) . Substituting this observation in RHS of Eq. 121 we get m c p ( h α 2 ( z P α )) -m c p ( h ∗ 2 ( z P ∗ )) =

$$= y _ { p } ^ { P _ { \alpha } } \left \{ \alpha \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P _ { \alpha } } ) \right \} - y _ { p } ^ { P _ { * } } \left \{ \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { * } } ) \right \}$$

$$= y _ { p } ^ { P _ { \alpha } } \left \{ \alpha \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P _ { \alpha } } ) \right \} - y _ { p } ^ { P _ { \alpha } } \left \{ \beta ( w _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) \right \}$$

$$= ( \alpha - 1 ) y _ { p } ^ { P _ { \infty } } \left \{ \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { P _ { \alpha } } ) \right \} + \sqrt { 1 - \alpha ^ { 2 } } y _ { p } ^ { P _ { \infty } } \left \{ \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P _ { \alpha } } ) \right \}$$

$$= ( \alpha - 1 ) y _ { p } ^ { P } \left \{ \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { P } ) \right \} + \sqrt { 1 - \alpha ^ { 2 } } y _ { p } ^ { P } \left \{ \| w _ { m } ^ { * } \| \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P } ) \right \}$$

where for ease of exposition we have defined P := P α . Substituting RHS (Eq. 130) and LHS (Eq. 126) back in Eq. 121 and rearranging we get:

$$\sqrt { 1 - \alpha ^ { 2 } } \left \{ y _ { m } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P } ) \right \} > ( 1 - \alpha ) \left \{ y _ { m } ^ { M } ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { P } ) \right \}$$

Now, since Assm B.1 is satisfied, the main task feature z M m is linearly separable w.r.t main-task label y M m . Thus we have y M m ( ˆ w ∗ m · z M m ) &gt; 0 .

Case 1: Main-task feature is not fully predictive of probing label ( ∃ z s.t. y p ( ˆ w ∗ m · z m ) &lt; 0 ). Since main-task feature is not fully predictive of the probing label y p , there will be some points which will be misclassified (will be on the opposite side of decision boundary) when probing classifier uses desired encoder c p ( h ∗ 2 ( z )) = βh ∗ 2 ( z ) = w ∗ m · z m . Thus margin for those points will be negative and one of them will be the margin point z P of the probing classifier. That is, m c p ( h ∗ 2 ( z P )) = y P p β ( ˆ w ∗ m · z P m ) &lt; 0 . Then the term ( y M m ( ˆ w ∗ m · z M m ) -y P p β ( ˆ w ∗ m · z P m ) ) &gt; 0 in the above Eq. 131. Hence, rewriting the above equation we have:

$$\frac { \left \{ y _ { m } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P } ) \right \} } { \left \{ y _ { m } ^ { M } ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { P } ) \right \} } > \frac { 1 - \alpha } { \sqrt { 1 - \alpha ^ { 2 } } }$$

Next, from Lemma D.1 both z M := z M α and z P := z P α are also the margin point of main-task and probing classifier respectively when the classifiers use the desired encoder h α 2 . Then, since Assm D.1 is satisfied the numerator in LHS of above equation ( y M m ( ˆ w p · z p M ) -y P p β ( ˆ w p · z P p ) ) &gt; 0 . Thus, the whole LHS in the above equation is greater than zero. Denoting the LHS by γ ( z M , z P ) gives us:

$$\gamma ( z ^ { M } , z ^ { P } ) & > \frac { 1 - \alpha } { \sqrt { 1 - \alpha ^ { 2 } } } & ( 1 3 3 )$$

$$\gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) > \frac { ( 1 - \alpha ) ( 1 - \alpha ) } { ( 1 - \alpha ) ( 1 + \alpha ) }$$

$$\gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) + \alpha \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) > 1 - \alpha$$

$$\left ( 1 + \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) \right ) \alpha > 1 - \gamma ^ { 2 } ( z ^ { M } , z ^ { P } )$$

$$\alpha > \frac { 1 - \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) } { 1 + \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) } = \alpha _ { l b } ^ { 3 } ( z ^ { M } , z ^ { P } )$$

Since γ 2 ( z M , z P ) &gt; 0 , α 3 lb ( z M , z P )) &lt; 1 . Let α 3 lb = max ( z M , z P ) ( α 3 lb ( z M , z P ))) which is &lt; 1 gives us the tight lower-bound on α such that Eq. 113 is satisfied for any pair of margin point z M and z P .

Case 2: Main-task is fully predictive of probing label. ( ∀ z , y p ( ˆ w ∗ m · z m ) &gt; 0 ). Since Assm B.1 (from Lemma B.1) is satisfied, we have that main-task features are fully predictive of main-task label i.e y m ( ˆ w ∗ m · z m ) &gt; 0 for all z . Thus for this case y m ( ˆ w ∗ m · z m ) &gt; 0 and y p ( ˆ w ∗ m · z m ) &gt; 0 = ⇒ y m = y p for all z . Also, for this case, there will be no misclassified points for the probing classifier when using the desired encoder h ∗ 2 . Thus the margin point for both the main and the probing classifier is same i.e z M = z P . Since Assm D.1 is satisfied, y m = y p for all z , y p ( ˆ w p · z ) &gt; 0 for all z from Assm 3.2 and z P = z M we have:

$$y _ { m } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) > y _ { p } ^ { P } \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P } ) \quad ( A s s m \ D . 1 )$$

$$1 \cdot ( \underline { y _ { p } ^ { P } } ( \hat { w } _ { \widehat { p } } \widehat { z } _ { p } ^ { P } ) ) > \beta ( y _ { p } ^ { P } ( \hat { w } _ { \widehat { p } } \widehat { z } _ { p } ^ { P } ) )$$

$$\beta < 1$$

Thus, in this case the RHS in Eq. 131, could be simplified to : y M m ( ˆ w ∗ m · z M m ) -y P p β ( ˆ w ∗ m · z P m ) = y M m ( ˆ w ∗ m · z M m ) -βy M m ( ˆ w ∗ m · z M m ) = (1 -β ) y M m ( ˆ w ∗ m · z M m ) &gt; 0 since 0 &lt; β &lt; 1 from above Eq. 140 and y m ( ˆ w ∗ m · z M m ) &gt; 0 from Assm B.1. Thus we can rewrite Eq. 131 as:

$$\frac { \left \{ y _ { m } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { P } ) \right \} } { \left \{ y _ { m } ^ { M } ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { M } ) - y _ { p } ^ { P } \beta ( \hat { w } _ { m } ^ { * } \cdot z _ { m } ^ { P } ) \right \} } > \frac { 1 - \alpha } { \sqrt { 1 - \alpha ^ { 2 } } }$$

Again, from Lemma D.1 both z M := z M α and z P := z P α are also the margin point of main-task and probing classifier respectively when the classifiers use the desired encoder h α 2 . Thus from Assm D.1, we have numerator of LHS in above equation greater than 0, thus we can follow the same steps from Eq. 133 to 137 to get the α 3 lb for this case.

So far, we have three lower bounds on α needed for this proof, so lets define α lb = max { α 1 lb , α 2 lb , α 3 lb } , where α 1 lb is enforced by Lemma B.1 on undesired encoder h α 2 construction, α 2 lb is enforced by Lemma D.1 and α 3 lb is enforced by Eq. 113. Thus, when α ∈ ( α lb , 1] we have a bigger combined objective (Eq. 105) for h α 2 than h ∗ 2 . Thus, we can always perturb the desired encoder h α 2 by choosing α ∈ ( α lb , 1] close to but less than 1 to create h α 2 which will have better combined encoder objective. Hence any optimizer will prefer to change the desired encoder h ∗ 2 and it is not an equilibrium solution to the overall adversarial objective.

Proof of Lemma D.1. First, we will prove the statement for the probing classifier. Let z M be one of the margin points of the probing classifier when using the desired encoder h ∗ 2 and let z R be any other (non-margin) points. Then we have to show that the margin-point of the probing classifier when using perturbed encoder h α 2 cannot be z R . This will imply that the margin points for probing classifier when using h α 2 has to be the same or a subset of margin points when using h ∗ 2 . Since norm of parameters of both c p ( h α 2 ( z )) = βh α 2 ( z ) and c p ( h ∗ 2 ( z )) = βh ∗ 2 ( z ) is the same and margin-point of a classifier is the point which have minimum margin, we have to show that m c p ( h α 2 ) ( z R ) &gt; m c p ( h α 2 ) ( z M ) for some α ∈ [0 , 1) . We have:

$$m _ { c _ { p } ( h _ { 2 } ^ { \alpha } ) } ( z ) = \alpha y _ { p } \left \{ \beta ( w _ { m } ^ { * } \cdot z _ { m } ) \right \} + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { p } \left \{ \beta ( \hat { w } _ { p } \cdot z _ { p } ) \right \}$$

$$= \alpha m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { p } \left \{ \beta ( \hat { w } _ { p } \cdot z _ { p } ) \right \}$$

Thus we have to find an α ∈ [0 , 1) s.t.:

$$\alpha m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { R } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { p } ^ { R } \left \{ \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { R } ) \right \} & > \\ \alpha m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { M } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { p } ^ { M } \left \{ \beta ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) \right \}$$

Rearranging we get:

$$\alpha \left \{ m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { R } ) - m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { M } ) \right \} > \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| \beta \left \{ y _ { p } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { p } ^ { R } ( \hat { w } _ { p } \cdot z _ { p } ^ { R } ) \right \}$$

Since z M is the margin point of the probing classifier when using h ∗ 2 , we have m c p ( h ∗ 2 ) ( z R ) &gt; m c p ( h ∗ 2 ) ( z M ) . Now, if β { y p M ( ˆ w p · z p M ) -y R p ( ˆ w p · z R p ) } ≤ 0 , then above equation is trivially satisfied for all values of α ∈ (0 , 1) , since RHS of above equation is greater than 0 and LHS is less than 0. For the case when β { y p M ( ˆ w p · z p M ) -y R p ( ˆ w p · z R p ) } &gt; 0 we need:

$$\frac { \alpha } { \sqrt { 1 - \alpha ^ { 2 } } } & > \frac { \| w _ { m } ^ { * } \| \beta \left \{ y _ { p } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { p } ^ { R } ( \hat { w } _ { p } \cdot z _ { p } ^ { R } ) \right \} } { \left \{ m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { R } ) - m _ { c _ { p } ( h _ { 2 } ^ { * } ) } ( z ^ { M } ) \right \} } \div \gamma ( z ^ { M } , z ^ { P } ) > 0 \\$$

$$\frac { \alpha ^ { 2 } } { 1 - \alpha ^ { 2 } } & > \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) \\$$

$$\alpha ^ { 2 } ( 1 + \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) ) > \gamma ^ { 2 } ( z ^ { M } , z ^ { P } )$$

$$\alpha > \sqrt { \frac { \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) } { 1 + \gamma ^ { 2 } ( z ^ { M } , z ^ { P } ) } } \div \alpha _ { l b } ^ { p } ( z ^ { M } , z ^ { P } )$$

Since we have γ &gt; 0 = ⇒ α p lb ( z M , z P ) &lt; 1 . Lets define α p lb := max ( z M , z P ) ( α p lb ( z M , z P )) &lt; 1 , which gives the tightest lower bound on α s.t. when α ∈ ( α p lb , 1) , the margin point of the probing classifier when using the perturbed encoder is same or is a subset of margin point when using desired encoder h ∗ 2 . This completes the first part of the proof.

Next, we prove the second part of this lemma for the main-task classifier. Let z M be one of the margin points of the main-task classifier when using the desired encoder h ∗ 2 and let z R be any other (non-margin) point. Then we have to show that the margin-point of the main-task classifier when using perturbed encoder h α 2 cannot be z R . Since norm of parameter of both c m ( h α 2 ( z )) = h α 2 ( z ) and c m ( h ∗ 2 ( z )) = h ∗ 2 ( z ) is same and margin-point of a classifier is the point which have minimum margin, we have to show that m c m ( h α 2 ) ( z R ) &gt; m c m ( h α 2 ) ( z M ) for some α ∈ [0 , 1) . We have:

$$m _ { c _ { m } ( h _ { 2 } ^ { \alpha } ) } ( z ) = \alpha y _ { m } \Big \{ ( w _ { m } ^ { * } \cdot z _ { m } ) \Big \} + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { m } \Big \{ ( \hat { w } _ { p } \cdot z _ { p } ) \Big \}$$

$$= \alpha m _ { c _ { m } ( h _ { 2 } ^ { * } ) } ( z ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { m } \left \{ ( \hat { w } _ { p } \cdot z _ { p } ) \right \}$$

Thus we have find an α s.t.:

$$\text {Thus we have find an $\alpha$ s.t.:} \\ \alpha m _ { c _ { m } ( h _ { 5 } ) } ( z ^ { R } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { m } ^ { R } \left \{ ( \hat { w } _ { p } \cdot z _ { p } ^ { R } ) \right \} & > \\ \alpha m _ { c _ { m } ( h _ { 5 } ) } ( z ^ { M } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { m } ^ { M } \left \{ ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) \right \} \\ \text {Rearranging we get:} \\ \left \{ \begin{array} { c } \alpha m _ { c _ { m } ( h _ { 5 } ) } ( z ^ { M } ) + \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| y _ { m } ^ { M } \left \{ ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) \right \} \\ \end{array} \right \} \\ \text {Rearranging we get:}$$

$$\alpha \left \{ m _ { c _ { m } ( h _ { 2 } ^ { * } ) } ( z ^ { R } ) - m _ { c _ { m } ( h _ { 2 } ^ { * } ) } ( z ^ { M } ) \right \} > \sqrt { 1 - \alpha ^ { 2 } } \| w _ { m } ^ { * } \| \left \{ y _ { m } ^ { M } ( \hat { w } _ { p } \cdot z _ { p } ^ { M } ) - y _ { m } ^ { R } ( \hat { w } _ { p } \cdot z _ { p } ^ { R } ) \right \}$$

Since z M is the margin point of the probing classifier when using h ∗ 2 , we have m c m ( h ∗ 2 ) ( z R ) &gt; m c m ( h ∗ 2 ) ( z M ) . We notice that, apart from y p being set to y m and β being set to 1 , the above equation is identical to Eq. 144. Since our argument (from Eq. 144 to 148) to derive the allowed value of α doesn't depend on y p and β , we could follow the same argument to get a lower bound α m lb s.t. the main-task classifier has the same or subset of the margin points when using the perturbed encoder as it has when using the desired encoder.

Let us define α 2 lb = max { α p lb , α m lb } . Thus when α ∈ ( α 2 lb , 1) , both the statements of this lemma are satisfied thus completing our proof.

## E Experimental Setup

## E.1 Dataset Description

As described in §4, we demonstrate the failure of Null-Space Removal (§4.2) and Adversarial Removal (§4.3) in removing the undesired concept from the latent representation on three real-world datasets: MultiNLI [46], TwitterPAN16 [31] and Twitter-AAE [6]; and a synthetic dataset, Synthetic-Text. The detailed generation and evaluation strategies for each dataset are given below.

MultiNLI Dataset. In the MultiNLI dataset, given two sentences-premise and hypothesis-the main task is to predict whether the hypothesis entails , contradicts or is neutral to the premise. As described in §4, we simplify it to a binary task of predicting whether a hypothesis contradicts the premise. The binary main-task label, y m = 1 when a given hypothesis contradicts the premise otherwise it is -1. That is, we relabel the MNLI dataset by assigning label y m = 1 to examples with contradiction labels and y m = -1 to the example with neutral or entailment label. It has been reported that the contradiction label is spuriously correlated with the negation words like nobody, no, never and nothing [16]. Thus, we created a 'negation' concept denoting the presence of these words in the hypothesis of a given (hypothesis, premise) pair. The concept-label y p = 1 when the negation concept is present in the hypothesis otherwise it is -1 .

The standard MultiNLI dataset 1 has approximately 90% of data points in the training set, 5% as publicly available development set and the rest of 5% in a separate held-out validation set accessible through online competition leaderboard not accessible to the public. Thus, we create our own train and test split by subsampling 10 k examples from the initial training set, converting it into binary contradiction vs. non-contradiction labels, labeling the negation-concept label, and splitting them into 80-20 train and test split. For pre-training a clean classifier that does not use the spuriousconcept, we create a special training set following the method described in §E.2. For evaluating the robustness of both null-space and adversarial removal methods, we create multiple datasets with different predictive-correlation as described in §E.3 .

Twitter-PAN16 Dataset. In Twitter-PAN16 dataset [31], following [12], given a tweet, the main task is to predict whether it contains a mention of another user or not. The dataset contains manually annotated binary Gender information (i.e Male or Female) of 436 Twitter users with at least 1k tweets each. The Gender annotation was done by assessing the name and photograph of the LinkedIn profile of each user [12]. The unclear cases were discarded in this process. We consider 'Gender' as a sensitive concept that should not be used for main-task prediction. The dataset contains 160k tweets for training and 10k tweets for the test. We merged the full dataset, subsampled 10k examples, and created an 80-20 train and test split. For pre-training a clean classifier, we create a special training set following the method described in §E.2. To generate datasets with different predictive correlation, we follow the method from E.3. The dataset is acquired and processed using the code 2 made available by the [12]. According to Twitter's policy, one has

1 MultiNLI dataset and its license could be found online at: https://cims.nyu.edu/~sbowman/multinli/

2 The code for Twitter-PAN16 and Twitter-AAE dataset acquisition is available at: https://github.com/yanaiela/ demog-text-removal

to download tweets from a personal account using Twitter Academic Research access and cannot be released to the public or used for commercial purposes. We also adhere to this policy and don't release any data to the public or use it elsewhere.

Twitter-AAE Dataset. In Twitter-AAE dataset [6], again following [12], the main task is to predict a binary sentiment (Positive or Negative) from a given tweet. The dataset contains 59.2 million tweets by 2.8 million users. Each tweet is associated with 'race' information of the user which is labeled based on both words in the tweet and the geo-location of the user. We consider 'race' as the sensitive concept which should not be used for the main task of sentiment prediction. We use the AAE (African America English) and SAE (Standard American English) as a proxy for non-Hispanic blacks and non-Hispanic whites automatically labeled using code made available by [12]. Again, we subsampled 10k examples with 80-20 split from the dataset and followed the method described in §E.2 and E.3 to generate a clean dataset for pre-training a clean classifier and datasets with different predictive correlation respectively. The dataset is made publicly available online 3 only for research-purpose.

Synthetic Dataset. To accurately evaluate the whether a classifier is using the spurious concept or not, we introduce a Synthetic-Text dataset where it is possible to change the text input based on the change in concept (thus implementing Def 2.1). The main-task is to predict whether a sentence contains a numbered word (e.g. one, fifteen etc) or not, and the spurious concept is the length of the sentence which is correlated with the main task label. To create a sentence with numbered words, we randomly sample 10 words from the following set and combine them to form the sentence.

Numbered Words = one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety, hundred, thousand

Otherwise, a sentence is created by adding 10 non-numbered words randomly sampled from the following set.

Non-Numbered Words = nice, device, try, picture, signature, trailer, harry, potter, malfoy, john, switch, taste, glove, balloon, dog, horse, switch, watch, sun, cloud, river, town, cow, shadow,

pencil, eraser

Next, we introduce the spurious concept (length) by increasing the length of the sentences which contain numbered words. We do so by adding a special word 'pad' 10 times. In our experiments, we use 1k examples created using the above method and create an 80-20 split for the train and test set. Again, we follow the method described in §E.2 and E.3 to generate a clean dataset for training a clean classifier and to generate datasets with different predictive correlations respectively. To simulate a real-world setting, we also introduce noise in the main-task and the probing label. To introduce noise (denoted by n = x ) in the labels, we randomly flip 100 x %of the main-task and probing label in the dataset. Wherever applicable, we will explicitly mention the amount of noise we add in the labels.

## E.2 Creating a 'clean' dataset with no spurious correlation with main-label

Unless otherwise specified, to construct a new dataset with no spurious correlation between the main-task and the concept label, we subsample only those examples from the the given dataset which have a fixed value of the spuriousconcept label ( y p ) . Thus, if we train main-task classifier using this dataset, it cannot use the spurious-concept since they are not discriminative of the main task label [35].

In MultiNLI dataset, we select only those examples which have no negation words in the sentence for creating a clean dataset. Similarly, for Twitter-PAN16 dataset, we only select those examples which have gender label y p = -1 (Female) in the processed dataset. And for Twitter-AAE dataset, we only select those examples which have non-Hispanic whites race label.

3 TwitterAAE dataset could be found online at: http://slanglab.cs.umass.edu/TwitterAAE/

## E.3 Creating datasets with spurious correlated main and concept label

In our experimental setup, both the main-task label ( y m ) and concept label ( y p ) are binary ( -1 or 1 ). This creates 2 × 2 subgroups for each combination of ( y m , y p ). In MultiNLI dataset, the contradiction label ( y m = 1) is correlated with the presence of negation words y p = 1 , this implies that the not-contradiction label y m = -1 is also correlated with absence of negation words in the sentence y p = -1 . Thus, the input example with ( y m = 1 , y p = 1) and ( y m = -1 , y p = -1) form the majority group, henceforth referred as S maj while groups ( y m = 1 , y p = -1) and ( y m = -1 , y p = 1) forms the minority group S min . To evaluate the robustness of the removal methods, we create multiple datasets with different predictive correlation ( κ ) between the two labels y m and y p where κ = P ( y m · y p ) &gt; 0 as defined in §4. In other words, to create a dataset with a particular predictive correlation κ , we vary the size of S maj and S min . More precisely, the predictive correlation can be equivalently defined in terms of the size of the these groups as:

$$\kappa = \frac { | S _ { m a j } | } { | S _ { m a j } | + | S _ { \min } | }$$

Similarly for Twitter-PAN16, Twitter-AAE, and Synthetic-Text datasets, we create datasets with different levels of spurious correlation between y m and y p by creating the S maj and S min to have the desired predictive correlation ( κ ).

## E.4 Encoder for real datasets

For all the experiments on real datasets in §4 we used RoBERTa as default encoder h . In §F, we report the results when using BERT instead of RoBERTa as input encoder.

RoBERTa We use the Hugging Face[47] transformers implementation of RoBERTa[25] roberta-base model, starting with pretrained weights for encoding the text-input to latent representation. We use a default tokenizer and model configuration in our experiment.

BERT We use the Hugging Face[47] transformers implementation of BERT[10] bert-base-uncased model, starting with pretrained weights for encoding the text-input to latent representation. We use a default tokenizer and model configuration in our experiment.

For both BERT and RoBERTa, the parameters of the encoder were fine-tuned as a part of training the main-task classifier for null-space removal and then frozen. For adversarial removal, the encoder, main-task classifier and the adversarial probing classifier are trained jointly. For both BERT and RoBERTa, we use the pooled output ( [ CLS ] token for BERT) from the the model, as the latent representation and is given to main-task and probing classifier. Main-task and probing classifier are a linear transformation layer followed by a softmax layer for prediction. We use a batch size of 32 samples for all training procedures that use BERT or RoBERTa for encoding the input.

## E.5 Encoder for synthetic Dataset

nBOW: neural Bag of Word. For Synthetic-Text dataset, we use sum of pretrained-GloVe embedding[30] of the words in the sentence to encode the sentence into latent representation. We used Gensim [36] library for acquiring the 100-dimensional GloVe embedding ( glove-wiki-gigaword-100 ). Throughout all our experiments, the word embedding was not trained. Post encoding, the latent representation were further passed through hidden layers consisting of a linear transformation layer followed by ReLU non-linearity. We will specify how many such hidden layers were used when discussing specific experiments in §F. The hidden layer dimensions were fixed to 50 dimensional space. We use a batch size of 32 samples for all training procedures that use nBOW for encoding the input.

## E.6 Null-Space Removal Experiment Setup

For null-space removal (INLP) experiment on both real and synthetic dataset the following procedure is followed:

1. Pretraining Phase: A clean pretrained main-task classifier is trained using the clean dataset obtained by method described in §E.2. This is to ensure that the main-task classifier does not use the spurious feature, so that the INLP method doesn't have any effect on the main-task classifier. The main-task classifier is a linear-transformation on the latent-representation provided by encoder followed by softmax layer for prediction. Both the encoder and main-task classifier is fine-tuned during this process.
2. Removal Phase: Both the encoder and main-task classifier is frozen (made non-trainable). Next, a probing classifier is trained from the latent representation of the encoder (refer §E.4 and E.5 for more details about

encoder). The probing classifier is also a linear transformation layer followed by softmax layer for prediction. For experiments on real-world datasets using BERT or RoBERTa as encoder, we train the the probing classifier for 1 epoch (one full pass though the probing dataset) before each projection step. For experiment on the Synthetic-Text dataset, we train the probing classifier for 10 epochs before each projection step. Note that, we also experiment with the setting when the main task classifier is also trained after every step of INLP projection (see §F.2 and Fig. 9 for results). The main task classifier is a linear transformation layer followed by a softmax layer for prediction trained using cross-entropy objective to predict the main task label. The main task classifier is trained for 1 epoch for the real-world datasets and 10 epochs for the Synthetic-Text dataset for the setting when we train the main task classifier in INLP removal phase. The encoder is frozen for both the setting (with or without main task classifier training) though-out the INLP removal phase.

The main-task classifier and encoder in the pretraining phase and the probing classifier in the removal phase is trained using cross-entropy loss for both real and synthetic datasets. For the real dataset, a fixed learning rate of 1 × 10 -5 is used when RoBERTa is used as encoder and 5 × 10 -5 when using BERT as encoder. For synthetic experiments, a fixed learning rate of 5 × 10 -3 is used when training both the nBOW encoder and main-task classifier in the pretraining stage and probing classifier in removal stage.

## E.7 Adversarial Removal Experiment Setup

For adversarial removal (AR) experiment, for both real and synthetic datasets, first the input text is encoded to latent representation using the encoder (§E.4 and E.5). Then for the main-task classifier, a linear transformation layer followed by a softmax layer is applied for the main-task prediction. The same latent representation output from the encoder is given to the probing classifier which is a separate linear transformation layer followed by a softmax layer. All components of the model, encoder, main-task classifier, and probing classifier are trained using the following modified objective from Eq. 95:

$$a r g \min _ { h , c _ { m } , c _ { p } } \left \{ L ( c _ { m } ( h ( z ) ) , y _ { m } ) + \lambda L ( c _ { p } ( g _ { - 1 } ( h ( z ) ) ) , y _ { p } ) \right \}$$

where h is the encoder, c m is the main task classifier, c p is the probing classifier, g -1 is the gradient reversal layer with fixed reversal strength of -1 . The first term in the objective is for training the main task classifier and the second term is the adversarial objective for training the probing classifier using gradient reversal method [15, 12] . The hyperparameter λ controls the strength of the adversarial objective. In our experiment we very λ ∈ { 0 . 00001 , 0 . 0001 , 0 . 001 , 0 . 01 , 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 } . When describing the experimental results in §F.3 we choose the λ which performs the best for all datasets with different predictive correlation κ in removing the undesired concept from the latent representation.

## E.8 Metrics Description

Analogous to spuriousness score (Def 3.1) for main-task classifier we define the score for probing classifier below.

Definition E.1 (Probe Spuriousness Score) . Given a dataset, D m,p = S min ∪ S maj with binary task label and binary concept, let Acc f ( S min ) be the minority group accuracy of a given probing classifier ( f ) and Acc ∗ ( S min ) be the minority group accuracy of a clean probing classifier that does not use the main-task feature. Then spuriousness score of f is: ψ ( f ) = | 1 -Acc f ( S min ) /Acc ∗ ( S min ) | .

For simplicity, in all our experiments we assume that both the main and the correlated attribute labels are binary. We measure the degree of spuriousness using the following two metrics:

1. Spuriousness Score: As defined in §3.4, this metric help us quantify, how much a classifier is using the spurious feature (see Def 3.1 and E.1).
2. ∆ Probability: In Synthetic-Text dataset as described in E.1, we have the ability to change the input corresponding to the change in concept label (thus implementing Def 2.1) . Thus we could measure if the main-classifier is using the spurious-concept by changing the concept in the input and measuring the corresponding change in the main-task classifier's prediction probability. The Higher the change in prediction probability higher the main-task classifier is dependent on spurious-concept.

## E.9 Compute and Resources

We used an internal cluster of Nvidia P40 and P100 GPUs for all our experiment. Each experiment setting was run on three random seed and mean results with variance are reported in all the experiment.

Figure 5: Failure Modes of Probing classifier: The first row in Fig. 5a and 5b shows that even when the latent representation doesn't contain the probing concept-causal feature, the probing classifier is still has &gt;50% accuracy when other correlated feature is present. The accuracy increases as the correlation κ between the probing concept-causal feature and other correlated features increases. The first row Fig. 5c shows that presence of correlated features could increase the probing classifier's accuracy thus increasing the confidence in the presence of concept-causal feature in latent representation. The second row of all the figures shows that the probing classifier is getting more spurious as the κ increases thus implying that the probing classifier is using some other correlated feature than concept-causal feature. For more discussion see §F.1.

<!-- image -->

## F Additional Results

## F.1 Probing classifier Quality

Fig. 5 shows different failure modes of the probing classifier. In Fig. 5a and 5b, a clean main-task classifier which doesn't use the concept feature is trained on Synthetic-Text and MultiNLI dataset respectively using the method described in §E.2. Thus the latent representation doesn't have the concept feature. Then, to test the presence of concept-causal feature in the latent representation we train a probing classifier to predict concept-label. The first row show the accuracy of the probing classifier for testing the presence of concept in latent space. When κ = 0 . 5 i.e no correlation between the main-task and the concept label, the probing accuracy is approximately 50% which correctly shows the absence of the concept-causal feature in the latent representation. The accuracy increases as the correlation κ between the main and concept-causal feature increases in dataset. This shows that even when concept-causal feature is not present in the latent representation, probing classifier will still claim presence of concept-causal feature if any correlated feature (main-task feature in this case) is present in the latent space. In Fig. 5c, the latent space contains the concept-causal feature as shown by accuracy of approximately 94.5% when κ = 0 . 5 . But as κ increases the probing classifier's accuracy increases in the presence of correlated main-task feature which falsely increases the confidence of presence of the concept-causal feature. The second row shows the spuriousness-score of concept-probing classifier is increasing as the correlation between the main-task and concept-causal feature increases which implies that the probing classifier is using relatively large amount of correlated main-task feature for concept-label prediction in all settings.

For all the experiments in this section (§F.1) with Synthetic-Text dataset, we didn't introduce any noise in the probing label (i.e. n=0.0) and have 1 hidden layer when training the encoder (see §E.5 for details). For the experiment on MultiNLI dataset, we use RoBERTa as the default encoder and rest of setup is same as described in §E.4.

## F.2 Extended Null-Space Removal Results

Fig. 6 and 7, shows the failure mode of null-space removal (INLP) in the real dataset when using RoBERTa and BERT as encoders respectively. Different columns of the figure are for three different real datasets - MultiNLI, Twitter-PAN16, and Twitter-AAE respectively. The x-axis from steps 8-26 is different INLP removal steps. The y-axis shows different metrics to evaluate the main task and probing classifier. Different colored lines show the spurious correlation ( κ ) in the probing dataset used by INLP for the removal of spurious-concept. The pretrained classifier is clean, i.e., does not use the spurious concept-causal feature; hence INLP shouldn't have any effect on main-classifier when removing concept-causal feature from the latent space. The first row shows that as the INLP iteration progresses, the norm of latent representation, which is being cleaned of concept-causal feature, decreases. This indicates that some features are being removed. However, the results are against our expectation from the second statement of Theorem 3.2, which

states that the norm of the classifier will tend to zero as the INLP removal progresses. The possible reason is that from Theorem 3.2 the norm of latent representation will go to zero when the latent representation only contains the spurious concept-causal feature and the other features correlated to it. But, the encoder representation could have other features which are not correlated with concept-label and hence not removed. Since, the pretrained classifier given for INLP was clean (using method described in §E.2), we do not expect the INLP to have any effect on the main-task classifier.

The second row in Fig. 6 and 7 shows that the main classifier accuracy drops to random guess i.e 50% except for the case when probing dataset have κ = 0 . 5 i.e no correlation between the main and concept label. Thus INLP method corrupted a clean classifier and made it useless. The reason behind this could be observed from the fourth and fifth rows. The fourth row shows the accuracy of the probing classifier before the projection step. We can see that at step 8 on the x-axis κ = 0 . 5 , the probing classifier correctly has an accuracy of 50% showing that the concept-causal feature is not present in the latent representation. But for other values of κ , the probing classifier accuracy is proportional to the value of κ implying that the probing classifier is using the main-task feature for its prediction. Hence at the time of removal, it removes the main-task feature which leads drop in the main-task accuracy. This can also be verified from the last row of Fig. 6 and 7, which shows that the spuriousness score of probing classifier is high; thus it is using the main-task feature for its prediction. We observe similar results for Synthetic-Text dataset when using INLP in Fig. 8. For all the INLP experiment on Synthetic-Text dataset, there were no hidden layers after the nBOW encoder (see §E.5).

So far, we have kept the main-task classifier frozen when performing INLP removal. Note that, we also experiment with the setting when the main task classifier is trained after every projection step of INLP (see §E.6 for experimental setup and Fig. 9 for a result description). We observe a similar drop in the main-task accuracy with prolonged removal using INLP and early stopping leads to an even higher reliance on the spurious concept-causal feature than it had at the beginning of INLP. The rest of the experimental configurations were kept the same as the other INLP experiments described above.

## F.3 Extended Adversarial Removal Results

Adversarial removal failure in real-world datasets. Fig. 10 shows the failure mode of adversarial removal AR on real-world datasets. In the x-axis we vary the predictive correlation κ between the main and the concept-label in different datasets and measure the performance of AR on different metrics on the y-axis. The second row shows the spuriousness score of the main-task classifier after AR as we vary κ on the x-axis. When using RoBERTa as the encoder, the orange curve in second row shows the spuriousness score of the main-task classifier when trained using the ERM loss. The spuriousness score describes how much unwanted concept-causal feature the main-task classifier is using. The blue curve shows that the AR method reduces the spuriousness of main-task though cannot completely remove it. The reason for this failure can be attributed to probing classifier. Even when AR has successfully removed the unwanted concept feature, the accuracy of concept-probing classifier will be proportion to κ due to presence of correlated main-task feature in the latent space. This can be seen in the third row of Fig. 10. Thus we cannot be sure if the unwanted concept-causal feature has been completely removed from the latent space or just became noisy enough to have accuracy proportional to κ after AR converges. In Fig. 10, for each dataset and encoder, we manually choose the hyperparameter described λ described in §E.7 which reduces the spuriousness score most for the main-task classifier while not hampering the main-task classifier accuracy. In Fig. 11, we show the trend in spuriousness score is similar for all choices of hyperparameter λ in our search. No value of λ is able to completely reduce the spuriousness score to zero.

Adversarial removal makes a classifier clean. Fig. 12 shows that when the adversarial classifier is initialized with a clean main-task classifier that doesn't use unwanted-concept causal features, it makes matters worse by making the main-task classifier use the unwanted-concept feature. For the Synthetic-Text dataset, since the word embeddings are non-trainable, one single hidden layer is applied after the nBOW encoder so that AR methods could remove the unwanted-concept feature from the new latent representation. We create a clean Synthetic-Text dataset by training a classifier (iteration 1-20) on dataset with predictive correlation κ = 0 . 5 between the main-task and concept label. κ = 0 . 5 which implies there is no correlation between the labels thus we can expect the main-task classifier to not use the concept-causal feature. This can be seen from Main classifier spuriousness score in Fig. 12a (2nd row) which is close to 0. We chose this method to create a clean classifier since this allows us to measure the spuriousness score for the main-task classifier. If we would have followed method described in [35], then we would have had only a single value of concept label ( y p ) in the dataset and couldn't have defined the majority and minority group required for calculation of spuriousness score (see Def 3.1). For all our experiments on Synthetic-Text dataset we use noise =0.3 and trained the main-task and probing classifier with 1 hidden layer. Similarly for training a clean classifier for MultiNLI dataset (iteration 1-6) we again use a dataset with predictive correlation κ = 0 . 5 . Post training the clean classifier the AR method is initialized with these clean classifiers for removal of concept-causal features. Since AR is initialized with clean classifier which doesn't use concept-causal feature, we expect AR to have no effect on the classifier. In contrast

Table 1: Correlation between Spuriousness Score and ∆ Prob on Synthetic-Text dataset: Pearson-correlation between Spuriousness score and ∆ Prob; the two metrics for quantifying the dependence of a classifier on a spurious feature. We measure the correlation for adversarial-removal experiment over two different noise setting on SyntheticText dataset. For more details, see §F.4. The first column shows different experimental settings and the second column shows the Pearson correlation between the two metrics. The third column shows the p-value under the null hypothesis that the two metrics are uncorrelated. Both correlations are statistically significant since p-value for both the case is &lt; 0.05.

|                        |   Pearson Correlation |   p-value |
|------------------------|-----------------------|-----------|
| Synthetic-Text + n=0.1 |                  0.83 |    0.0403 |
| Synthetic-Text + n=0.3 |                  0.95 |    0.0033 |

we observe that the spuriousness score of main-classifier for both Synthetic-Text and MultiNLI dataset increases (2nd row in Fig. 12a and 12b) which shows that AR when initialized clean /fair classifier could make them unclean/unfair.

## F.4 Synthetic-Text dataset Ablations

Adversarial Removal Failure in Synthetic-Text dataset: Figure 13 shows the failure of AR on the synthetic dataset as we vary the noise in the main-task label and unwanted concept-label. For the experiment, since the word embeddings are non-trainable, one single hidden layer is applied after the nBOW encoder so that AR methods could remove the unwanted-concept feature from the new latent representation.

Dropout Regularization Helps AR method: Continuing on observation from Fig. 14a, 14b and 14c shows the ∆ -Prob of the main-task classifier after we apply the AR on Synthetic-Text dataset (with noise=0.3) and how they changes as we increase the dropout regularization. As we increase the dropout (drate in the figure), the ∆ -Prob of the main classifier decreases showing that the regularization methods could help improve the removal methods.

## G Comparison between Spuriousness Score and ∆ Prob

In this section we compare the Spuriousness Score proposed in §3.4 for measuring a classifier's use of a binary spurious feature with the ideal, ground-truth metric, ∆ Probability ( ∆ Prob for short) defined in §E.8. ∆ Prob measures the reliance on a spurious feature by changing the spurious feature in the input space (when possible) and measuring the change in the prediction probability of the given classifier. Hence ∆ Prob is a direct and intuitive measure of spuriousness in a given classifier. But changing the spurious feature is difficult in the input space for real-world data, thus we only evaluate this metric on the Synthetic-Text dataset.

To do so, we use the result from Fig. 13 that showed failure of the adversarial removal method on the Synthetic-Text dataset under various noise settings (refer §F.4 for details). For the setting with noise n = 0 . 0 , both Spuriousness Score and ∆ Prob curve for Adversarial Removal (marked as ADV in Fig. 13) are identical (close to 0 for all values of κ with mean = 0 . 0 and standard-deviation = 0 . 0 ). For the other settings with non-zero noise, we compute the Pearson correlation between the Spuriousness score and ∆ Prob for the ADV curve. As Table 1 shows, we observe high Pearson correlation of 0 . 83 and 0 . 95 for the noise setting, n = 0 . 1 and n = 0 . 3 respectively. The third column in the table shows p-value ( &lt; 0 . 05 ) assuming a null hypothesis that the two metrics are uncorrelated. These results suggest that Spuriousness-Score can be a good approximation for the ideal ∆ Prob metric.

Figure 6: Failure of Null Space Removal when using RoBERTa as encoder: Different columns of the figure are for three different real datasets - MultiNLI, Twitter-PAN16, and Twitter-AAE respectively. The x-axis from steps 8-26 is different INLP removal steps. The y-axis shows different metrics to evaluate the main task and probing classifier. Different colored lines show the spurious correlation ( κ ) in the probing dataset used by INLP for removal of spurious-concept. The pretrained classifier is clean i.e. doesn't use the spurious concept-causal feature, hence INLP shouldn't have any effect on main classifier when removing concept-causal feature from the latent space. Against our expectation, the second row shows that the main-task classifier's accuracy is decreasing even when it is not using the concept-feature. The main reason for this failure to learn a clean concept-probing classifier. This can be verified from the last row which shows that the concept-probing classifier has a high spuriousness score thus implying that it is using the main-task feature for concept label prediction and hence during the removal step, wrongly removing the main-task feature which leads to a drop in main-task accuracy. For more discussion see §F.2.

<!-- image -->

Figure 7: Failure of Null Space Removal when using BERT as encoder: The observation is similar to the case when RoBERTa was used as encoder (see Fig. 6) . Different columns of the figure are for three different real datasets MultiNLI, Twitter-PAN16, and Twitter-AAE respectively. The x-axis from steps 8-26 is different INLP removal steps. The y-axis shows different metrics to evaluate the main task and probing classifier. Different colored lines show the spurious correlation ( κ ) in the probing dataset used by INLP for removal of spurious-concept. The pretrained classifier is clean i.e. doesn't use the spurious concept-causal feature, hence INLP shouldn't have any effect on main-classifier when removing concept-causal feature from the latent space. Against our expectation, the second row shows that the main-task classifier's accuracy is decreasing even when it is not using the concept-feature. The main reason for this failure to learn a clean concept-probing classifier. This can be verified from the last row which shows that the concept-probing classifier has high spuriousness score thus implying that it is using the main-task feature for concept label prediction and hence during the removal step, wrongly removing the main-task feature which leads to a drop in main-task accuracy. For more discussion see §F.2.

<!-- image -->

Figure 8: Failure Mode of INLP in Synthetic-Text dataset : Different columns of the figure are Synthetic-Text dataset with different levels of noise in the main task and probing task label. Here, n=0.0 means there is 0% noise and n=0.3 means there is 30% noise in the labels. The x-axis from steps 22-40 is different INLP removal steps. The y-axis shows different metrics to evaluate the main task and probing classifier. Different colored lines show the spurious correlation ( κ ) in the probing dataset used by INLP for the removal of spurious-concept. The pretrained classifier is clean i.e. doesn't use the spurious concept-causal feature, hence INLP shouldn't have any effect on main classifier when removing concept-causal feature from the latent space. Contrary to our expectation, the first row shows main-task classifier accuracy drops as the INLP progresses. Higher the correlation between the main-task and concept label, faster the drop in the main task accuracy. The last row shows the change in prediction probability ( ∆ -Prob) of main-task classifier when we change the input corresponding to concept-label. This shows, how much sensitive the main task classifier is wrt. to concept feature. We observe that the ∆ -Prob increases in the middle of INLP showing that the main-classifier which was not using the concept initially (as in iteration 21), started using the sensitive concept because of INLP removal. Thus stopping INLP prematurely could lead to a more unclean classifier than before whereas running INLP longer removes all the correlated features and could make the classifier useless. For more discussion see §F.2.

<!-- image -->

Figure 9: Failure Mode of INLP + Main Task Classifier head retraining : Given a pretrained encoder and the main task classifier as input to INLP for spurious concept removal, in this experiment, we retrain the main task classifier after every step of null-space projection by INLP. All the other experiment configurations for these experiments are kept the same as the case when we don't retrain the main-task classifier. The first, second, and third columns show the results for Synthetic-Text, MultiNLI, and Twitter-AAE datasets respectively. We observe a similar trend as the case when the main task classifier was not trained after each projection step (see Fig. 6, 7 and 8). The main task classifier's accuracy drops as the null-space removal proceeds (iteration 21-40 for Synthetic-Text and iteration 7-26 for MultiNLI and Twitter-AAE datasets). Though the drop is not as severe as in the previous setting (when we didn't train the main task classifier), it is significant enough to impact the practical utility of the model (greater than 20% drop in the accuracy when κ &gt; 0 . 8 for all the datasets above). Similar to previous setting, early-stopping of INLP removal may lead to a classifier that has a higher reliance on the spurious concept than it had before the INLP removal. For example, for κ = 0 . 8 in Synthetic-Text dataset, the main-task classifier's performance drops for the first time at iteration 29 (a valid heuristic for early stopping), but it has high ∆ Prob ≈ 10% as shown in the last row of the Synthetic-Text dataset column of this figure. For discussion of the case when the main task classifier is not trained after every projection step, see §F.2 and §4.2.

<!-- image -->

Figure 10: Failure Mode of Adversarial removal on real-dataset: Different column shows the result on three different real datasets -MultiNLI, Twitter-PAN16, and Twitter-AAE respectively. The second row shows the accuracy of spuriousness score of the main-task classifier after AR when the dataset contains different levels of spurious correlation between the main-task and unwanted-concept label, denoted by κ in the x-axis. When using RoBERTa as the encoder, the orange curve in second row shows the spuriousness score of the main-task classifier when trained using the ERM loss. The spuriousness score describes how much unwanted concept-causal feature the main task classifier is using. The blue curve shows that the AR method reduces the spuriousness of main-task though cannot completely remove it. When using BERT as encoder, the observation is same i.e green curve in second row shows AR is able to reduce the spuriousness of main classifier than the red curve which is trained using ERM, but is not able to completely remove the spurious feature. For more discussion see §F.3.

<!-- image -->

Figure 11: Choice of Adversarial Strength Parameter λ : The second plot shows that trend in spuriousness score after AR is similar for all the choices of hyperparameter λ we have taken in our search. None of the settings of λ is able to completely reduce the spuriousness score to zero. For more discussion see §F.3.

<!-- image -->

Figure 12: Adversarial Removal Makes a classifier unclean : We test if the AR method increases the spuriousness of a main-task classifier if initialized with a clean classifier. In 12a, from iteration 1-20 in x-axis, a clean classifier is trained on Synthetic-Text dataset (with 30% noise i.e n=0.3 in main-task and probing labels) such that it doesn't uses the unwanted concept-causal feature by training on a dataset with κ = 0 . 5 (see §F.3 for details). Then the classifier is given to AR method for removing the unwanted concept feature which makes the initially clean classifier unclean. This can be seen from the second row of the 12a which shows the spuriousness score of main-classifier is 0 during 1-20 iteration but increases after the AR start from 21-40. Also, the last row shows the ∆ -Prob of the main-task classifier on changing the unwanted-concept in input which increases for datasets which have large κ i.e correlation between the main and concept label. A similar result can be seen for the MultiNLI dataset where a clean classifier is trained in iterations 1-6 (using a dataset with κ = 0 . 5 ) which is made unclean by AR. Second row again shows that spuriousness score of main-task classifier increases after AR starts in iteration 7-12. For more discussion see §F.3.

<!-- image -->

-Probability(all)

Main

Figure 13: Failure of Adversarial Removal method on Synthetic-Text dataset: Different columns show the adversarial removal method on Synthetic-Text dataset with different levels of noise in the main-task and concept label. When there is no noise, from the second row in Fig. 13a, we see that both the classifier trained by ERM and AR has zero-spuriousness score. But as we increase the noise to 10% in Fig. 13b, we observe that the spuriousness score increases when AR is applied in contrast to classifier trained by ERM which stays at 0. Also, higher the predictive correlation κ , higher the increase in spuriousness. This observation augments the observation in Fig. 12 which shows that using AR makes a clean classifier unclean. Similarly in Fig. 13c when we increase the noise to 30% we observe in second row, AR is increased the spuriousness, unlike ERM which is at 0. For discussion see §F.4

<!-- image -->

Figure 14: Dropout Regularization helps in Adversarial Removal: ∆ -Prob of the main-task classifier after we apply the AR on Synthetic-Text dataset (with noise=0.3) decreases as we increase the dropout regularization from 0.0 to 0.9. For discussion see §F.4.

<!-- image -->