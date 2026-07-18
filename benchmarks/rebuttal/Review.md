Official Review of Submission4856 by Reviewer WzsD
Official Reviewby Reviewer WzsD16 May 2026, 15:35 (modified: 28 May 2026, 21:48)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer WzsD, AuthorsRevisions
Strengths:
Hybridizing retrieval-based and drafter-based SD for VLA is a compelling idea that leverages complementary strengths (draft quality vs. system overhead).
The kinematics-informed boundary selection (via fused radius–displacement metric) is novel and well-motivated for embodied control.
Addressing VLA latency with SD is a high-impact problem; a 2×+ speedup with minimal SR loss can meaningfully improve real-time deployment.
Weaknesses:
The kinematic fused metric is heuristic; there is no quantitative correlation analysis between “overlap vs. non-overlap” and the proposed radius/displacement signals, and no robustness analysis across tasks/robots.
The retrieval database construction and embedding choices are only sketched (e.g., feature types, conditioning on language/task), complicating reproducibility and interpretation of Top-1 “confidence.”
The paper cites Spec-VLA and KERV but does not compare empirically. The novelty claims would be stronger if contrasted quantitatively against KERV’s kinematics-aware SD adaptations and Spec-VLA’s relaxed acceptance.
Review:
HeiSD tackles the accelerating VLA inference problem by proposing a thoughtful hybridization of retrieval-based and drafter-based speculative decoding, augmented with kinematics-aware boundary selection and VLA-specific verification relaxations. The reported 2×-class speedups on both simulation and real hardware are compelling, and the methodology contains several promising, domain-aware ideas (sequence-wise acceptance, gripper handling, CPU–GPU mapping). However, several key aspects are underspecified or internally inconsistent: the kinematic fused metric lacks quantitative validation against overlap/acceptance. Baseline coverage omits closely related SD variants (Spec-VLA, KERV), and real-world reporting needs more detail and safety analysis.

Fit: 4: Large audience
Fit Justification:
The content of this paper aligns with the scope of ACM MM.

Technical Quality: 3: Good
Technical Presentation: 3: Fair
Rating: 3: Borderline
Confidence: 4: Expert
Add:
Official Review of Submission4856 by Reviewer NBur
Official Reviewby Reviewer NBur15 May 2026, 00:14 (modified: 28 May 2026, 21:48)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer NBur, AuthorsRevisions
Strengths:
Practical problem. The paper addresses inference acceleration for VLA models, which is important for real-time robotic control.
Clear motivation. It shows that drafter-based SD and retrieval-based SD have complementary strengths, motivating a hybrid design.
Embodied-specific design. The kinematic-aware boundary based on trajectory curvature and displacement is well aligned with robotic action generation.
Promising results. HeiSD achieves clear speedups in both simulation and real-world tasks while maintaining reasonably high success rates.
Weaknesses:
Some accuracy loss. HeiSD still causes success-rate drops in several settings, especially harder or longer-horizon tasks.
Heuristic design. The kinematic fused metric and threshold choice are intuitive but not deeply justified.
Database dependence. The method relies on the coverage and quality of the retrieval database, which may limit OOD generalization.
Limited architecture coverage. Experiments are mainly based on OpenVLA-style models, so generality to other VLA architectures remains unclear.
Review:
This paper proposes HeiSD, a hybrid speculative decoding framework for accelerating VLA inference. It combines retrieval-based SD and drafter-based SD, and uses a kinematic-aware metric to decide which decoding strategy to use at different trajectory stages. The idea is practical and well motivated, and the real-world experiments strengthen the contribution.

However, the method is somewhat heuristic and still shows success-rate degradation in some settings. Its dependence on retrieval database quality also deserves more discussion. Overall, I lean toward Weak Accept because the paper addresses an important problem and provides a useful system-level solution.

Fit: 4: Large audience
Fit Justification:
The paper is relevant to ACM MM because it studies efficient inference for VLA models involving vision, language, and action. Although the main contribution is system-level acceleration, it fits embodied multimodal AI and real-time multimodal model deployment.

Technical Quality: 3: Good
Technical Presentation: 4: Good
Rating: 4: Weak Accept
Confidence: 2: Familiar
Add:
Official Review of Submission4856 by Reviewer t4h4
Official Reviewby Reviewer t4h414 May 2026, 21:53 (modified: 28 May 2026, 21:48)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer t4h4, AuthorsRevisions
Strengths:
The paper targets the slow inference speed of VLA models, which is a critical bottleneck for real-time robotic deployment.
The paper provides a reasonable analysis of the complementary properties of drafter-based SD and retrieval-based SD. Drafter-based SD provides higher-quality drafts but introduces additional overhead, while retrieval-based SD is faster but less reliable.
The trajectory analysis shows that database retrieval is reliable for some overlapping trajectory segments but unreliable for deviated segments. This gives a clear motivation for hybridizing retrieval-based and drafter-based SD.
Weaknesses:
The method relies heavily on a prebuilt demonstration database. If the test trajectory has low overlap with stored demonstrations, retrieval-based SD may become unreliable, which may limit generalization to novel tasks, environments, or camera settings
Although HeiSD achieves clear speedups, it also reduces task success rates in several settings. For example, on LIBERO-Goal, SR drops from 77.0% under AR inference to 73.0%; on LIBERO-Long, it drops from 54.4% to 47.0%. This weakens the claim of maintaining a high success rate.
The main experiments are based on OpenVLA. More experiments on different VLA architectures would make the generality claim more convincing.
Many recent VLA models no longer rely on conventional autoregressive action-token generation. Instead, they often introduce an additional action head or action expert to generate continuous action chunks, such as diffusion- or flow-matching-based policies. Therefore, the proposed speculative-decoding framework may be difficult to directly transfer to related architectures such as the π-series models and GR00T.
Review:
This paper proposes HeiSD, a hybrid speculative decoding framework for accelerating VLA model inference. The method combines drafter-based SD and retrieval-based SD, using retrieval for reliable trajectory segments and drafter-based decoding for more uncertain segments. To improve retrieval-based SD, the paper introduces adaptive verify-skip and sequence-wise relaxed acceptance. It also uses a kinematic-based fused metric to automatically decide the hybrid boundary. Experiments on LIBERO and real-world robot tasks show clear speedups, but the method also causes some success rate degradation and depends heavily on database coverage. Moreover, the proposed framework appears to be closely tied to autoregressive action-token generation. Many recent VLA architectures, such as the π-series models and GR00T, adopt action heads or action experts to generate continuous action chunks through diffusion or flow matching, which raises concerns about the generality and transferability of the proposed method to non-autoregressive VLA architectures. Overall, the paper addresses an important problem and presents a practical acceleration framework, but its robustness, generalization, and applicability to broader VLA architectures need further validation.

Fit: 4: Large audience
Fit Justification:
The paper is a strong fit for ACM Multimedia because it focuses on efficient Vision-Language-Action modeling, which is closely related to multimodal learning, vision-language understanding, embodied AI, and efficient inference. Although the application is robotic manipulation, the core problem involves multimodal representation, autoregressive decoding, retrieval, and acceleration of large multimodal models. These topics are likely to attract a relatively broad audience within the ACM Multimedia community, especially researchers working on multimodal systems, embodied intelligence, and efficient model inference.

Technical Quality: 3: Good
Technical Presentation: 3: Fair
Rating: 3: Borderline
Confidence: 3: Knowledgeable
Add:
Official Review of Submission4856 by Reviewer JF71
Official Reviewby Reviewer JF7113 May 2026, 19:20 (modified: 28 May 2026, 21:48)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer JF71, AuthorsRevisions
Strengths:
The method makes an interesting attempt to incorporate robot kinematics into speculative decoding. In particular, the hybrid boundary is determined using a fused metric based on curvature radius and cumulative displacement, rather than relying entirely on fixed heuristic rules.
The reported speedup is very significant. HeiSD achieves strong acceleration in both simulation and real-world settings, which makes the problem and direction practically meaningful.
Weaknesses:
The paper mainly shows that retrieval-based SD has low accepted length and slow speed, but it lacks an end-to-end latency breakdown. It is therefore hard to quantify the runtime contributions of ViT, LLM, drafter, retrieval, and verification. Whether the main bottleneck indeed comes from SD verification is not fully demonstrated.

The paper directly skips part of the verification process and states that the impact on the output distribution is outside the scope. However, verification is a core guarantee of speculative decoding. Skipping it may introduce action distribution shifts and safety risks. The paper does not analyze wrong actions, gripper errors, collisions, or unsafe trajectories, which could be important for real-world robot deployment.

The description of CPU-GPU communication overhead is confusing. The text says the communication overhead is “negligible overhead (>100 ms),” but an overhead larger than 100 ms is usually not negligible. Meanwhile, Figure 7 reports the embedding transfer latency as 0.25 ms. This inconsistency makes the system analysis hard to interpret.

The authors argue that diffusion-based action generators still require autoregressive intermediate features from the LLM, so HeiSD can still accelerate them. However, the paper does not explain how the multi-step generation process of a diffusion action expert would be handled by speculative decoding, nor does it provide experiments to support this claim.

The paper claims that HeiSD has no specific requirements on tasks or robot platforms. However, the experiments are mainly based on OpenVLA, LIBERO, and one real-world robot platform. The adaptation to diffusion-based VLAs is only discussed conceptually and is not experimentally validated.

The method requires a pre-built database. In real-world tasks, the authors also need to collect new demonstration data, rebuild the database, and fine-tune the models. This makes the system highly dependent on data and scenarios, which may increase the difficulty of practical deployment.

There seem to be some issues in Algorithm 1. First, in the offline stage, minS is initialized to 0, but the condition is if minS > S > T. If the similarity score S and threshold T are positive, this condition will almost never be satisfied. Second, the online-stage else condition is written as else B_t = True then ..., which repeats the previous if B_t = True condition. It should likely be B_t = False or another condition.

The paper says “Fig. 3(b) details the proposed adaptive verify-skip mechanism,” but the adaptive verify-skip mechanism actually corresponds to Figure 4 rather than Figure 3.

There are several typos. For example, the title of Figure 8 is written as “Hybrid Raito,” and Figure 7 contains “Dafter.”

There is also a terminology issue. Throughout the paper, SD is defined as Speculative Decoding, including in the title. However, on page 8, the paper refers to “Retrieval-based Stable Diffusion (Retrieval-based SD).” This is confusing and should be clarified, since such terminology inconsistency may reduce the credibility of the paper.

Review:
Please refer to Strengths and Weaknesses.

Fit: 5: Perfect match
Fit Justification:
This paper fits well within the scope of ACM Multimedia. It studies efficient vision-language-action models, which are closely related to multimodal representation learning, vision-language reasoning, embodied AI, model compression, and efficient inference. These topics are relevant to the ACM Multimedia community, especially with the growing interest in embodied foundation models and deployable AI systems.

Technical Quality: 3: Good
Technical Presentation: 3: Fair
Rating: 2: Weak Reject
Confidence: 3: Knowledgeable
Add:
Official Review of Submission4856 by Reviewer 3r5v
Official Reviewby Reviewer 3r5v12 May 2026, 15:50 (modified: 28 May 2026, 21:48)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 3r5v, AuthorsRevisions
Strengths:
The paper studies an important practical problem for Vision-Language-Action models: inference latency during robot control. The idea of combining drafter-based and retrieval-based speculative decoding is intuitive and well motivated, since the two approaches have complementary strengths and weaknesses. This paper does not simply propose a hybrid framework directly, but first analyzes retrieval trajectories and shows that some segments overlap well with VLA inference while others diverge. This observation gives a reasonable motivation for switching between the two decoding modes. The proposed framework is fairly complete and includes several concrete designs, such as adaptive verify-skip, sequence-wise relaxed acceptance, and a kinematic metric for determining the hybrid boundary. The experimental section is also reasonably solid, including four LIBERO suites with 50 trials per task as well as real-world robot experiments.

Weaknesses:
My major concern is that the baseline OpenVLA is out of date. While I understand that the speculative decoding will require the use of an autoregressive model, however, the current trend of VLA is to use parallel decoding, which is more efficient and achieves much better performance. We can also see the results on LIBERO in Table 3. Right now, most VLA model have already achieved more than 95% successful rate in LIBERO, yet the proposed method seems to only have around 60-70% average SR. If the authors can not prove HeiSD's performance on more advanced VLA model (or at least demonstrating under which condition AR VLA model will perform better), it raises doubt about whether the combination of HeiSD and VLA is meaningful.

No other compared methods are included in the experiment. There are many methods to build efficient VLA, like token pruning. If the authors can not prove that SD-based method has some unique advantages under some certain settings, they should include other efficient VLA methods for comparison.

No compared method provided in real robot experiment (at least in Table 3, we have SpecVLA for comparison).

Review:
The paper addresses an important practical problem in VLA models: reducing inference latency during robot control. The motivation is reasonable, and the idea of combining drafter-based and retrieval-based speculative decoding is intuitive because the two strategies have complementary strengths. The paper also provides useful analysis of retrieval trajectories and proposes a fairly complete framework with adaptive verify-skip, sequence-wise relaxed acceptance, and a kinematic metric. The experiments cover four LIBERO suites and real-world robot settings, which makes the evaluation reasonably solid.

However, my main concern is the practical relevance of the method. The main baseline, OpenVLA, is relatively outdated, while many recent VLA models use parallel decoding and achieve much stronger performance on LIBERO. Since HeiSD is only demonstrated on an AR-based VLA model with relatively low success rates, it is unclear whether the proposed acceleration is meaningful for current state-of-the-art VLA systems.

In addition, the paper lacks comparison with other efficient VLA methods, such as token pruning or other computation-reduction approaches. The real-world robot experiment also does not include a competing baseline. Therefore, although the problem and framework are interesting, the current experiments are not sufficient to fully support the practical advantage of the proposed method.

Fit: 3: Relevant to part of the community
Fit Justification:
VLA is a popular research direction in embodied AI community. Therefore, it should have some relevance with MM conference.

Technical Quality: 3: Good
Technical Presentation: 3: Fair
Rating: 2: Weak Reject
Confidence: 4: Expert
