# ASL Model Hyperparameter Inventory

This document inventories the tunable settings and architectural choices in the current ASL learning pipeline. The production model accepts a `60 x 694` sequence, uses three bidirectional LSTM blocks with a total hidden width of 384, applies single-head temporal attention, predicts 250 vocabulary classes, and has 3,030,011 trainable parameters.

## 1. Core architectural hyperparameters

1. Input feature width: `694`.
2. Sequence length: `60` timesteps.
3. Hidden width: `384`.
4. Number of recurrent layers: `3`.
5. LSTM directionality: bidirectional.
6. Hidden units per direction: `192`.
7. Recurrent cell type: LSTM.
8. LSTM projection size: `0` (no internal projection).
9. LSTM bias: enabled.
10. LSTM batch layout: batch-first.
11. Internal stacked-LSTM dropout: `0`.
12. Initial hidden state: zeros.
13. Initial cell state: zeros.
14. Raw-input projection: `Linear(694 -> 384)`.
15. Input-projection bias: enabled.
16. Input-projection normalization: LayerNorm.
17. Input-projection activation: ReLU.
18. Residual connections: enabled after every LSTM.
19. Residual topology: `activation(norm(LSTM(x)) + x)`.
20. Inter-layer normalization: LayerNorm.
21. Inter-layer activation: ReLU.
22. Number of residual recurrent blocks: `3`.
23. Post-block dropout probability: `0.5`.
24. Dropout placement: after every residual activation.
25. Dropout in-place operation: disabled.
26. Temporal pooling method: learned attention-weighted sum.
27. Number of attention heads: `1`.
28. Attention score network: `Linear(384 -> 1)`.
29. Attention-score bias: enabled.
30. Attention normalization: softmax.
31. Attention softmax axis: temporal dimension.
32. Classifier input width: `384`.
33. Classifier output width: vocabulary size, currently `250`.
34. Classifier bias: enabled.
35. Output activation during training: none; the model returns raw logits.
36. Prediction rule: maximum-logit class.

Changing the input width, hidden width, layer count, attention structure, vocabulary size, or classifier shape makes the existing checkpoint incompatible.

## 2. LayerNorm hyperparameters

37. Normalization type: LayerNorm.
38. LayerNorm epsilon: `1e-5`.
39. Learnable LayerNorm scale: enabled.
40. Learnable LayerNorm bias: enabled.
41. Normalized dimension: `384`.
42. Normalization granularity: independently per timestep.
43. Batch statistics: not used.

## 3. Input representation hyperparameters

44. Body-frame coordinate width: `282`.
45. Canonical-handshape width: `126`.
46. Velocity width: `282`.
47. Presence-mask width: `4`.
48. Total cached coordinate width: `408`.
49. Coordinate axes per landmark: `3` (`x`, `y`, and `z`).
50. Right-hand landmarks retained: `21`.
51. Left-hand landmarks retained: `21`.
52. Lip landmarks retained: `40`.
53. Pose landmarks retained: `12`.
54. Canonical-handshape copies per frame: `2`.
55. Velocity feature inclusion: enabled.
56. Presence feature inclusion: enabled.
57. Face features beyond lips: excluded.
58. Lower-body pose features: excluded.
59. Hand visibility/confidence values: excluded.
60. Pose visibility/confidence values: excluded.

The exact landmark subsets are categorical hyperparameters: the retained lips, pose points, fingers, face regions, and confidence features can all be varied.

## 4. Coordinate-normalization hyperparameters

61. Primary body-frame origin: midpoint of the left and right shoulders.
62. Primary body scale: shoulder distance.
63. Primary body orientation: shoulder axis made horizontal.
64. Fallback origin: face nose tip.
65. Fallback scale: outer-eye distance.
66. Fallback orientation: outer-eye axis made horizontal.
67. Handshape origin: wrist.
68. Handshape scale: wrist-to-middle-finger-MCP distance.
69. Handshape target angle: `-pi/2`.
70. Body-frame target angle: `0`.
71. Minimum normalization scale: `1e-6`.
72. Rotation dimensionality: 2D `x/y`.
73. Full 3D orientation normalization: disabled.
74. Per-hand canonical normalization: enabled.
75. Shared body-coordinate normalization: enabled.
76. Missing-part fill value: zero.
77. Behavior without a body or face reference: retain canonical handshape only.

## 5. Temporal preprocessing hyperparameters

78. Fixed resampled sequence length: `60`.
79. Resampling strategy: uniform across the complete gesture.
80. Resampling index interpolation: linear position generation.
81. Frame-index quantization: nearest integer via rounding.
82. Coordinate interpolation: none; nearest selected frame.
83. Long-clip handling: resample the entire clip.
84. Short-clip handling: duplicate selected frames.
85. Temporal cropping: disabled.
86. Multiple windows per clip: disabled.
87. Examples per video: one.
88. Labels per video: one.
89. Sequence padding: disabled.
90. Variable-length batching: disabled.
91. Padded-timestep masking: disabled.

## 6. Velocity-feature hyperparameters

92. Velocity order: first-order difference.
93. Velocity source: body-frame features only.
94. Canonical-handshape velocity: excluded.
95. Velocity at the first timestep: zero.
96. Velocity masking: require the part in both adjacent frames.
97. Velocity time offset: one frame.
98. Velocity normalization by elapsed time: disabled.
99. Acceleration features: disabled.
100. Velocity clipping: disabled.
101. Velocity smoothing: disabled.
102. Velocity scaling multiplier: `1.0`.

## 7. Presence-feature hyperparameters

103. Number of presence channels: `4`.
104. Presence categories: right hand, left hand, lips, and pose.
105. Presence test: absolute-coordinate sum greater than zero.
106. Presence threshold: exactly `0`.
107. Presence encoding: binary float.
108. Presence confidence: not used.
109. Missing-landmark interpolation: disabled.
110. Presence history features: disabled.

## 8. Data-augmentation hyperparameters

111. Master training-augmentation switch: enabled.
112. Validation augmentation: disabled.
113. Horizontal-mirror probability: `0.5`.
114. Coordinate-jitter standard deviation: `0.02`.
115. Jitter distribution: independent Gaussian.
116. Jitter mean: `0`.
117. Minimum retained temporal fraction: `0.7`.
118. Maximum retained temporal fraction: `1.0`.
119. Temporal keep-fraction distribution: uniform.
120. Minimum retained frame count: `2`.
121. Temporal frame selection: random without replacement.
122. Selected-frame ordering: sorted chronologically.
123. Post-drop temporal resampling: back to the original 60 steps.
124. Jitter applied to absent landmarks: no.
125. Jitter correlation across coordinates: none.
126. Jitter correlation across frames: none.
127. Jitter correlation across landmarks: none.
128. Mirror hand-slot swapping: enabled.
129. Mirror lip-pair swapping: enabled.
130. Mirror pose-pair swapping: enabled.
131. Mirrored x-coordinate sign inversion: enabled.
132. Rotation augmentation: disabled.
133. Scale augmentation: disabled.
134. Translation augmentation: disabled.
135. Temporal reversal: disabled.
136. Feature dropout or landmark masking: disabled.
137. Mixup: disabled.
138. CutMix: disabled.
139. Label smoothing through augmentation: disabled.

## 9. Loss-function hyperparameters

140. Primary loss: categorical cross-entropy.
141. Class weighting: none.
142. Label smoothing: `0`.
143. Loss reduction: mean.
144. Ignored class index: `-100`, although targets never use it.
145. Auxiliary losses: none.
146. Attention regularization: none.
147. Temporal-consistency loss: none.
148. Focal-loss gamma: not applicable.
149. Distillation loss: none.
150. Metric-learning loss: none.

## 10. Optimizer hyperparameters

151. Optimizer: Adam.
152. Initial learning rate: `5e-4`.
153. Adam beta1: `0.9`.
154. Adam beta2: `0.999`.
155. Adam epsilon: `1e-8`.
156. Weight decay: `1e-4`.
157. Weight-decay formulation: Adam's coupled L2 penalty, not AdamW.
158. AMSGrad: disabled.
159. Gradient maximization: disabled.
160. Differentiable optimizer step: disabled.
161. Parameter-specific learning rates: none.
162. Parameter-specific weight decay: none.
163. Bias exclusion from weight decay: none.
164. LayerNorm exclusion from weight decay: none.
165. Attention-specific learning rate: none.
166. Backbone freezing: disabled.
167. Warm-start checkpoint: disabled.
168. Gradient accumulation steps: `1`.

The command-line option named `momentum` configures Adam beta1; it is not SGD momentum.

## 11. Learning-rate scheduler hyperparameters

169. Scheduler: exponential decay.
170. Per-epoch decay multiplier: `0.99`.
171. Scheduler update interval: once per epoch.
172. Warmup epochs: `0`.
173. Minimum learning rate: none.
174. Restart schedule: none.
175. Plateau monitoring: none.
176. Final learning rate after 50 scheduler steps: approximately `3.03e-4`.
177. Scheduler stepping relative to validation: after validation.
178. Scheduler stepping despite early-stopping state: yes.

## 12. Gradient-management hyperparameters

179. Maximum gradient norm: `5.0`.
180. Gradient norm type: L2.
181. Gradient clipping strategy: global norm.
182. Gradient clipping scope: all model parameters.
183. Non-finite gradient error: disabled by PyTorch default.
184. Gradient clipping enabled: yes.
185. Optimizer gradient-clearing frequency: every batch.
186. Backpropagation frequency: every batch.
187. Truncated backpropagation through time: disabled.
188. Detached recurrent states between windows: not applicable; states restart for every clip.

## 13. Training-duration and early-stopping hyperparameters

189. Maximum epochs: `50`.
190. Early-stopping patience: `10`.
191. Early-stopping monitors: validation loss and validation accuracy.
192. Improvement condition: either loss decreases or accuracy increases.
193. Loss improvement minimum delta: `0`.
194. Accuracy improvement minimum delta: `0`.
195. Comparison strictness: strict `<` for loss and `>` for accuracy.
196. Initial best validation loss: infinity.
197. Initial best validation accuracy: `0`.
198. Patience counter reset: whenever either metric improves.
199. Restore best weights after stopping: no.
200. Checkpoint selection: latest epoch, not best epoch.
201. Checkpoint frequency: every epoch.

Production training can therefore stop after degradation while leaving the final, degraded epoch in `checkpoint.pt`.

## 14. Batch and DataLoader hyperparameters

202. Training batch size: `16`.
203. Validation batch size: `16`.
204. Training shuffle: enabled.
205. Validation shuffle: disabled.
206. Training worker count: `8`.
207. Maximum validation worker count: `2`.
208. Persistent workers: enabled whenever the worker count is nonzero.
209. Pinned host memory: enabled.
210. Nonblocking device transfer: enabled during training.
211. Drop incomplete final batch: disabled.
212. DataLoader prefetch factor: PyTorch default, normally `2` per worker.
213. DataLoader timeout: `0`.
214. Custom sampler: none.
215. Balanced class sampler: none.
216. Batch sampler: standard automatic batching.
217. Lazy per-video loading: enabled.
218. Cache preference: `.npy` if its complete shape matches.
219. Required cache shape: `(60, 408)`.

Worker count, pinning, and persistence affect throughput rather than learned quality, but remain runtime hyperparameters.

## 15. Numerical-precision hyperparameters

220. Device selection: CUDA when available, otherwise CPU.
221. Automatic mixed precision: enabled on CUDA.
222. Automatic mixed precision on CPU: disabled.
223. CUDA autocast dtype: PyTorch default, normally float16.
224. Gradient scaling: enabled with CUDA AMP.
225. Initial gradient scale: PyTorch default, normally `65,536`.
226. Gradient-scale growth factor: default `2.0`.
227. Gradient-scale backoff factor: default `0.5`.
228. Gradient-scale growth interval: default `2,000`.
229. Model parameter storage: float32.
230. Input coordinate storage: float32.
231. Deterministic algorithms: not enabled.
232. cuDNN deterministic mode: not enabled.
233. cuDNN benchmark mode: not explicitly configured.
234. TensorFloat-32 behavior: not explicitly configured.

## 16. Randomness and reproducibility hyperparameters

235. Production Python random seed: unset.
236. Production NumPy seed: unset.
237. Production PyTorch CPU seed: unset.
238. Production CUDA seed: unset.
239. DataLoader worker seed policy: PyTorch-generated.
240. Deterministic augmentation: no.
241. Deterministic batch order: no.
242. Split-generation seed: `42`.

The missing production training seed means otherwise identical runs are not fully reproducible.

## 17. Dataset-split hyperparameters

243. Training fraction: `0.60`.
244. Validation fraction: `0.20`.
245. Test fraction: `0.20`.
246. Split seed: `42`.
247. Split unit: signer group.
248. Signer leakage policy: all clips from one signer stay together.
249. Group assignment algorithm: shuffled greedy proportional fill.
250. Stratification target: global file ratios, not per-class ratios.
251. Rare-class balancing across splits: none.
252. Unknown files during split-aware loading: skipped.
253. Test split use during tuning: prohibited; tuning should use validation only.

Changing the split requires intentionally regenerating `data/split.json`.

## 18. Label-space hyperparameters

254. Vocabulary source: `src/vocab.txt`.
255. Current vocabulary size: `250`.
256. Vocabulary order: file order.
257. Target encoding: one-hot in the dataset.
258. Training target conversion: one-hot `argmax` to class index.
259. Unknown-label handling: error through `vocab.index`.
260. Class hierarchy: none.
261. Synonym merging: none.
262. Blank or background class: none.
263. Unknown or reject class: none.

Vocabulary content and ordering are part of the checkpoint compatibility contract.

## 19. Live-inference hyperparameters

264. Minimum gesture length: `14` detected-hand frames.
265. Maximum gesture length: `384` detected-hand frames.
266. Gesture-start condition: at least one hand present.
267. Gesture-end condition: both hands absent.
268. Overlong-gesture forced classification: enabled at 384 frames.
269. Live resampled length: `60`.
270. Classification batch size: `1`.
271. Confidence threshold: none.
272. Softmax temperature: effectively `1.0`, although argmax is taken directly from logits.
273. Top-k prediction count: `1`.
274. Unknown or rejection threshold: none.
275. Prediction smoothing across gestures: none.
276. Repeated-word suppression: none.
277. Minimum absence gap between gestures: one frame.
278. Camera index: `0` by default.
279. MediaPipe running mode: video.
280. Hand-detection threshold: MediaPipe model default.
281. Pose-detection threshold: MediaPipe model default.
282. Face-detection threshold: MediaPipe model default.
283. Tracking thresholds: MediaPipe defaults.

## 20. Attention experiment hyperparameters

The separate `src/attention_experiments.py` runner does not define the production model, but adds these tunable settings:

284. Pooling variant: baseline.
285. Pooling variant: nonlinear attention.
286. Pooling variant: multihead attention.
287. Pooling variant: attention-weighted statistics.
288. Pooling variant: presence-conditioned attention.
289. Pooling variant: entropy-regularized attention.
290. Nonlinear-attention hidden width: `128`.
291. Nonlinear-attention activation: Tanh.
292. Nonlinear final-score bias: disabled.
293. Multihead attention head count: `4`.
294. Multihead score bias: disabled.
295. Multihead classifier input width: `4 x 384 = 1,536`.
296. Statistics-pooling outputs: weighted mean and standard deviation.
297. Variance floor before square root: `1e-5`.
298. Presence-attention input width: `4`.
299. Presence-score bias: disabled.
300. Attention entropy strength: `0.01`.
301. Entropy numerical clamp: `1e-8`.
302. Entropy normalization: divide by `log(sequence_length)`.
303. Experiment epochs: `15`.
304. Experiment batch size: `64`.
305. Experiment seed: `20260813`.
306. Experiment learning rate: `5e-4`.
307. Experiment gradient norm: `5.0`.
308. Experiment beta1: `0.9`.
309. Experiment beta2: `0.999`.
310. Experiment weight decay: `1e-4`.
311. Experiment scheduler gamma: `0.99`.
312. Best-model criterion: highest validation accuracy, then lowest validation loss.

Current implementation note: the experiment encoder calls `self.relu`, while the production model defines `self.activation`. The experiment variants therefore appear likely to fail before completing a forward pass unless that mismatch is corrected.

## Highest-priority tuning candidates

Tune these before spending resources on lower-impact details:

1. Dropout: try roughly `0.2-0.5`.
2. Hidden width: compare `256`, `384`, and `512`.
3. Recurrent depth: compare `2-4` blocks.
4. Learning rate: search roughly `1e-4-1e-3`.
5. Compare AdamW with the current Adam optimizer.
6. Weight decay: search roughly `1e-6-1e-3`.
7. Batch size: compare `16-128`, constrained by GPU memory.
8. Sequence length: compare `40`, `60`, `80`, and `100`.
9. Coordinate-jitter standard deviation.
10. Temporal minimum-keep fraction.
11. Cross-entropy label smoothing.
12. Class-weighted loss or balanced sampling.
13. Attention-pooling variant.
14. Bidirectional LSTM versus GRU or a temporal Transformer.
15. Presence- and velocity-feature ablations.
16. Explicit training seeds with repeated runs.
17. Saving and restoring the best validation checkpoint.
18. Confidence rejection for live inference.

All tuning decisions should be selected against the signer-disjoint validation split. The held-out test split should remain untouched until the final configuration is chosen. Architecture, feature-layout, vocabulary, or sequence-length changes may require cache migration and/or full checkpoint retraining.
