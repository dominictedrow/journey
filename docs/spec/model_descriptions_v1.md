# Model Descriptions v1 (Dom, 2026-11-26)

> This is Dom's own text, copied word for word with only the line breaks tidied. **Never edit it.** It sits next to `system_spec_v2.md` as a second source document.
> Scope: every model component **except** the execution algo and the rules algo.
> It is turned into build specs by the questioning process in `docs/plan/MODEL_PLAN.md`. Where the component docs or DECISIONS.md disagree with it, they win.

---

Cool Data Transforms To Experiment With At Data Collection Time:
Could map all data points to a circle by creating a circle for all of my training data on that data point then applying the new data point to that circle
All circles will then be combined and converted into a tensor
Scalers:
Can scale all other numerical values off of one important number like GDP or S&P price
Can also scale certain values based on correlations between one another, for example the S&P scaled against whatever NQ is at the time
All percentages can be scaled according to the price levels or vice versa
Depending on the market regime the numbers could be scaled
Could apply some fibonacci model to the data
Could have each number correspond to a frequency, then feed the noise all of the frequencies make into an audio encoder
I could transform the data into art by having each number generate a splotch of color based on what level it is at
I would then feed the entire canvas into visual encoder and use that as the input tensor
Could randomize the word order
Could only use letters
Could translate everything into spanish or mandarin

General:
All operations should be fused
All models should use mixed precision training
All models should use k-fold training
All models should use the Adam optimizer
All should use TP where applicable
All should use PP where applicable (experiment with this one)
All numbers should have z-score normalization applied to them
Gradient clipping on every model
Dropout on every model
Data augmentation on every model
L2 regularization
Temporal augmentation
Speculative decoding
Maybe should use other custom kernels to make sure memory isn't a bottleneck because at 16 gb of ram could very easily be a problem

Tokenizer:

SVM:
Initializer: Kaiming normal
N-gram should be made from all training
Classifier should classify whether the article is relevant or not
Should use a Relu activation
Should then update the database with whatever became classified
Just like append a valid or non valid tag to each article
MSE loss function

LLM:
I want this model to intake literally all of the data from the data collection pipeline
I want to implement the PagedAttention format of attention
I want it to be an MoE
Just 2 one that outputs the passage digestion
Both should intake all of the data but the price manipulation one should intake what the language one outputs
I want it to be 8 heads of attention and have 16 layers
256 embedding dim
When the articles are being passed in number data should zeroed
Initializer: for the MLP parts kaiming and then the default init for the attention blocks
Cross entropy loss
Attention Residual

Transformer:
Intakes number related data including the raw data collection and the LLM output for words and numbers
256 embed dim
Only outputs a vote on bullish or bearish
Should use flash attention this time
Should only be 8 layers with 4 heads
Initializer: for the MLP parts kaiming and then the default init for the attention blocks
Sigmoid activation function
Cross entropy loss
Attention residual

MLP:
Init: Xavier_normal
Emded dim 256
Intakes all number related data from the LLM and not
Only outputs a vote on bullish or bearish
6 layers
Cross entropy loss
LayerNorm
tanh activations

CNN #1:
Init: Xavier_normal
Intakes all of the price data from the LLM
4 layers of feature map, activations, and then pooling
Spatial pyramid pooling
Sigmoid activation for the classification layer on the vote
LayerNorm
Data: I want a table that arranges the number data along the rows and then has columns for days
Cross entropy loss

CNN #2:
Init: Xavier_normal
Intakes LLM word output
4 layers of feature map, activations, and then pooling
Spatial pyramid pooling
Sigmoid activation for the classification layer on the vote
LayerNorm
Data: I want a table that has the stocks on rows and then the keyword pairs as columns
Cross entropy loss

Meta Learner:
Initializer: for the MLP parts kaiming and then the default init for the attention blocks
Cross entropy loss
12 layers deep of paged attention with 8 attention heads
Intakes all 4 votes and then all data from the LLM output as well as raw numbers
Embed dim of 256
Output Dim of 500 for every stock in S&P 500 but screened afterward on the screened list I made that week
LayerNorm
Relu activation

Market Transformer:
Initializer: for the MLP parts kaiming and then the default init for the attention blocks
Cross entropy loss
6 layers with 16 attention heads
Intakes all raw data from the data collection aside from the words those are from the LLM and the LLM price mods, Also intakes the order book for whatever ticker is being evaled which is also passed in
Should then output the price, the ticker, and buy or sell as a tuple
Embed dim of 256
Output dim of 1 since it's outputting a number then appending it to a tuple
LayerNorm
