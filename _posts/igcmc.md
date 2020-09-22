---
title: 'Graph Based Recommender Systems: iGC-MC'
date: 17-09-2020
---


This chapter presents the main model and contribution of this thesis, which adapts an advanced method for explicit feedback, graph convolutional matrix completion (GC-MC) \cite{berg2017graph} to the implicit feedback setting. GC-MC views the problem of matrix completion on our observation matrix $R$ from the point of view of link prediction on graphs. We use a graph-based auto-encoder framework that produces latent features of user and item nodes through a form of message passing on the bipartite interaction graph. As with our previous methods, these latent factors are then used to reconstruct the full dense rating links, through a bilinear decoder. We refer to our method in what follows as implicit graph convolutional matrix completion (iGC-MC).

## Preliminaries
Consider our observation matrix $R$ of size $N \times M$ which, as before, we split into the matrix of binary values $P$ and its corresponding confidence matrix $C$, where we have applied an appropriate logarithmic or linear transformation (Eq:\ref{eq:linear_transform}-\ref{eq:log_transform}). We can represent this information as an undirected graph $G = (\mathcal{W},\mathcal{E})$ with entries as user nodes $n_u \in \mathcal{U}$ with $u\in\{1,...,N\}$ and item nodes $n_i \in \mathcal{V}$ with $i\in\{1,...,M\}$, such that $\mathcal{U}\cup \mathcal{V}=\mathcal{W}$. The edges $(n_u,c_{ui},n_i) \in \mathcal{E}$ have weights corresponding exactly to the confidence $c_{ui} \in \mathbb{R}^{\geq0}$  of our observation $p_{ui}\in \{0,1\}$, interpreted as a positive `count' of how many times user $u$ has interacted with item $i$. Of importance is that unobserved user-item pairs ($r_{ui}=0$) are absent in this graph but do indeed factor into the method at a later stage. Note that the adjacency matrix of this graph $A$ is exactly our original observed ratings matrix $R$, we use these terms interchangeably going forward. 

In its original formulation for explicit feedback, the undirected graph $G=(\mathcal{W},\mathcal{E},\mathcal{R})$ had a third attribute $\mathcal{R}=\{1,...,r_{max}\}$. These were the labels of each edge, which represented the ordinal rating levels such as movie ratings from $\{1,...,5\}$. For the implicit feedback setting we have one single ordinal category: $\mathcal{R}=1$, indicating whether or not that edge represents an interaction between user $u$ and item $i$.

Graph auto-encoders in general comprise of an \textit{encoder} $Z = f(O,A)$ that has as input a feature matrix $O$ of size $n \times d$, with $n$ the number of nodes and $d$ number of features, as well as an adjacency matrix $A$ to produce a node embedding matrix $Z=[z_1^T,...,z_N^T]^T$ of size $n \times f$, where $f$ is the latent dimension size. This is followed by a pairwise \textit{decoder} model $\hat{A}=g(Z)$ which takes a pair of node embeddings $(z_i,z_j)$ and predicts respective entries in the new adjacency matrix $\hat{A}$.

For a bipartite implicit recommender system $G = (\mathcal{W},\mathcal{E})$ we reformulate the above as follows. The \textit{encoder} is $[X,Y]=f(O,A)$ where $A \in \mathbb{R}^{N\times M}$ is the now the weighted adjacency matrix, with entries corresponding to confidence levels. The matrices $X$ and $Y$ correspond to user and item embeddings and are of shape $N \times f$ and $M \times f$, with $f$ the embedding size. \textit{Decoder}: $\hat{A}=g(X,Y)$, a function acting on the user and item embeddings and returning a reconstructed rating matrix $\hat{A}$ of shape $N\times M$.


## Graph convolution encoder
The encoder of our model uses a graph convolutional layer to perform local operations that gather information from the first-order neighbourhood of a node. These are a form of message-passing, where messages that are vector-valued are passed and transformed across edges of the graph. 

First the message from each node is formed, where a transformation is applied to the initial node vectors, which is the same across all locations (nodes) in the graph. Each message $\mu_{a \to b} \in \mathbb{R}^h$ has dimension $h$, a hyperparameter referred to as the hidden dimension size. Messages take the following form, here from item node $i$ to user node $u$:
\begin{align}
    \mu_{i \to u} = Wo_i 
\end{align}
where $W \in \mathbb{R}^{h \times d}$ is a parameter matrix that is to be learned during training, which maps the initial feature vector $o_i \in \mathbb{R}^d$ into $\mu_{i \to u} \in \mathbb{R}^h$. Messages from users to items $\mu_{u \to i}$ are calculated in an analogous way.

After the message-passing step, we sum the incoming messages to each node, multiplying each message by weight $\bar{c}_{ui}$ to result in a single intermediate vector denoted $h_i$, that represents a weighted average of messages:
\begin{align}
    h_u = \sigma \big( \sum_{i \in \mathcal{N}_u} \bar{c}_{ui} \mu_{i \to u} \big) 
Here $\mathcal{N}_u$ denotes the neighbours of node $n_u$ and $\sigma(\cdot)$ denotes an element-wise non-linear activation function chosen to be $ReLU(\cdot)=\max(0,\cdot)$. We normalise each message with respect to the in-degree of each node where each message is weighed by the edge label (i.e. the confidence parameter $c_{ui}$ associated with the observation between user $u$ and $i$) :
\begin{align}
    \bar{c}_{ui} = \frac{c_{ui}}{\sum_{i \in \mathcal{N}_u} c_{ui}} \label{eq:gcmc_confidence}
\end{align}
To arrive at the final embedding of user node $u$ we passs the intermediate output $h_u$ through a fully-connected layer:
\begin{align}
    x_u = \sigma(W'h_u) \label{eq:gcmc_dense_hidden_layer}
\end{align}
where $W' \in \mathbb{R}^{f \times h}$ is a separate parameter matrix to be learnt during training. The item embedding $y_i$ is calculated analogously with the same parameter matrix $W'$ or if side-information is included we use separate parameter matrices for user and item nodes, further detailed in Section \ref{sec:gcmc_side_info}. Eq.\ref{eq:gcmc_conv_layer} is referred to as as the \textit{graph convolutional layer} and Eq??? as the \textit{dense} layer.

We mention here, as in \cite{berg2017graph}, that instead of a simple linear message transformation \ref{eq:gcmc_message}, variations are possible such as $\mu_{i \to u}=nn(o_u,o_i)$ where $nn$ is a neural network itself. In addition, instead of choosing a specific normalisation constant for individual messages, one could deploy some a form of attention mechanism, explored in some recent work \cite{yang2020hagerec, gong2020attentional}.

## Bilinear decoder
To reconstruct links in the bipartite interaction graph the decoder produces a probability of user $u$ and item $i$ interacting as follows:
\begin{align}
    \hat{a}_{ui} = p(A_{ui} > 0) = \sigma(x_u^T Q y_i) \label{eq:gcmc_output}
\end{align}
where $Q$ is a trainable parameter matrix of shape $f \times f$. Here $\sigma(x)=1/(1+e^{-x})$ is the usual sigmoid function that maps the bilinear form into $[0,1]$ so that we gain a probabilistic interpretation of our output.

## Model training
We train this graph auto-encoder by minimising the log likelihood of the predicted ratings $\hat{A}_{ij}$. As previously noted in Section \ref{sec:background_implicit_feedback}, unlike with explicit feedback, our loss is not only calculated over observed user-item pairs in $\mathcal{O}$. In the implicit feedback setting we need to account for the binary nature of interactions by sampling a number of negative user-item pairs from $\mathcal{O}^-$, as was done for training \textit{LMF} (Section \ref{sec:lmf}). The number of negative samples per positive sample is an integer hyperparameter $c$. We call this combined set of positive and negative samples $\mathcal{S}$. Our objective function, equivalent to a binary cross-entropy loss, which we seek to minimise is:
\begin{align}
    \mathcal{L} &= -\sum_{u,i \in \mathcal{S}} p_{ui} \log{\hat{a}_{ui}} + (1-p_{ui})\log{(1-\hat{a}_{ui})}
\end{align}
where $p_{ui} \in \{0,1\}$ is the true interaction between user $u$ and item $i$, whilst $\hat{a}_{ui}$ is our model's output - a probability of interaction - given by Eq.\ref{eq:gcmc_output}.

## Node dropout
As a form of regularisation we randomly drop out all outgoing messages of a particular node with probability $p_{\text{dropout}}$ , which we refer to as \textit{node dropout} as in \cite{berg2017graph}. We also employ regular dropout \cite{srivastava2014dropout} in the fully connected layers of our model. The original GC-MC authors note that node dropout served as a better regulariser than message dropout, whereby outgoing  messages are dropped out independently making the model more robust to the absence of single edges that represent a single user-item observation.


## Batching
In order for the full \textit{Kindred} dataset to fit into memory, batching is necessary to train the model, also serving as a means of regularisation. Mini-batching is done at \textit{user node level} - at each epoch, we split the full set of users $\mathcal{U}$ into batches $\mathcal{U}_1,...,\mathcal{U}_K$ where the loss function for each batch is calculated over all user-item pairs $(u,i)$ such that $u \in \mathcal{U}_k, i \in \mathcal{V}$. In essence, we remove respective rows of our observation matrix $R$, splitting it into $K$ submatrices, and in doing so perform an equivalent to node dropout on the incoming messages to the item nodes. Where previously without batching the sum in 
Eq.\ref{eq:gcmc_batch_approx} would be taken over all $u \in \mathcal{N}_i$, we now have:
\begin{align}
    h_i \approx \sigma \Big( \sum_{u \in \mathcal{N}_i: \; u \in \mathcal{U}_k} \bar{c}_{ui} \mu_{u \to i} \Big) 
    \label{eq:gcmc_batch_approx}
\end{align}

## Incorporating side information
With our problem setting abstracted into the non-Euclidean domain of a graph, we find a natural way to include a feature vector of side information for each node directly into the input feature matrix. However, as noted by in the original GC-MC paper, when the content information is not rich enough to distinguish different users (or items), this leads to a bottleneck of information flow. Thus, in our case, with no user side information and only limited item side information (in the form of \textit{Item2Vec} embeddings), we opt for a different method to include these features that uses a separate processing channel directly into the dense hidden layer.

Suppose item node $i$ has side information vector $y_i^f$. We pass $y_i^f$ through a dense layer to output a vector $f_i \in \mathbb{R}^m$, where $m$ is a hyperparameter. We then concatenate $f_i$ with the intermediate node embedding $h_i$ (as in Eq.\ref{eq:gcmc_conv_layer}), then passed through a separate dense layer to produce our final node embedding $y_i \in \mathbb{R}^f$. Where previously, as in  Eq.\ref{eq:gcmc_dense_hidden_layer}, we had $y_i = \sigma(W'h_i)$, this now becomes:
\begin{align}
    y_i = \sigma(W_2[h_i, f_i]) \;\;\; \text{with} \;\;\; f_i = \sigma(W_1 y_i^f + b)  \label{eq:gcmc_side_info_eq}
\end{align}
with $W_1, W_2, b$ all trainable weight matrices. User nodes, without any side information, are processed through the same dense layer as before Eq.\ref{eq:gcmc_dense_hidden_layer} with a different parameter matrix than the one used above for the item nodes. In the presence of side information, the initial input feature matrix $O$ is simply chosen as an identity matrix, with a unique one-hot vector for every node in the graph.

We call the method which integrates our \textit{Item2Vec} embeddings into the implicit graph convolutional matrix completion method \textit{iGC-MC + Item2Vec} for the remainder of this thesis.

## Our contributions
For clarity, we outline the changes and contributions made in this work to adapt the original GC-MC method \cite{berg2017graph} to the implicit feedback setting. These are as follows:
- Using a single edge type and processing channel to model whether that edge represents an interaction between the two nodes it connects (one user, one item). Previously each rating level was given its own processing channel.
- Weigh messages according to their confidence (Eq.\ref{eq:gcmc_confidence}), as given by the weighted average of edges weights incoming to that node.
- Loss function. We change the model output from a score per rating level to a single scalar output, passed through a sigmoid nonlinearity so as to interpret it as the probability of interaction. Accordingly, our loss function became a binary cross-entropy loss vs what was previously a cross-entropy loss, where a softmax had been applied in the original presentation of the method.
- Negative interactions. We sample a number of negative unobserved user-item pairs to contribute to the loss function. Learning would not be possible without this step, since the label of all items contributing to the loss would be positive. These unobserved user-item pairs correspond to `empty' edges on the graph.
