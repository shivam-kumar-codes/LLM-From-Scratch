import torch


inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)


query = inputs[1]  # 2nd input token is the query

# step 1: calculating unnormalized attention scores
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)


# step 2: normalizing attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# However, in practice, using the softmax function for normalization, which is better at handling extreme values and has more desirable gradient properties during training, is common and recommended.

def softmax_naive(x) :
    return torch.exp(x) / torch.exp(x).sum(dim = 0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention wights:", attn_weights_2_naive)
print("sum:", attn_weights_2_naive.sum())

# The naive implementation above can suffer from numerical instability issues for large or small input values due to overflow and underflow issues
# Hence, in practice, it's recommended to use the PyTorch implementation of softmax instead, which has been highly optimized for performance:

attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)

print("Attention weights:", attn_weights_2)
print("sum:", attn_weights_2.sum())


# step 3: Step 3: compute the context vector by multiplying the embedded input tokens, with the attention weights and sum the resulting vectors:

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)


# ------------------------------------------Generalizing now for all input tokens -----------------------------------------------

attn_scoress = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scoress[i, j] = torch.dot(x_i, x_j)

print(attn_scoress)

# we can acheieve the same by matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

# similar to step 2 previously, we normalize each row so that each row sum up to 1
attn_weights = torch.softmax(attn_scores, dim = -1)
print("Attention weights:", attn_weights)

# Quick verificaiton that the vaules in each row sums up to 1
row2_sum = sum([0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565])
print("Row 2 Sum:", row2_sum)
print("All rows sums:", attn_weights.sum(dim = -1))

# Step 3: to compute all context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)