import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from util.Utils import aeq


class MultiHeaded_Sent_Attention(nn.Module):
	"""
	Multi-Head Attention module from
	"Attention is All You Need"
	:cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

	Similar to standard `dot` attention but uses
	multiple attention distributions simulataneously
	to select relevant items.

	.. mermaid::

	   graph BT
		  A[key]
		  B[value]
		  C[query]
		  O[output]
		  subgraph Attn
			D[Attn 1]
			E[Attn 2]
			F[Attn N]
		  end
		  A --> D
		  C --> D
		  A --> E
		  C --> E
		  A --> F
		  C --> F
		  D --> O
		  E --> O
		  F --> O
		  B --> O

	Also includes several additional tricks.

	Args:
	   head_count (int): number of parallel heads
	   model_dim (int): the dimension of keys/values/queries,
		   must be divisible by head_count
	   dropout (float): dropout parameter
	"""
	def __init__(self, head_count, model_dim, dropout=0.1):
		assert model_dim % head_count == 0
		self.dim_per_head = model_dim // head_count
		self.model_dim = model_dim

		super(MultiHeaded_Sent_Attention, self).__init__()
		self.head_count = head_count

		self.linear_keys = nn.Linear(model_dim,
									 head_count * self.dim_per_head)
		self.linear_values = nn.Linear(model_dim,
									   head_count * self.dim_per_head)
		self.linear_query = nn.Linear(model_dim,
									  head_count * self.dim_per_head)
		self.sm = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)
		self.final_linear = nn.Linear(model_dim, model_dim)
		self.final_linear_2 = nn.Linear(model_dim, model_dim)

	def forward(self, key, value, query, mask=None, return_key=False, all_attn=False):
		"""
		Compute the context vector and the attention vectors.


		Args:
		   key (`FloatTensor`): set of `key_len`
				key vectors `[batch, key_len, dim]`
		   value (`FloatTensor`): set of `key_len`
				value vectors `[batch, key_len, dim]`
		   query (`FloatTensor`): set of `query_len`
				 query vectors  `[batch, query_len, dim]`
		   mask: binary mask indicating which keys have
				 non-zero attention `[batch, query_len, key_len]`
		Returns:
		   (`FloatTensor`, `FloatTensor`) :

		   * output context vectors `[batch, query_len, dim]`
		   * one of the attention vectors `[batch, query_len, key_len]`
		"""

		# CHECKS
		# batch, k_len, d = key.size()
		# batch_, k_len_, d_ = value.size()
		# aeq(batch, batch_)
		# aeq(k_len, k_len_)
		# aeq(d, d_)
		# batch_, q_len, d_ = query.size()
		# aeq(batch, batch_)
		# aeq(d, d_)
		# aeq(self.model_dim % 8, 0)
		# if mask is not None:
		# 	batch_, q_len_, k_len_ = mask.size()
		# 	aeq(batch_, batch)
		# 	aeq(k_len_, k_len)
		# 	aeq(q_len_ == q_len)
		# END CHECKS

		batch_size = key.size(0) #2
		dim_per_head = self.dim_per_head #768/n_head
		head_count = self.head_count
		key_len = key.size(1)
		query_len = query.size(1)

		def shape(x):
			return x.view(batch_size, key_len, key_len, head_count, dim_per_head) \
				.transpose(1, 2)

		def shape_q(x):
			return x.view(batch_size, query_len, head_count, dim_per_head)

		def unshape(x):
			return x.transpose(1, 2).contiguous() \
					.view(batch_size, key_len, head_count * dim_per_head)

		# 1) Project key, value, and query.
		key_up = shape(self.linear_keys(key))
		value_up = shape(self.linear_values(value))

		query_up = shape_q(self.linear_query(query))
		query_up = query_up.unsqueeze(1)

		key_up = key_up.transpose(1, 3).transpose(3, 4)

		# 2) Calculate and scale scores.
		query_up = query_up / math.sqrt(dim_per_head)
		scores = torch.matmul(query_up.transpose(1,3), key_up) #(2, 1, 14, 1, 768 )(2, 1, 14, 768, 14)->(2, 1, 14, 1, 14)

		if mask is not None:
			# mask = mask.unsqueeze(1).expand_as(scores)
			# scores = scores.masked_fill(Variable(mask), -1e18)
			mask = mask.unsqueeze(1)
			mask = mask.unsqueeze(3)
			scores = scores * mask

		# 3) Apply attention dropout and compute context vectors.
		attn = self.sm(scores)

		drop_attn = self.dropout(attn)
		#reconstruct shape
		context = unshape(torch.matmul(drop_attn, value_up.transpose(1, 3)))
		#(2, max_len, 1, 1, 768)

		# context = unshape(torch.matmul(drop_attn, value_up))

		output = self.final_linear(context).squeeze(1)

		#(2, max_len, 1, 768)

		# batch_, q_len_, d_ = output.size()

		# aeq(q_len, q_len_)
		# aeq(batch, batch_)
		# aeq(d, d_)

		# END CHECK
		return output

class MultiHeaded_Token_Attention(nn.Module):
	"""
	Multi-Head Attention module from
	"Attention is All You Need"
	:cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

	Similar to standard `dot` attention but uses
	multiple attention distributions simulataneously
	to select relevant items.

	.. mermaid::

	   graph BT
		  A[key]
		  B[value]
		  C[query]
		  O[output]
		  subgraph Attn
			D[Attn 1]
			E[Attn 2]
			F[Attn N]
		  end
		  A --> D
		  C --> D
		  A --> E
		  C --> E
		  A --> F
		  C --> F
		  D --> O
		  E --> O
		  F --> O
		  B --> O

	Also includes several additional tricks.

	Args:
	   head_count (int): number of parallel heads
	   model_dim (int): the dimension of keys/values/queries,
		   must be divisible by head_count
	   dropout (float): dropout parameter
	"""
	def __init__(self, head_count, model_dim, dropout=0.1):
		assert model_dim % head_count == 0
		self.dim_per_head = model_dim // head_count
		self.model_dim = model_dim

		super(MultiHeaded_Token_Attention, self).__init__()
		self.head_count = head_count

		self.linear_keys = nn.Linear(model_dim,
									 head_count * self.dim_per_head)
		self.linear_values = nn.Linear(model_dim,
									   head_count * self.dim_per_head)
		self.linear_query = nn.Linear(model_dim,
									  head_count * self.dim_per_head)
		self.sm = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)
		self.final_linear = nn.Linear(model_dim, model_dim)
		self.final_linear_2 = nn.Linear(model_dim, model_dim)


	def forward(self, key, value, query, mask=None, return_key=False, all_attn=False):
		"""
		Compute the context vector and the attention vectors.


		Args:
		   key (`FloatTensor`): set of `key_len`
				key vectors `[batch, key_len, dim]`
		   value (`FloatTensor`): set of `key_len`
				value vectors `[batch, key_len, dim]`
		   query (`FloatTensor`): set of `query_len`
				 query vectors  `[batch, query_len, dim]`
		   mask: binary mask indicating which keys have
				 non-zero attention `[batch, query_len, key_len]`
		Returns:
		   (`FloatTensor`, `FloatTensor`) :

		   * output context vectors `[batch, query_len, dim]`
		   * one of the attention vectors `[batch, query_len, key_len]`
		"""

		batch_size = key.size(0)  # 2
		dim_per_head = self.dim_per_head  # 768/n_head
		head_count = self.head_count
		key_len_1 = key.size(1)
		ken_len_2 = key.size(3)
		query_len = query.size(1)

		def shape(x):
			return x.view(batch_size, key_len_1, key_len_1, ken_len_2, head_count, dim_per_head) \
				.transpose(1, 2)

		def shape_q(x):
			return x.view(batch_size, query_len, head_count, dim_per_head)

		def unshape(x):
			return x.transpose(1, 2).contiguous() \
				.view(batch_size, key_len_1, key_len_1, head_count * dim_per_head)

		# 1) Project key, value, and query.
		key_up = shape(self.linear_keys(key))
		value_up = shape(self.linear_values(value))

		query_up = shape_q(self.linear_query(query))
		query_up = query_up.unsqueeze(1).expand(batch_size, query_len, query_len, head_count, dim_per_head).unsqueeze(3)



		key_up = key_up.transpose(3, 5).transpose(3, 4)

		# 2) Calculate and scale scores.
		query_up = query_up / math.sqrt(dim_per_head)
		scores = torch.matmul(query_up,
							  key_up)  # (2, 1, 14, 1, 768 )(2, 1, 14, 768, 14)->(2, 1, 14, 1, 14)
		# score (2, 17, 17, 1, 23)
		# value (2, 17, 17, 23, 768)

		if mask is not None:
			# mask = mask.unsqueeze(1).expand_as(scores)
			# scores = scores.masked_fill(Variable(mask), -1e18)
			mask = mask.unsqueeze(1)
			mask = mask.unsqueeze(3)
			scores = scores * mask

		# 3) Apply attention dropout and compute context vectors.
		attn = self.sm(scores)

		drop_attn = self.dropout(attn)
		# reconstruct shape
		# context = unshape(torch.matmul(drop_attn, value_up.transpose(1, 3)))
		context = unshape(torch.matmul(drop_attn, value_up.transpose(3,4))) #(2, 17, 17, 1, 768)(2, 17, 17, 768)

		# (2, max_len, 1, 1, 768)

		# context = unshape(torch.matmul(drop_attn, value_up))

		output = self.final_linear(context).squeeze(1)

		# (2, max_len, 1, 768)

		return output