import math
import numpy as np
import torch
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, n_token = 77, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.n_token = n_token
        
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        
        self.div = torch.sqrt(torch.tensor(self.d_head, dtype = self.query.weight.dtype))
        
        self.softmax = torch.nn.Softmax(dim = -1)
        self.dropout = torch.nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query.weight)
        torch.nn.init.xavier_uniform_(self.key.weight)
        torch.nn.init.xavier_uniform_(self.value.weight)
        torch.nn.init.xavier_uniform_(self.proj.weight)
        
        torch.nn.init.constant_(self.query.bias, 0.)
        torch.nn.init.constant_(self.key.bias, 0.)
        torch.nn.init.constant_(self.value.bias, 0.)
        torch.nn.init.constant_(self.proj.bias, 0.)
        
    def forward(self, q, k, v, mask = None, weight = None, alpha = None):
        b, s = q.shape[:2]
        b2, s2 = k.shape[:2]
        
        q = self.query(q) #b, s, f
        k = self.key(k) #b, s, f
        v = self.value(v) #b, s, f
        
        q = q.view(-1, s, self.n_head, self.d_head).transpose(1, 2) #b, h, s, hf
        k = k.view(-1, s2, self.n_head, self.d_head).transpose(1, 2) #b, h, s, hf
        v = v.view(-1, s2, self.n_head, self.d_head).transpose(1, 2) #b, h, s, hf
        
        score = torch.matmul(q, k.transpose(-2, -1)) / self.div #b, h, s, s

        if mask is not None:
            mask = mask.unsqueeze(1) #b, 1, s
            if mask.dim() != score.dim():
                mask = mask.unsqueeze(2) #b, 1, 1, s
            score = score * mask
            
        if weight is not None:
            weight = weight.unsqueeze(1) #b, 1, s
            if weight.dim() != score.dim():
                weight = weight.unsqueeze(2) #b, 1, 1, s
        if self.n_token == s2:
            w = self.softmax(score) #b, h, s, s2
            if weight is not None:
                w = w * weight
                w = w / (w.sum(dim = -1, keepdim = True) + 1e-12)
        else:
            target, ref = torch.split(score, [self.n_token, s2 - self.n_token], dim = -1)
            target = self.softmax(target)
            if alpha is None:
                alpha = 0.5
            if weight is not None:
                ws = weight.shape[-1]
                target_weight, ref_weight = torch.split(weight, [self.n_token, ws - self.n_token], dim = -1)
                ref = ref.view(b2, self.n_head, s, ws - self.n_token, self.n_token)
                ref = self.softmax(ref)
                ref = ref * ref_weight.unsqueeze(-1)
                ref = ref.view(b2, self.n_head, s, s2 - self.n_token)
                ref = alpha * (ref / (ref.sum(dim = -1, keepdim = True) + 1e-12))
                target = target * (1 - alpha) * target_weight
            w = torch.cat([target, ref], dim = -1)
            w = w / (w.sum(dim = -1, keepdim = True) + 1e-12)
        w = self.dropout(w)
        
        out = torch.matmul(w, v) #b, h, s, hf
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model) #b, s, d
        out = self.proj(out)
        return out
    
class QuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, emb_dim, n_head, ff_dim, n_token = 77, activation = "quick_gelu", dropout = 0.1):
        super().__init__()
        self.attn = MultiheadAttention(emb_dim, n_head, n_token = n_token, dropout = dropout)
        if activation.lower() == "gelu" or activation is None:
            self.act = torch.nn.GELU()
        elif activation.lower() == "relu":
            self.act = torch.nn.ReLU()
        elif activation.lower() == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = activation
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ff_dim),
            self.act,
            torch.nn.Linear(ff_dim, emb_dim),
        )
        self.norm1 = torch.nn.LayerNorm(emb_dim)
        self.norm2 = torch.nn.LayerNorm(emb_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ff[0].weight)
        torch.nn.init.xavier_uniform_(self.ff[2].weight)
    
        torch.nn.init.constant_(self.ff[0].bias, 0.)
        torch.nn.init.constant_(self.ff[2].bias, 0.)
        
    def forward(self, x, context = None, mask = None, weight = None, alpha = None):
        context = context if context is not None else x
        out = self.attn(x, context, context, mask = mask, weight = weight, alpha = alpha)
        out = x + self.dropout1(out)
        out = self.norm1(out)
        
        ff_out = self.ff(out)
        out = out + self.dropout2(ff_out)
        out = self.norm2(out)
        return out

class PersonalizedAdapter(torch.nn.Module):
    def __init__(self, emb_dim, n_head, ff_dim, n_layer = 4, n_token = 77, proj = False, extra_proj = False, pos = True, cls_pos = False, cls_token = True, encode_ratio = None, activation = "quick_gelu", dropout = 0.1):
        super().__init__()
        self.n_layer = n_layer
        self.n_token = n_token
        self.cls_pos = cls_pos
        self.cls_token = cls_token
        self.encode_ratio = encode_ratio
        
        self.pre_proj = self.post_proj = None
        if encode_ratio and encode_ratio != 1:
            self.pre_proj = torch.nn.Linear(emb_dim, int(emb_dim // encode_ratio))
            self.post_proj = torch.nn.Linear(int(emb_dim // encode_ratio), emb_dim)
            emb_dim = int(emb_dim // encode_ratio)
            n_head = int(n_head // encode_ratio)
        
        if activation.lower() == "gelu" or activation is None:
            self.act = torch.nn.GELU()
        elif activation.lower() == "relu":
            self.act = torch.nn.ReLU()
        elif activation.lower() == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = activation
        self.base_query = torch.nn.Parameter(torch.empty(1, n_token + int(cls_token), emb_dim))
        self.pos = torch.nn.Parameter(torch.empty(1, n_token + int(cls_pos and cls_token), emb_dim)) if pos else None
        self.init_query = None
        
        self.proj = None
        if proj:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, ff_dim),
                self.act,
                torch.nn.Linear(ff_dim, emb_dim),
            )
        
        self.extra_proj = None
        self.tf = torch.nn.ModuleList([TransformerBlock(emb_dim, n_head, ff_dim, n_token = n_token, activation = activation, dropout = dropout) for _ in range(n_layer)])
        if extra_proj:
            self.extra_proj = torch.nn.ModuleList([torch.nn.Linear(emb_dim, emb_dim) for _ in range(n_layer)])
            
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.base_query, std = 0.02)
        if self.pos is not None:
            torch.nn.init.normal_(self.pos, std = 0.01)
            
        for proj in [self.pre_proj, self.post_proj]:
            if proj is not None:
                torch.nn.init.xavier_uniform_(proj.weight)
                torch.nn.init.constant_(proj.bias, 0.)
        for proj in [self.proj]:
            if proj is not None:
                torch.nn.init.xavier_uniform_(proj[0].weight)
                torch.nn.init.xavier_uniform_(proj[2].weight)

                torch.nn.init.constant_(proj[0].bias, 0.)
                torch.nn.init.constant_(proj[2].bias, 0.)
        if self.extra_proj is not None:
            for l in self.extra_proj:
                torch.nn.init.xavier_uniform_(l.weight)
                torch.nn.init.constant_(l.bias, 0.)
    
    def set_base_query(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=self.base_query.dtype).to(self.base_query.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.init_query = x
            
    def normal_forward(self, x, context, mask = None, weight = None, alpha = None):
        out = x
        for i in range(self.n_layer):
            if self.extra_proj is not None:
                _context = self.extra_proj[i](self.act(context))
            else:
                _context = context
            out = self.tf[i](out, _context, mask = mask, weight = weight, alpha = alpha) #n, b, f
        if self.cls_token:
            return out[:, :-1], out[:, -1]
        else:
            return out, None
    
    def forward(self, context, mask = None, weight = None, alpha = None, base_query = None):
        dtype = self.base_query.dtype
        if base_query is not None:
            x = base_query
        else:
            x = self.base_query if self.init_query is None else self.init_query
        x = x.type(dtype)
        if context is not None:
            context = context.type(dtype)
        if weight is not None:
            weight = weight.type(dtype)
        if self.encode_ratio is not None and x.shape[-1] != self.base_query.shape[-1]:
            x = self.pre_proj(x)
        if self.n_token < x.shape[1]:
            x, cls = x[:, :self.n_token], x[:, self.n_token:]
        else:
            cls = self.base_query[:, self.n_token:] if self.cls_token else None
        if self.pos is not None:
            if self.cls_pos and self.cls_token:
                x = x + self.pos[:, :self.n_token]
                if cls is not None:
                    cls = cls + self.pos[:, self.n_token:]
            else:
                x = x + self.pos
        if self.cls_token:
            x = torch.cat([x, cls], dim = 1)
        x = x.repeat_interleave(context.shape[0], dim = 0)
        if self.encode_ratio is not None:
            if context is not None:
                context = self.pre_proj(context)
        if self.proj is not None:
            context = self.proj(context)
        out = self.normal_forward(x, context, mask = mask, weight = weight, alpha = alpha)
        if self.encode_ratio is not None:
            out = (self.post_proj(out[0]), self.post_proj(out[1]) if out[1] is not None else out[1])
        return out

class DrUM:
    def __init__(self, model, processor, n_layer = 8, proj = False, extra_proj = False, mlp_ratio = 4, pos = True, cls_pos = False, cls_token = True, encode_ratio = None, max_token_size = 256, activation = "quick_gelu", dropout = 0.1):
        config = model.config.text_config if hasattr(model.config, "text_config") else model.config
        if hasattr(config, "model_type") and config.model_type == "t5":
            self.d_model = config.d_model
            self.n_head = config.num_heads
            self.n_token = min(processor.model_max_length, max_token_size)
            self.clip = False
            self.cls_token = False
        else:
            self.d_model = config.hidden_size
            self.n_head = config.num_attention_heads
            self.n_token = config.max_position_embeddings
            self.clip = True
            self.cls_token = cls_token
        self.n_layer = n_layer
        self.proj = proj
        self.extra_proj = extra_proj
        self.mlp_ratio = mlp_ratio
        self.pos = pos
        self.cls_pos = cls_pos
        self.encode_ratio = encode_ratio
        self.activation = activation
        self.dropout = dropout
        
        self.model = model
        self.processor = processor
        self.adapter = PersonalizedAdapter(self.d_model, self.n_head, self.d_model // mlp_ratio, n_layer, self.n_token, proj = proj, extra_proj = extra_proj, pos = pos, cls_pos = cls_pos, cls_token = self.cls_token, encode_ratio = encode_ratio, activation = activation, dropout = dropout).to(model.device)
        
        self.train()
        self.to(model.device)
        
    def preprocess(self, text = None, image = None, return_tensors = "pt", padding = "max_length", truncation = True, **kwargs):
        feed = {"text":([text] if np.ndim(text) == 0 else list(text)) if text is not None else None,
                "return_tensors":return_tensors, 
                "max_length":self.n_token,
                "padding":padding, 
                "truncation":truncation,
                **kwargs}
        if not self.clip:
            feed["add_special_tokens"] = True
        if image is not None:
            feed["images"] = image
        return self.processor(**feed)
    
    def pool_text_hidden_state(self, hidden_state, x, padding = "max_length", truncation = True, **kwargs):
        if not self.clip:
            raise TypeError("T5 encoder does not support this function (pool_text_hidden_state).")
        if not hasattr(x, "items"):
            x = self.preprocess(text = x, padding = padding, truncation = truncation, **kwargs)
        if self.model.text_model.eos_token_id == 2:
            out = hidden_state[torch.arange(hidden_state.shape[0], device = hidden_state.device),
                              x["input_ids"].to(dtype = torch.int, device = hidden_state.device).argmax(dim = -1),]
        else:
            out = hidden_state[torch.arange(hidden_state.shape[0], device = hidden_state.device),
                              (x["input_ids"].to(dtype = torch.int, device = hidden_state.device) == self.model.text_model.eos_token_id).int().argmax(dim = -1),]
        return out
    
    def normalize_text_hidden_state(self, hidden_state):
        out = self.model.text_model.final_layer_norm(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model.text_model, "final_layer_norm") else hidden_state
        return out
    
    def projection_text_hidden_state(self, hidden_state):
        out = self.model.text_projection(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model, "text_projection") else hidden_state
        return out
    
    def encode_prompt(self, x, pooling = True, skip = -1, skip_pool = None, padding = "max_length", truncation = True, use_attn_mask = False, normalize = True, normalize_pool = True, **kwargs):
        if not hasattr(x, "items"):
            x = self.preprocess(text = x, padding = padding, truncation = truncation, **kwargs)
        input_ids = x["input_ids"].to(self.device)
        attention_mask = x["attention_mask"].to(self.device) if use_attn_mask else None
        with torch.no_grad():
            if self.clip:
                hidden_state = self.model.text_model(output_hidden_states = True, input_ids = input_ids, attention_mask = attention_mask)["hidden_states"]
                pool, hidden_state = hidden_state[skip_pool if skip_pool is not None else skip], hidden_state[skip]
                hidden_state = self.normalize_text_hidden_state(hidden_state) if normalize else hidden_state
            else:
                hidden_state = self.model(input_ids = input_ids, attention_mask = attention_mask)[0]
                pool = None
        if pooling:
            if self.clip:
                with torch.no_grad():
                    pool = self.pool_text_hidden_state(self.normalize_text_hidden_state(pool) if normalize_pool else pool, x, **kwargs)
            return (hidden_state, pool)
        return hidden_state
    
    def get_text_feature(self, x, ref_x = None, weight = None, alpha = 0.3, skip = -1, batch_size = 64, padding = "max_length", truncation = True, use_attn_mask = False, **kwargs):
        if not self.clip:
            raise TypeError("T5 encoder does not support this function (get_text_feature).")
        with torch.no_grad():
            pool_hidden_state = self(x, ref_x, weight = weight, alpha = alpha, pooling = True, skip_pool = skip, batch_size = batch_size, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize_pool = True, **kwargs)[1]
            result = self.projection_text_hidden_state(pool_hidden_state)
        return result
    
    def get_image_feature(self, x, return_tensors = "pt", **kwargs):
        if not self.clip:
            raise TypeError("T5 encoder does not support this function (get_image_feature).")
        if hasattr(x, "items"):
            x = x["pixel_values"]
        elif not torch.is_tensor(x):
            x = self.preprocess(image = x, return_tensors = return_tensors, **kwargs)["pixel_values"]
        with torch.no_grad():
            result = self.model.get_image_features(pixel_values = x.to(self.device))
        return result
    
    def encode_context(self, ref_x, pooling = False, skip = -1, skip_pool = None, batch_size = 64, padding = "max_length", truncation = True, use_attn_mask = False, normalize = False, normalize_pool = False, **kwargs):
        if not hasattr(ref_x, "items"):
            if np.ndim(ref_x) == 0:
                ref_x = [[ref_x]]
            elif np.ndim(ref_x) == 1:
                ref_x = [ref_x]
            b, ref_size = len(ref_x), len(ref_x[0])
            ref_x = np.reshape(ref_x, [b * ref_size])
            ref_x = self.preprocess(text = list(ref_x), padding = padding, truncation = truncation, **kwargs)
            ref_x = {k:v for k, v in ref_x.items() if k in (["input_ids", "attention_mask"] if use_attn_mask else ["input_ids"])}
        else:
            b, ref_size = ref_x["input_ids"].shape[:2]
            ref_x = {k:v.view(b * ref_size, -1) for k, v in ref_x.items() if k in (["input_ids", "attention_mask"] if use_attn_mask else ["input_ids"])}
        hidden_state, pool_hidden_state = [], []
        batch_indices = [(i * batch_size, min((b * ref_size), (i + 1) * batch_size)) for i in range(int(np.ceil((b * ref_size) / batch_size)))]
        for start, end in batch_indices:
            h, p = self.encode_prompt({k:v[start:end] for k, v in ref_x.items()}, pooling = True, skip = skip, skip_pool = skip_pool, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = normalize, normalize_pool = normalize_pool, **kwargs)
            hidden_state.append(h)
            if p is not None:
                pool_hidden_state.append(p)
        hidden_state = torch.cat(hidden_state, dim = 0) if 1 < len(hidden_state) else hidden_state[0]
        pool_hidden_state = torch.cat(pool_hidden_state, dim = 0) if 1 < len(pool_hidden_state) else (pool_hidden_state[0] if len(pool_hidden_state) == 1 else None)
        with torch.no_grad():
            hidden_state = hidden_state.view(b, ref_size * hidden_state.shape[1], -1)
            if pooling:
                if self.clip:
                    pool_hidden_state = pool_hidden_state.view(b, ref_size, -1)
                hidden_state = (hidden_state, pool_hidden_state)
        return hidden_state
    
    def __call__(self, x, ref_x = None, weight = None, alpha = 0.3, pooling = True, skip = -1, skip_pool = None, batch_size = 64, padding = "max_length", truncation = True, use_attn_mask = False, normalize = True, normalize_pool = True, training = False, **kwargs):
        if ref_x is not None or training:
            if training:
                context = weight = None
            else:
                _context, _context_pool = self.encode_context(ref_x, pooling = True, skip = skip, skip_pool = None, batch_size = batch_size, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = False, **kwargs)
                if weight is not None:
                    if not torch.is_tensor(weight):
                        weight = torch.tensor(weight)
                    if weight.dim() == 0:
                        weight = weight.unsqueeze(0).unsqueeze(0)
                    elif weight.dim() == 1:
                        weight = weight.unsqueeze(0)
                    weight = weight.to(self.device)
                else:
                    weight = torch.ones((1, _context.shape[1] // self.n_token), dtype = torch.float32, device = _context.device)
                context = _context
                del _context, _context_pool
            result = self.encode_personalized_prompt(x, context, weight = weight, alpha = alpha, pooling = pooling, skip = skip, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = normalize, normalize_pool = normalize_pool, **kwargs)
            return result
        else:
            return self.encode_prompt(x, pooling = pooling, skip = skip, skip_pool = skip_pool, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = normalize, normalize_pool = normalize_pool, **kwargs)
    
    def encode_personalized_prompt(self, x, context = None, weight = None, alpha = 0.3, pooling = True, skip = -1, padding = "max_length", truncation = True, use_attn_mask = False, normalize = True, normalize_pool = True, **kwargs):
        if not torch.is_tensor(x):
            if not hasattr(x, "items"):
                x = self.preprocess(text = x, padding = padding, truncation = truncation, **kwargs)
            x = self.encode_prompt(x, pooling = False, skip = skip, skip_pool = None, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = False, **kwargs)
        if context is None:
            context = x
        else:
            batch_size, n_token = x.shape[:2]
            if context.shape[0] == 1 and batch_size != 1:
                context = context.repeat_interleave(batch_size, dim = 0)
                if weight is not None and weight.shape[0] == 1:
                    weight = weight.repeat_interleave(batch_size, dim = 0)
            context_size = context.shape[1]
            context = torch.cat([x, context], dim = 1)
            if weight is not None:
                extra_weight = torch.ones((batch_size, n_token), dtype = torch.float32, device = weight.device)
                weight = torch.cat([extra_weight, weight], dim = 1)
        hidden_state, pool = self.adapter(context, weight = weight, alpha = alpha)
        hidden_state = self.normalize_text_hidden_state(hidden_state) if normalize else hidden_state
        if pooling:
            pool = self.normalize_text_hidden_state(pool) if normalize_pool else pool
            return (hidden_state, pool)
        return hidden_state
    
    def to(self, device):
        self.model.to(device)
        self.adapter.to(device)
        self.device = device
        return self
        
    def eval(self):
        self.model.eval()
        if self.clip and hasattr(self.model, "text_projection"):
            self.model.text_model.final_layer_norm.requires_grad_(False)
            self.model.text_projection.requires_grad_(False)
        self.adapter.eval()
        return self
        
    def train(self):
        self.model.eval()
        if self.clip and hasattr(self.model, "text_projection"):
            self.model.text_model.final_layer_norm.requires_grad_(False)
            self.model.text_projection.requires_grad_(False)
        self.adapter.train()
        return self
        
    def parameters(self):
        return list(self.adapter.parameters())
    
    def named_parameters(self):
        return list(self.adapter.named_parameters())