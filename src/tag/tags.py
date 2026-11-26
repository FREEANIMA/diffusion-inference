import torch
import pandas as pd
import json

from src.network import LLM
from src.storage import download_model, load_llm_model
from src.core import state, log

from .utils import build_tag_maps, unk_separation, mapping

class TagGenration:
    def __init__(self, console=None):
        self.MIN_FIXED_TAGS = 5

        self.NAS = state.config.NAS
        self.init_prompt = state.config.init_prompt

        download_model(config=state.config, filename="tags_llm.pt")
        type_path = download_model(config=state.config, filename="tags_type.parquet")
        vocab_path = download_model(config=state.config, filename="vocab.json")    
        print(vocab_path)    

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)
        self.model = load_llm_model(config=state.config, network=LLM, vocab_size=vocab_size, console=console)

        tags_type_df = pd.read_parquet(type_path)
        self.type_map, self.dup_map, self.tag_tree = build_tag_maps(tags_type_df=tags_type_df)

        self.PAD_ID = self.vocab["<pad>"]
        self.CLS_ID = self.vocab["<cls>"]
        self.EOS_ID = self.vocab["<eos>"]
        self.UNK_ID = self.vocab["<unk>"]

        self.id_to_tag = {v: k for k, v in self.vocab.items()}
        self.console = console

    def to_ids(self, xs, max_len=256):
        body = []
        for t in xs:    
            tid = self.vocab.get(t)
            if tid is None:
                continue
            body.append(tid)
            if len(body) >= max_len - 1:
                break
        return [self.CLS_ID] + body

    @torch.inference_mode()
    @torch.amp.autocast(device_type=state.config.device, dtype=state.config.torch_dtype)
    def one_step(self, idx, past=None, step=0):
        logits, past = self.model(x=idx, attn_mask=None, past_kv=past, use_cache=True)
        next_logits = logits[:, -1, :].float() / max(state.config.temperature, 1e-5)

        for bad in (self.PAD_ID, self.CLS_ID, self.UNK_ID):
            next_logits[:, bad] = float("-inf")
        if step < self.MIN_FIXED_TAGS:
            next_logits[:, self.EOS_ID] = float("-inf")

        if state.config.top_k and state.config.top_k > 0:
            kth = torch.topk(next_logits, min(state.config.top_k, next_logits.size(-1))).values[:, -1:]
            next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))

        if state.config.top_p is not None and state.config.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            remove = probs.cumsum(dim=-1) - probs >state.config.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        return next_id, past

    @torch.inference_mode()
    def inference(self, ids):
        generated = []

        next_id, past = self.one_step(ids, past=None, step=0)

        for step in range(state.config.max_words):
            tok = next_id.item()
            if tok == self.EOS_ID:
                break
            generated.append(tok)

            next_id, past = self.one_step(next_id, past=past, step=step + 1)

        return [self.id_to_tag[i] for i in generated if i in self.id_to_tag]
    
    def generation(self, tags, nsfw, H, W, inferenc_mode=False):
        if isinstance(nsfw, bool):
            nsfw = [nsfw] * len(tags)

        tags_list = [
            [x.strip() for x in tag.split(",") if x.strip()]
            for tag in tags
        ]
        unks_list, dups_list = unk_separation(self.dup_map, tags_list)
        
        short = set(state.config.NAS_SHORT)
        attempts = max(1, state.config.retry)

        def _types(tags):
            return {self.type_map[t] for t in tags if t in self.type_map}

        generation_list = [list(d) for d in dups_list]
        pending = list(range(len(dups_list)))
        use_inference = inferenc_mode

        for i in range(attempts):
            for idx in pending:
                dups = dups_list[idx]
                if use_inference:
                    ids = torch.tensor([self.to_ids(dups)], dtype=torch.long, device=state.config.device)
                    gen = self.inference(ids=ids)
                    generation_list[idx] = list(dict.fromkeys(dups + gen))
                else:
                    generation_list[idx] = dups

            pending = [idx for idx in pending if not short <= _types(generation_list[idx])]
            if not pending:
                break

            use_inference = True 
            missing = {idx: short - _types(generation_list[idx]) for idx in pending}
            log(self.console, f"tags generation retry {i + 1} / {attempts} missing: {missing}")


        nsfw_text_list = [f"nsfw: {str(n).lower()}; " for n in nsfw]
        size_text_list = [f"size: {H}x{W}; " for _ in nsfw]
        init_text_list = [state.config.init_prompt for _ in nsfw]

        return mapping(generation_list, nsfw_text_list, size_text_list, init_text_list, unks_list, self.NAS, self.type_map)