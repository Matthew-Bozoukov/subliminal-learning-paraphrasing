import torch
import os
import pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import copy
import einops

class OPORP():
    """
    This class is used to compress vectors using the OPORP algorithm
    Attributes:
        shuffle_lambda (int): the number of times to shuffle the vector
        filepath (str): the path to the file to save the compressed vectors
        device (str): the device to use for the compressed vectors
        seed (int): the seed to use for generating the random matrices
        K (int): the target dimension of the vector after compression
    """
    def __init__(self, shuffle_lambda, filepath, device, K, seed=42):
        self.filepath = filepath
        self.device = device
        self.seed = seed
        self.is_init = False
        self.K = K
        self.shuffle_lambda = shuffle_lambda
        
        self._D = None
        self._random_mat = None
        self._perm_mat_list = []
        self._perm_dim_list = []
        

    def __call__(self, vec):
        vec = self._pad(vec)
        if self.is_init == False:
            print("Creating random and shuffling matrices. It may take a few minutes.")
            self._init()
        assert self._D % self.K == 0, f"{self._D=}, {self.K=}, the dimension of the vector is not a multiple of target dimension"
        for i, (dim, perm_mat) in enumerate(zip(self._perm_dim_list, self._perm_mat_list)):
            if i%2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        
        vec = vec.reshape((-1))
        vec = vec*self._random_mat
        step = self._D//self.K
        vec = torch.sum(vec.reshape((-1, step)), dim=1)
        return vec

    def _init(self):
        self.is_init = True
        np.random.seed(self.seed)
        self.file_name = os.path.join(
            self.filepath,
            f"RapidGrad_D{self._D}_n{self.shuffle_lambda}_seed{self.seed}.obj"
        )
        if not self._load():
            self._create_random_mat()
            self._create_perm_mat()
            self._save()
        self._random_mat = torch.from_numpy(self._random_mat).to(dtype=torch.float16).to(self.device)

    def _create_random_mat(self):
        self._random_mat = np.random.randint(0, 2, (self._D,), dtype=np.int8)
        self._random_mat[self._random_mat < 1e-8] = -1

    def _create_perm_mat(self):
        lt = []
        D = int(self._D)
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D // i
                    break
        for _ in tqdm(range(self.shuffle_lambda)):
            x = np.random.randint(len(lt)//4, len(lt)//2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self._perm_dim_list.append(dim)
            self._perm_mat_list.append(np.random.permutation(dim))

    def _save(self):
        if os.path.exists(self.file_name):
            return
        with open(self.file_name, 'wb') as f:
            pickle.dump(self, f)

    def _load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, 'rb') as f:
            new_obj = pickle.load(f)
        device = self.device
        self.__dict__ = copy(new_obj.__dict__)
        self.device = device
        return True

    def _pad(self, x):
        """
        Pad the given vector to the nearest multiple of K &
        Stores the dimenstion of the padded vector in self.D (original vector's dimension)
        """
        self._D = ((len(x)- 1)//self.K + 1)*self.K
        x = F.pad(x, (0, self._D - len(x)), "constant", 0)
        return x

class InfluenceEngine:
    def __init__(self,
        max_length,
        tokenizer,
        target_model,
        compressor,
        device,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.compressor = compressor
        self.device = device
        self.avg_val_grad = None

        for k, v in self.target_model.named_parameters():
            if 'lora' in k:
                v.requires_grad = True

    def compute_avg_val_grad(self, prompts: list[str], completions: list[str]):
        """
        This function is used to compute the average gradients of the target model on the validation set.
        """
        tokenized_list = self.create_tokenized(prompts, completions)
        grads = []
        for tokenized in tqdm(tokenized_list, desc="Computing average validation gradients"):
            self.target_model.zero_grad()
            loss = self.target_model(**tokenized).loss
            loss.backward()
            del tokenized
            grad_list = []
            for k, v in self.target_model.named_parameters():
                if 'lora_A' in k or 'lora_B' in k:
                    grad_list.append(v.grad.reshape(-1))
                else:
                    pass
            vec = torch.cat(grad_list)
            compressed_vec = self.compressor(vec)
            grads.append(compressed_vec)
            del grad_list, vec, compressed_vec
        # compute the average of the validation gradietns
        self.avg_val_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
        print(f"Average validation gradients: {self.avg_val_grad.shape}, mean: {self.avg_val_grad.mean()}, std: {self.avg_val_grad.std()}")
        del grads

    def _compute_grads(self, prompts: list[str], completions: list[str]):
        """
        This function is used to compute the gradients of the target model.
        It return the compressed gradients.

        Inputs:
            prompts: list[str]
            completions: list[str]
        Output:
            List of compressed gradients.
        """
        tokenized_list = self.create_tokenized(prompts, completions)
        decoded_example = self.tokenizer.batch_decode(tokenized_list[0]['input_ids'], skip_special_tokens=True)
        output = torch.empty((len(tokenized_list), self.compressor.K), device=self.device)
        for i, tokenized in tqdm(enumerate(tokenized_list), total=len(tokenized_list), desc="Computing gradients for training set"):
            self.target_model.zero_grad()
            loss = self.target_model(**tokenized).loss
            loss.backward()
            del tokenized
            grad_list = []
            for k, v in self.target_model.named_parameters():
                if 'lora_A' in k or 'lora_B' in k:
                    grad_list.append(v.grad.reshape(-1))
                else:
                    pass
            vec = torch.cat(grad_list)
            compressed_vec = self.compressor(vec)
            output[i] = compressed_vec
            del grad_list, vec, compressed_vec
        return output # (batch_size, K)

    def compute_influence_simple(self, prompts: list[str], completions: list[str]):
        train_grads = self._compute_grads(prompts, completions)
        print(f"{train_grads.shape=}")
        train_grads = train_grads / (torch.linalg.norm(train_grads) + 1e-8)
        val_grad = self.avg_val_grad
        val_grad = val_grad / (torch.linalg.norm(val_grad) + 1e-8)
        lam = train_grads.pow(2).mean() / 10.0
        train_dot = torch.matmul(train_grads, train_grads.T)
        val_dot = torch.matmul(train_grads, val_grad)
        diag = train_dot.diag()
        influence = -1 / lam * val_dot
        correction_term = ((train_dot * val_dot.unsqueeze(0)) / (lam + diag).unsqueeze(0)).mean(dim=1) / lam
        influence += correction_term

        print(f"Influence: mean {influence.mean()}, std {influence.std()}, max {influence.max()}, min {influence.min()}")
        return influence.cpu().tolist()


        
    def create_tokenized(self, prompts: list[str], completions: list[str]):
        """
        This function is used to create the list of input_ids, attention_masks, and labels to pass to the model.
        
        Inputs:
            prompts: list[str]
            completions: list[str]
        Output:
            List of dictionaries, each with keys: "input_ids", "attention_mask", "labels"
                input_ids: list[int]
                attention_mask: list[int]
                labels: list[int]
        """
        
        encodings = []  # match the padding length

        for prompt, completion in zip(prompts, completions):
            messages = []
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": completion})

            input = self.tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized_input = self.tokenizer(input, return_tensors="pt").to(self.device)

            # Compute the prompt token length (everything before assistant content)
            prompt_only_messages = [{"role": "user", "content": prompt}]
            prompt_only_text = self.tokenizer.apply_chat_template(
                prompt_only_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_only_tokenized = self.tokenizer(prompt_only_text, return_tensors="pt").to(self.device)
            prompt_len = prompt_only_tokenized.input_ids.shape[1]

            encoding = {
                "input_ids": tokenized_input.input_ids,
                "attention_mask": tokenized_input.attention_mask,
                # Provide labels so that the model computes a loss. For a minimal fix,
                # compute loss over the whole sequence by setting labels = input_ids.
                # If you want to ignore prompt tokens, mask them with -100 instead.
                "labels": tokenized_input.input_ids.clone(),
            }
            # Ignore the prompt (and assistant prefix) tokens in the loss
            encoding["labels"][:, :prompt_len] = -100
            encodings.append(encoding)
            del tokenized_input
            del prompt_only_tokenized

        return encodings
