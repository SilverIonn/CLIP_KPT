from yacs.config import CfgNode
from torch.utils.data.dataset import Dataset
from typing import *
import torch
from tqdm import tqdm
from clip import clip


def cal_logits(clip_model, verbalizer, dataloader, temp):
    r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
    
    Args:
        prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
        dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
    
    Return:
        (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
    """
    img_ft_list = []
    clip_model.eval()
    
    extended_classnames = [word for words_per_label in verbalizer.label_words for word in words_per_label]
    prompts = [temp.format(c.replace("_", " ")) for c in extended_classnames]
    # print(f"Prompts: {prompts}")
    prompts = torch.cat([clip.tokenize(p) for p in prompts])
    prompts = prompts.to('cuda')
    with torch.no_grad():
        text_features = clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) #[num of extended label words * feature size]
    print(f'cali text features shape: {text_features.shape}')

    
    for batch in tqdm(dataloader,desc='ContextCali'):
        #########cuda? batch to cuda?
        # batch = batch.to(clip_model.device)
        image = batch['img']
        image = image.to('cuda')
        with torch.no_grad():
            img_ft = clip_model.encode_image(image)
            img_ft = img_ft / img_ft.norm(dim=-1, keepdim=True)
            img_ft_list.append(img_ft)
            ############detach() or not
        
    image_features = torch.cat(img_ft_list, dim=0)  #[200 * features]
    
    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale *image_features @ text_features.t() # [200 * num of extended label words]
        
    return logits




def register_calibrate_logits(logits, verbalizer, frac):
    r"""For Knowledgeable Verbalizer, it's nessessory to filter the words with has low prior probability.
    Therefore we re-compute the label words after register calibration logits.
    """
    if logits.requires_grad:
        logits = logits.detach()
    calibrate_logits = logits
    print(f'register_calibrate_logits starting shape: {logits.shape}')
    # cur_label_words_ids = self.label_words_ids.data.cpu().tolist()
    #############cpu?
    rm_calibrate_ids = set(torch.argsort(calibrate_logits)[:int(frac*logits.shape[-1])].cpu().tolist())
    print(f'logits shape -1 :{logits.shape[-1]}')
    print(f'rm_calibrate_ids: {rm_calibrate_ids}, shape {len(rm_calibrate_ids)}')
    
    old_label_words = verbalizer.label_words
    
    assert sum([len(old_label_words[i]) for i in range(len(old_label_words))]) == len(calibrate_logits), f'before not match! {sum([len(old_label_words[i]) for i in range(len(old_label_words))])}!={len(calibrate_logits)}'

    new_label_words = []
    ovr_idx = 0
    for i_label, words_per_label in enumerate(old_label_words):
        # new_label_words.append([])
        new_row = []
        for j_word, word in enumerate(words_per_label):
            if j_word == 0:
                # new_label_words[-1].append(word)
                new_row.append(word)
                rm_calibrate_ids = rm_calibrate_ids.difference(set([ovr_idx]))
            elif ovr_idx not in rm_calibrate_ids:
                # new_label_words[-1].append(word)
                new_row.append(word)
            ovr_idx += 1
        new_label_words.append(new_row)
    
    for row in old_label_words:
        assert len(row) == len(set(row)), f"{row}"
    
    # flatten2d = lambda arr: [elem for row in arr for elem in row]
    # arr = flatten2d(old_label_words)
    # assert len(arr) == len(set(arr)), "AAAAAAAA"
    # raise NotImplementedError
                 
    verbalizer.label_words = new_label_words
    
    # new_calibrate_logits = [logit for i, logit in enumerate(calibrate_logits) if i not in rm_calibrate_ids]
    # src = len(new_calibrate_logits)
    # des = sum(len(row) for row in new_label_words)
    # assert src == des, "src: {}, des: {}".format(src, des)
    
    new_calibrate_logits = [calibrate_logits[i] for i in range(len(calibrate_logits)) if i not in rm_calibrate_ids]
    assert sum([len(new_label_words[i]) for i in range(len(new_label_words))]) == len(new_calibrate_logits), f'final not match! {sum([len(new_label_words[i]) for i in range(len(new_label_words))])}!={len(new_calibrate_logits)}'
    
    return new_calibrate_logits
    ########device?
    # self.to(calibrate_logits.device)




