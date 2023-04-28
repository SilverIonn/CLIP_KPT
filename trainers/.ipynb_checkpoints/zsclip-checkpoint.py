import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

from utils import gen_labelwords, FewShotSampler, cal_logits, register_calibrate_logits

from verbalizers import KnowledgeableVerbalizer
from dassl.data.data_manager import build_data_loader, DatasetWrapper
from dassl.data.transforms import build_transform

import torch.nn.functional as F

from tqdm import tqdm
import numpy as np


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model


@TRAINER_REGISTRY.register()
class ZeroshotCLIP3(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        
        # generate label words
        
        # if cfg.DATASET.NAME == 'EuroSAT':
        #     gen_labelwords(classnames,['Annual Crop', 'Forest', ['Herbaceous','Vegetation'], ['Road','Highway'], 'Industrial Buildings', 'Pasture', 'Permanent Crop', 'Residential Buildings', 'River', ['Sea','Lake']],cfg)
        # else:
        #     gen_labelwords(classnames,classnames,cfg)
        
        
        gen_labelwords(classnames,classnames,cfg)
        
        # gen_labelwords(classnames,cfg)
    
        ######parameters
        self.verbalizer = KnowledgeableVerbalizer(len(self.dm.dataset.classnames)).from_file(f'{cfg.VERBALIZER.DIR}/knowledgeable_verbalizer.txt')
        
        if cfg.CALIBRATION:
            support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
            seed = 144
            self.dm.dataset.support = support_sampler(self.dm.dataset.train_x, seed=seed)
            # for example in dataset['support']:
            #     example.label = -1
            tfm_test =  build_transform(cfg, is_train=False)
            support_dataloader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dm.dataset.support,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=DatasetWrapper
            )
            print(f'Load support dataloader successfully! size: {len(self.dm.dataset.support)}')
            
            org_label_words_num = [len(self.verbalizer.label_words[i]) for i in range(len(classnames))]
            print(f'num of org label words: {sum(org_label_words_num)}')
            assert len(self.verbalizer.label_words)==len(classnames), 'label num not matching!'
            
            # calculate the calibration logits
            self.cc_logits = cal_logits(clip_model, self.verbalizer, support_dataloader, temp)  # [200 * num of extended label words]
                                  
            print("the calibration logits is", self.cc_logits)
            print(f'cc_logits shape: {self.cc_logits.shape}')
            print("Phase 1 {}\n".format(org_label_words_num))

            self.cali_logits = register_calibrate_logits(self.cc_logits.mean(dim=0), self.verbalizer, frac = cfg.CALIBRATION_CUT)
            self.cali_logits = torch.tensor(self.cali_logits).to(self.device)
            
            
            new_label_words_num = [len(self.verbalizer.label_words[i]) for i in range(len(classnames))]
            assert len(self.verbalizer.label_words)==len(classnames), 'label num not matching!'
            print("Phase 2 {}\n".format(new_label_words_num))
            print(f'num of label words after cc: {sum(new_label_words_num)}')
            print(f'cali_logits shape: {self.cali_logits.shape}')
            
            kb_file = f'{cfg.VERBALIZER.DIR}/knowledgeable_verbalizer_2.txt'
            with open(kb_file, 'w') as f:
                for line in self.verbalizer.label_words:
                    for word in line[:-1]:
                        f.write(f"{word},")
                    f.write(f"{line[-1]}\n")
            

        
        # if cfg.RELEVANCE_REFINE:
        #     if args.filter == "tfidf_filter":
        #         tfidf_filter(self.verbalizer, self.cc_logits, classnames)
        #     elif args.filter == "none":
        #         pass
        #     else:
        #         raise NotImplementedError
            
        labelwords = [word for words_per_label in self.verbalizer.label_words for word in words_per_label]
        prompts = [temp.format(c.replace("_", " ")) for c in labelwords]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        print(f'text feature size: {self.text_features.shape}')
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits       # [batchsize * num of label words]
    
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        #########change evaluator
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)         # output : [batchsize * num of label words]
            output = self.process_logits(output).to(self.device)    #########cuda?  # output : [batchsize * num of labels]
            
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # for k, v in results.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def process_logits(self, logits):    # [batchsize * num of label words]
        cfg = self.cfg
        label_words_probs = F.softmax(logits,dim=-1)
        if cfg.CALIBRATION:
            if  hasattr(self, "cali_logits") and self.cali_logits is not None:
                assert self.cali_logits.shape[0] == label_words_probs.shape[1], f'process_logits shape not match! {self.cali_logits.shape[0]}!={label_words_probs.shape[1]}'
                calibrate_label_words_probs = F.softmax(self.cali_logits.unsqueeze(0),dim=-1)
                assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] and calibrate_label_words_probs.shape[0]==1, "shape not match"
                label_words_probs /= (calibrate_label_words_probs+1e-15)
                norm = label_words_probs.sum(dim=-1,keepdim=True) 
                label_words_probs /= norm
                label_words_probs = torch.log(label_words_probs+1e-15)
        assert len(self.verbalizer.label_words)==len(self.dm.dataset.classnames), 'process logits shape not match'
        
        
        label_words = self.verbalizer.label_words
        label_words_num = [len(label_words[i]) for i in range(len(label_words))]
        
        # print(f'process logits num of label words: {sum(label_words_num)}')
        
        starting_idx = 0
        probs = []
        
        for i_label, words_per_label in enumerate(label_words):
            probs.append(label_words_probs[ : ,starting_idx:starting_idx+label_words_num[i_label]].mean(dim=1, keepdims=True))   
            starting_idx += label_words_num[i_label]
            
        label_probs = torch.cat(probs, dim=1)
        # print(f'label_probs shape: {label_probs.shape}')
        # print(f'sum?1 label_probs: {torch.sum(label_probs,dim=1)}')
        return label_probs
    

        
        