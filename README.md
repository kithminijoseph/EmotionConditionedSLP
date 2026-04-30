The following repository contains the files used to create an emotion conditioned sign language production pipeline, which outputs 3D FLAME avatars. The raw clips from Boston ASLLRP's corpus have been cropped and uploaded here, however it should be noted these clips are the intellectual property of Boston University. 

Additionally clip annotations from ASLLRP and emotion labels from the EmoSign dataset have been parsed here into `merged_records.jsonl` for ease, but these can be found on HuggingFace and ASLLRP. ASLLRP Data Access Interface: http://www.bu.edu/asllrp/. EmoSign: https://huggingface.co/datasets/catfang/emosign.

To extract MediaPipe poses from the clips and map them to FLAME parameters, please run `mediapipe_to_flame_extractor.py`. To do this you will also need to clone an external repository https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame to get the mappings. The work in that repository is the intellectual property of Yan (2024) and is referenced below.

Alternatively, the extracted params can be found in the folder `flame_params_for_clips`. 

To train on these params please run `flame_generator.py` with the `train` mode. You can train on the `full` 199 sample dataset if you would like to test with novel inputs or on a train `split` (160 samples) if you want to test with the heldout test set. 

Afterwards, to run the inference with the heldout set, run on `reconstruct` mode. Alternatively to supply novel inputs, run on `generate` mode. Sample commands for these are all within the file.

Once you run an inference or novel generation you will generate a `.npz` file. This file needs to be run with `flame_renderer.py` to generate a `mp4` video that shows the 3D avatar. 

Additionally, files used for hyperparameter tuning: `hyperparam_tuning.py` and the ablation comparison overall: `ablation_comparison.py` and per sample `per_sample_ablation_comparison.py` are given below as well as the folders with the results and k fold splits for the ablation.

It is advised to train this model on Hex or a GPU.

## References

If you use this repository, please cite the original datasets and resources:

```bibtex

@techreport{neidle2020asllrp,
  title = {A User's Guide to the American Sign Language Linguistic Research Project (ASLLRP) Data Access Interface (DAI) 2 -- Version 2},
  author = {Neidle, Carol and Opoku, Augustine},
  year = {2020},
  number = {Report No. 18},
  institution = {American Sign Language Linguistic Research Project, Boston University},
  url = {http://www.bu.edu/asllrp/rpt18/asllrpr18.pdf}
}

@inproceedings{neidle2012web,
  title = {A New Web Interface to Facilitate Access to Corpora: Development of the ASLLRP Data Access Interface},
  author = {Neidle, Carol and Vogler, Christian},
  booktitle = {Proceedings of the 5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon},
  year = {2012}
}

@article{chua2025emosign,
  title = {EmoSign: A Multimodal Dataset for Understanding Emotions in American Sign Language},
  author = {Chua, Phoebe and Fang, Cathy Mengying and Ohkawa, Takehiko and Kushalnagar, Raja and Nanayakkara, Suranga and Maes, Pattie},
  journal = {arXiv preprint arXiv:2505.17090},
  year = {2025},
  url = {https://arxiv.org/abs/2505.17090}
}

@misc{yan2024mediapipe2flame,
  title = {Mediapipe-Blendshapes-to-FLAME},
  author = {Yan, Peizhi},
  year = {2024},
  howpublished = {\url{https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame}},
  note = {GitHub repository}
}