# EENDEDA_VIB

This is the official implementation for the paper:

**[Do End-to-End Neural Diarization Attractors Need to Encode Speaker Characteristic Information?](https://arxiv.org/abs/2402.19325)**

Lin Zhang, Themos Stafylakis, Federico Landini, Mireia Diez, Anna Silnova, Lukáš Burget

---

Please cite this paper if this repo is useful for you:

```latex
@inproceedings{zhang24_odyssey,
  author={Lin Zhang, Themos Stafylakis, Federico Landini, Mireia Diez, Anna Silnova, Lukáš Burget},
  title={{Do End-to-End Neural Diarization Attractors Need to Encode Speaker Characteristic Information?
}},
  year=2024,
  booktitle={Proc. The Speaker and Language Recognition Workshop (Odyssey 2024)},
  pages={},
  doi={}
}
```



## Folder Structure

```shell

EENDEDA_VIB
├── eendedavib            # The main implementation for EEND-EDA with VIB
├── example               # An example including YAML files and top-level scripts.
├── train.sh              # Scripts {train, adap, infer}.sh are  used to pass  \
├── adap.sh               #    custom parameters to YAML files and call corresponding \
├── infer.sh              #    python scripts ../{train, adap, infer}.py.
├── parse_options.sh      # Script that helps to assign attributes, copied from Kaldi
├── requirements.txt
├── README.md
└── LICENSE
```



Specificly for the example folder: `EENDEDA_VIB/example`. It includes several top-level scripts and yaml config files. Please see more information in those scripts.

* Train: `01_train.sh` and `train.yaml` 
* Adap: `02_adap.sh` and `adap.yaml `
* Finetune: `03_ft.sh` and `ft.yaml`
* Inference: `04_inf.sh`and  `infer_callhome_part1_2spk.yaml`



## Reference Repositories

[EEND in PyTorch implemented by BUT](https://github.com/BUTSpeechFIT/EEND)



## Acknowledgement

The work was supported by the Czech Ministry of Interior project No. VJ01010108 ``ROZKAZ'', Horizon 2020 Marie Sklodowska-Curie grant ESPERANTO, No. 101007666, SOKENDAI Student Dispatch Program, and Japan Science and Technology Agency Grants JPMJFS2136. Computing on IT4I supercomputer was supported by the Czech Ministry of Education, Youth and Sports through the e-INFRA CZ (IDs 90140 and 90254).



## License

This project is mainly licensed under the MIT License.



## Contact

If you have any comment or question, please contact [lzhang.as@gmail.com](mailto:lzhang.as@gmail.com)



