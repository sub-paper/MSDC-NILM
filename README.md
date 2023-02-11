# MSDC-NILM
Our paper MSDC: Exploiting Multi-State Power Consumption in Non-intrusive Load Monitoring based on A Dual-CNN is accepted by AAAI 2023. The pre-print version is uploaded (MSDC-preprint.pdf)
authors: Jialing He 1, Jiamou Liu 2, Zijian Zhang 3, Yang Chen 2,  Yiwei Liu 4,  Bakh Khoussainov 5, Liehuang Zhu 3.
affiliations: 1 College of Computer Science, Chongqing University, Chongqing, China, 400044.
              2 School of Computer Science, The University of Auckland, Auckland 1142, New Zealand.
              3 School of Cyberspace Science and Technology, Beijing Institute of Technology, Beijing, China, 100081.
              4 Defence Industry Secrecy Examination and Certification Center, Beijing, China, 100089.
              5 School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, China, 611731.
emails: hejialing@cqu.edu.cn, jiamou.liu, yang.cheng@auckland.ac.nz, zhangzijian, liehuangz@bit.edu.cn, yiweiliu disecc@163.com, bmk@uestc.edu.cn

Code:
%run my_s2s_crf.py --appliance_name dishwasher --data_dir /dishwasher/ --batch_size 256 --n_epoch 10 --patience 5 --seed 776

You can change ‘dishwasher’ to other appliances like microwave, wachingmachine, fridge etc. 
